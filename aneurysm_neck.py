"""Provide infrastructure to calculate the aneurysm neck.

The module provides a function to compute the aneurysm neck plane,
as defined by Piccinelli et al. (2009).
"""

import vtk
import numpy as np

from vmtk import vtkvmtk
from vmtk import vmtkscripts
from scipy import interpolate

from vtk.numpy_interface import dataset_adapter as dsa

# Local modules
from constants import *
import polydatatools as tools 

# Flags
_write   = False
_debug   = False
_inspect = False

_write_dir = '/home/iagolessa/'

INCR = 0.01 
HUGE = 1e30

_dimensions = intThree
_xComp = intZero
_yComp = intOne
_zComp = intThree


def _id_min_dist_to_point(point, polydata):
    """Get ID os closest point in vtkPolyData.
    
    Get the id of a point in an array which 
    is the closest from a point given by the 
    user that does not necessarely belongs 
    to the array.
    """
    
    # Computes distance vector array between point and array
    locator = vtk.vtkPointLocator()
    locator.SetDataSet(polydata)
    locator.BuildLocator()    
    
    return locator.FindClosestPoint(point)


def _vtk_vertices_to_numpy(polydata):
    """Convert vtkPolyData object to Numpy nd-array.
    
    Return the point coordinates of a vtkPolyData
    object as a n-dimensional Numpy nd-array. It 
    only extracts its points, ignoring any array 
    or field in the vtkPolyData.
    
    Arguments:
    polydata -- the vtkPolyData
    """
    vertices = []
    nPoints  = polydata.GetNumberOfPoints()

    for index in range(nPoints):
        vertex = polydata.GetPoint(index)
        vertices.append(list(vertex))

    return np.array(vertices)


def _rotate3d_matrix(tilt, azim):
    """Rotation matrix traformation for a vector in 3D space."""
    return np.array([[             np.cos(azim),             -np.sin(azim),      intZero],
                     [np.sin(azim)*np.cos(tilt), np.cos(azim)*np.cos(tilt), -np.sin(tilt)],
                     [np.sin(azim)*np.sin(tilt), np.cos(azim)*np.sin(tilt),  np.cos(tilt)]])


def _connected_region(regions,method,closest_point=None):
    """Extract the largest or closest to point patch of a disconnected domain. 
    
    Given a disconnected surface, extract a portion of the surface
    by choosing the largest or closest to point patch."""
    
    triangulator = vtk.vtkTriangleFilter()
    triangulator.SetInputData(tools.cleaner(regions))
    triangulator.Update()

    connectivity = vtk.vtkPolyDataConnectivityFilter()
    connectivity.SetInputData(triangulator.GetOutput())
    
    if method == 'largest':
        connectivity.SetExtractionModeToLargestRegion()
        
    elif method == 'closest':
        connectivity.SetExtractionModeToClosestPointRegion()
        connectivity.SetClosestPoint(closest_point)

    connectivity.Update()

    return connectivity.GetOutput()


def _compute_Voronoi(surface_model):
    """Compute Voronoi diagram of a vascular surface."""
    voronoiDiagram = vmtkscripts.vmtkDelaunayVoronoi()
    voronoiDiagram.Surface = surface_model
    voronoiDiagram.CheckNonManifold = True
    voronoiDiagram.Execute()
    
    return voronoiDiagram.Surface


def _clip_with_plane(surface, plane_center, plane_normal, inside_out=False):
    """
    Clip a surface with a plane defined with
    a point and its normal.
    """
    cutPlane = vtk.vtkPlane()
    cutPlane.SetOrigin(plane_center)
    cutPlane.SetNormal(plane_normal)

    clipSurface = vtk.vtkClipPolyData()
    clipSurface.SetInputData(surface)
    clipSurface.SetClipFunction(cutPlane)

    if _inspect:
        clipSurface.GenerateClipScalarsOn()    
    
    if inside_out:
        clipSurface.InsideOutOn()
    else:
        clipSurface.InsideOutOff()
    
    clipSurface.Update()
    
    return clipSurface.GetOutput()


def _contour_is_closed(contour):
    """Check if contour as vtkPolyData is closed."""
    nVertices = contour.GetNumberOfPoints()
    nEdges    = contour.GetNumberOfCells()
    
    return nVertices == nEdges


def _tube_surface(centerline, smooth=True):
    """Reconstruct tube surface of a given vascular surface.
    
    The tube surface is the maximum tubular structure ins-
    cribed in the vasculature. 
    
    Arguments:
    centerline -- the centerline to compute the tube surface
                  with the radius array.
    
    Keyword arguments:
    smooth -- to smooth tube surface (default True)
    """
    # List to collect objects 
    objects = []
    
    radiusArray   = 'MaximumInscribedSphereRadius'
    boxDimensions = _dimensions*[128]

    # Build tube function of the centerlines
    tubeImage = vmtkscripts.vmtkCenterlineModeller()
    tubeImage.Centerlines      = centerline
    tubeImage.RadiusArrayName  = radiusArray
    tubeImage.SampleDimensions = boxDimensions
    tubeImage.Execute()
    objects.append(tubeImage)

    # Convert tube function to surface
    tubeSurface = vmtkscripts.vmtkMarchingCubes()
    tubeSurface.Image = tubeImage.Image
    tubeSurface.Execute()
    objects.append(tubeSurface)
    
    tube = _connected_region(tubeSurface.Surface,'largest')
    
    if _debug:
        for obj in objects:
            obj.PrintInputMembers()
            obj.PrintOutputMembers()
            
    if _inspect:
        tools.viewSurface(tube)
    
    if smooth:
        return tools.smoothSurface(tube)
    else:
        return tube


def _compute_surfaces_distance(isurface, 
                               rsurface, 
                               array_name='DistanceArray', 
                               signed_array=True):
    """Compute point-wise distance between two surfaces.
    
    Compute distance between a reference
    surface, rsurface, and an input surface, isurface, with 
    the resulting array written in the isurface. 
    """
    
    if signed_array:
        normalsFilter = vtk.vtkPolyDataNormals()
        normalsFilter.SetInputData(rsurface)
        normalsFilter.AutoOrientNormalsOn()
        normalsFilter.SetFlipNormals(False)
        normalsFilter.Update()
        rsurface.GetPointData().SetNormals(
            normalsFilter.GetOutput().GetPointData().GetNormals()
        )
    
    surfaceDistance = vtkvmtk.vtkvmtkSurfaceDistance()
    surfaceDistance.SetInputData(isurface)
    surfaceDistance.SetReferenceSurface(rsurface)
    
    if signed_array:
        surfaceDistance.SetSignedDistanceArrayName(array_name)
    else:
        surfaceDistance.SetDistanceArrayName(array_name)
        
    surfaceDistance.Update()
    
    return surfaceDistance.GetOutput()  


def _clip_aneurysm_Voronoi(VoronoiSurface, tubeSurface):
    """Extract the Voronoi diagram of the aneurysmal portion."""

    # Compute distance between complete Voronoi 
    # and the parent vessel tube surface 
    DistanceArrayName = 'DistanceToTubeArray'
    VoronoiDistance   = _compute_surfaces_distance(
                            VoronoiSurface, 
                            tubeSurface, 
                            array_name=DistanceArrayName
                        )

    # Clip the original voronoi diagram at 
    # the zero distance (intersection)
    VoronoiClipper = vmtkscripts.vmtkSurfaceClipper()
    VoronoiClipper.Surface = VoronoiDistance
    VoronoiClipper.Interactive   = False
    VoronoiClipper.ClipArrayName = DistanceArrayName
    VoronoiClipper.ClipValue     = intZero
    VoronoiClipper.InsideOut     = True
    VoronoiClipper.Execute()

    aneurysmVoronoi = _connected_region(VoronoiClipper.Surface,'largest')
    
    if _inspect:
        tools.viewSurface(aneurysmVoronoi)
        
    return tools.cleaner(aneurysmVoronoi)


def _Voronoi_envelope(Voronoi):
    """Compute the envelope surface of a Voronoi diagram."""

    # List to collect objects 
    objects = list()
    
    radiusArray   = 'MaximumInscribedSphereRadius'

    VoronoiBounds     = Voronoi.GetBounds()
    radiusArrayBounds = Voronoi.GetPointData().GetArray(radiusArray).GetValueRange()
    maxSphereRadius   = radiusArrayBounds[intOne]
    enlargeBoxBounds  = (intFour/intTen)*maxSphereRadius
    
    modelBounds = np.array(VoronoiBounds) + \
                  np.array(_dimensions*[-enlargeBoxBounds, enlargeBoxBounds])
    
    # Building the envelope image function
    envelopeFunction = vmtkscripts.vmtkPolyBallModeller()
    envelopeFunction.Surface = Voronoi
    envelopeFunction.RadiusArrayName = radiusArray
    envelopeFunction.ModelBounds     = list(modelBounds)
    envelopeFunction.Execute()
    objects.append(envelopeFunction)
    
    # Get level zero surface
    envelopeSurface = vmtkscripts.vmtkMarchingCubes()
    envelopeSurface.Image = envelopeFunction.Image
    envelopeSurface.Execute()
    objects.append(envelopeSurface)

    envelope = _connected_region(envelopeSurface.Surface, 'largest')
    
    if _debug:
        for obj in objects:
            obj.PrintInputMembers()
            obj.PrintOutputMembers()
            
    if _inspect:
        tools.viewSurface(envelope)
        
    return tools.smoothSurface(envelope)


def _clip_initial_aneurysm(surface_model, aneurysm_envelope, parent_tube):
    """Clip initial aneurysm surface from the original vascular model.
    
    Compute distance between the aneurysm 
    envelope and parent vasculature tube 
    function from the original vascular 
    surface model. Clip the surface at the 
    zero value of the difference between 
    these two fields.
    
    Arguments:
    surface_model --  the original vascular surface
    aneuysm_envelope -- the aneurysm surface computed from its Voronoi
    parent_tube -- tube surface of the parent vessel
    """

    # Array names
    clipAneurysmArray    = 'ClipInitialAneurysmArray'
    tubeToModelArray     = 'ParentTubeModelDistanceArray'
    envelopeToModelArray = 'AneurysmEnvelopeModelDistanceArray'

    # Computes distance between original surface model 
    # and the aneurysm envelope, and from the parent 
    # tube surface
    aneurysmEnvelopeDistance = _compute_surfaces_distance(surface_model, 
                                                          aneurysm_envelope, 
                                                          array_name=envelopeToModelArray)
    
    modelSurfaceWithDistance = _compute_surfaces_distance(aneurysmEnvelopeDistance, 
                                                          parent_tube, 
                                                          array_name=tubeToModelArray)


    # Compute difference between the arrays
    clippingArray = vmtkscripts.vmtkSurfaceArrayOperation()
    clippingArray.Surface   = modelSurfaceWithDistance
    clippingArray.Operation = 'subtract'
    clippingArray.InputArrayName  = envelopeToModelArray
    clippingArray.Input2ArrayName = tubeToModelArray
    clippingArray.ResultArrayName = clipAneurysmArray
    clippingArray.Execute()

    # Clip the model surface at the zero level of the difference array
    clipAneurysm = vmtkscripts.vmtkSurfaceClipper()
    clipAneurysm.Surface       = clippingArray.Surface
    clipAneurysm.ClipArrayName = clippingArray.ResultArrayName
    clipAneurysm.ClipValue     = intZero
    clipAneurysm.Interactive   = False
    clipAneurysm.InsideOut     = False
    clipAneurysm.Execute()

    aneurysm = _connected_region(clipAneurysm.Surface,'largest')
    
    # Remove fields
    aneurysm.GetPointData().RemoveArray(clipAneurysmArray)
    aneurysm.GetPointData().RemoveArray(tubeToModelArray)
    aneurysm.GetPointData().RemoveArray(envelopeToModelArray)
    
    aneurysm = tools.cleaner(aneurysm)
    
    if _inspect:
        tools.viewSurface(aneurysm)
    
    if _write:
        tools.writeSurface(aneurysm, _write_dir+'initial_aneurysm_surface.vtp')
    
    return aneurysm


def _clip_tube(parent_tube, parent_centerline, clipping_points):
    """Clip the tube surface portion between clipping points."""
    
    # Resample centerline first
    # (avoid flipped normals to clip plane) November, 14, 2019
    clResampling = vmtkscripts.vmtkCenterlineResampling()
    clResampling.Centerlines = parent_centerline
    clResampling.Length = intOne/intTen
    clResampling.Execute()
    
    parent_centerline = clResampling.Centerlines
    
    # Search clipping points in the new centerlines
    clipCenters = list()
    clipNormals = list()
    
    clipPointsArray = _vtk_vertices_to_numpy(clipping_points)
    
    for point in clipPointsArray:

        # Get id of point with min distance
        idMinDist = _id_min_dist_to_point(point, parent_centerline)

        # Get plane centers
        center = np.array(parent_centerline.GetPoint(idMinDist))
        clipCenters.append(center)

        # Build normals to each clipping point by getting next abscissa coordinates
        nextCenter = np.array(parent_centerline.GetPoint(idMinDist + 1))
        normal     = nextCenter - center

        clipNormals.append(normal)

        
    clipCenters = np.array(clipCenters)
    clipNormals = np.array(clipNormals)

    # Clip the parent artery first
    center = clipCenters[intZero]
    normal = clipNormals[intZero]

    # Get barycenter of clipping points
    clipPointsPoints     = clipping_points.GetPoints()    
    clipPointsBarycenter = _dimensions*[intZero]

    vtkvmtk.vtkvmtkBoundaryReferenceSystems.ComputeBoundaryBarycenter(
        clipPointsPoints,clipPointsBarycenter
    )    
    
    clippedTube = _connected_region(_clip_with_plane(parent_tube,center,normal),
                                    'closest',
                                    clipPointsBarycenter)
        
        
    # Update clip points
    clipCenters = np.delete(clipCenters, intZero, axis=0)
    clipNormals = np.delete(clipNormals, intZero, axis=0)

    # Clip remaining daughters
    for center, normal in zip(clipCenters,clipNormals):
        
        clipDaughter = _clip_with_plane(clippedTube,
                                        center,
                                        normal,
                                        inside_out=True)

        clippedTube  = _connected_region(clipDaughter,
                                         'closest',
                                         clipPointsBarycenter)

    if _inspect:
        tools.viewSurface(clippedTube)
        
    if _write:
        tools.writeSurface(clippedTube,_write_dir+'clipped_tube.vtp')
    
    return clippedTube


def _contour_perimeter(contour):
    """Compute the perimeter of a contour defined in 3D space."""

    nContourVertices = contour.GetNumberOfPoints()

    # Compute neck perimeter
    perimeter = intZero
    previous  = contour.GetPoint(intZero)

    for index in range(nContourVertices):
        if index > intZero:
            previous = contour.GetPoint(index - intOne)

        vertex = contour.GetPoint(index)

        # Compute distance between two consecutive points
        distanceSquared = vtk.vtkMath.Distance2BetweenPoints(previous, vertex)
        increment = np.sqrt(distanceSquared)    

        perimeter += increment

    return perimeter

def _contour_plane_area(contour):
    """Compute plane surface area enclosed by a 3D contour path."""
    # Fill contour
    fillContour = vtk.vtkContourTriangulator()
    fillContour.SetInputData(contour)
    fillContour.Update()

    # Convert vtkUnstructuredData to vtkPolyData
    meshToSurfaceFilter = vtk.vtkGeometryFilter()
    meshToSurfaceFilter.SetInputData(fillContour.GetOutput())
    meshToSurfaceFilter.Update()

    computeArea = vtk.vtkMassProperties()
    computeArea.SetInputData(meshToSurfaceFilter.GetOutput())
    computeArea.Update()

    return computeArea.GetSurfaceArea()


def _contour_hydraulic_diameter(contour):
    """
        Computes the hydraulic diameter of a cross 
        section, provided its poly line contour.
    """

    contourPerimeter = _contour_perimeter(contour)
    contourSurfArea  = _contour_plane_area(contour)

    # Return hydraulic diameter of neck
    return intFour * contourSurfArea/contourPerimeter



def _sac_centerline(aneurysm_sac,distance_array):
    """Compute aneurysm sac centerline.
    
    Compute spline that travels alongs the 
    aneurysm sac from the intersection with the pa-
    rent vessel tube. Its points are defined by the
    geometric place of the barycenters of iso-
    contours of a distance_array defined on the 
    aneurysm surface.

    The function returns the spline vertices in a
    Numpy nd-array.
    """
    
    # Get wrapper object of vtk numpy interface
    surfaceWrapper = dsa.WrapDataObject(aneurysm_sac)
    distanceArray  = np.array(surfaceWrapper.PointData.GetArray(distance_array))
    
    minTubeDist = float(distanceArray.min())
    maxTubeDist = float(distanceArray.max())

    # Build spline along with to perform the neck search
    
    nPoints       = intOne*intHundred   # number of points to perform search
    barycenters   = list()              # barycenter's list    
    
    aneurysm_sac.GetPointData().SetActiveScalars(distance_array)

    # Get barycenters of iso-contours
    for isovalue in np.linspace(minTubeDist, maxTubeDist, nPoints):

        # Get isocontour polyline
        isoContour = vtk.vtkContourFilter()
        isoContour.SetInputData(aneurysm_sac)
        isoContour.ComputeScalarsOff()
        isoContour.ComputeNormalsOff()
        isoContour.SetValue(intZero,isovalue)
        isoContour.Update()

        # Get largest connected contour
        contour = _connected_region(isoContour.GetOutput(),'largest')
        
        try:
            contourPoints  = contour.GetPoints()
            nContourPoints = contour.GetNumberOfPoints()

            if _contour_is_closed(contour) and nContourPoints != intZero:
                barycenter = _dimensions*[intZero]
                vtkvmtk.vtkvmtkBoundaryReferenceSystems.ComputeBoundaryBarycenter(
                    contourPoints,
                    barycenter
                )

                barycenters.append(barycenter)

        except:
            continue

    barycenters = np.array(barycenters)

    # Compute list with distance coordinate along spline
    distance = intZero                      
    previous = barycenters[intZero]
    distanceCoord = list()                    # path distance coordinate
    
    for index, center in enumerate(barycenters):
        
        if index > intZero:
            previous = barycenters[index - intOne]

        # Interpoint distance
        increment  = center - previous
        distance  += np.linalg.norm(increment,intTwo)  
        distanceCoord.append(distance)

    distanceCoord = np.array(distanceCoord)

    # Find spline of barycenters and get derivative == normals
    limitFraction = intSeven/intTen
    
    tck, u = interpolate.splprep(barycenters.T, u=distanceCoord)
    
    # Max and min t of spline
    minSplineDomain = min(u)-intFive/intTen
    maxSplineDomain = limitFraction*max(u)
    
    domain  = np.linspace(minSplineDomain, maxSplineDomain, intTwo*nPoints)
    
    deriv0  = interpolate.splev(domain, tck, der=0)
    deriv1  = interpolate.splev(domain, tck, der=1)

    points   = np.array(deriv0).T            # spline points
    tangents = np.array(deriv1).T            # spline tangents
    
    if _write:
        # Write spline to vtp file
        data = vtk.vtkPoints()
        for point in points:
            data.InsertNextPoint(point)

        spline = vtk.vtkPolyData()
        spline.SetPoints(data)

        pointDataArray = vtk.vtkFloatArray()
        pointDataArray.SetNumberOfComponents(3)
        pointDataArray.SetName('Normals')
        for pointData in tangents:         
            pointDataArray.InsertNextTuple(pointData)

        spline.GetPointData().SetActiveVectors('Normals')
        spline.GetPointData().SetVectors(pointDataArray)

        tools.writeSurface(spline,_write_dir+'sac_spline.vtp')
    
    return points,tangents


def _local_minimum(array):
    """Find local minimum closest to beginning of array.
    
    Given an array of real numbers, return the id of the 
    smallest value closest to the beginning of the array.
    """
    minimum = np.r_[True, array[1:] < array[:-1]] & \
              np.r_[array[:-1] < array[1:], True]

    # Local minima index    
    return int(np.where(minimum == True)[intZero][intZero])
    

def _search_neck_plane(anerysm_sac,centers,normals,min_variable='area'):
    """Search neck plane of aneurysm by minimizing a contour variable.
    
    This function effectively searches for the aneurysm neck
    plane: it interactively cuts the aneurysm surface with
    planes defined by the vertices and normals to a spline
    travelling through the aneurysm sac. 

    The cut plane is further precessed by a tilt and azimuth
    angle and the minimum search between them, as originally
    proposed by Piccinelli et al. (2009).

    It returns the local minimum solution: the neck plane as 
    a vtkPlane object.
    """
    # Rotation angles 
    tiltIncr = intOne
    azimIncr = intTen
    tiltMax  = 32
    azimMax  = 360
    
    tilts = np.arange(intZero, tiltMax, tiltIncr) * degToRad
    azims = np.arange(intZero, azimMax, azimIncr) * degToRad

    # Minimum area seacrh
    sectionInfo  = list()
    previousVariable = HUGE

    # Iterate over barycenters and clip surface with closed surface
    for index, (center, normal) in enumerate(zip(centers,normals)): 

        # Store previous area to compare and find local minimum
        if index > intZero:
            previousVariable = minVariable

        # Iterate over rotated planes
        # identifies the minimum area
        minArea  = HUGE
        minPlane = None
        
        minHDiameter = HUGE
        minPerimeter = HUGE
        minVariable  = HUGE
        
        for tilt in tilts:
            for azim in azims:

                # Set plane with barycenters and 
                # normals as tangent to spline
                plane = vtk.vtkPlane()
                plane.SetOrigin(tuple(center)) 

                # Rotate normal
                matrix  = _rotate3d_matrix(tilt, azim)
                rNormal = np.dot(matrix, normal)

                # Set rotate plane normal
                plane.SetNormal(tuple(rNormal))

                # Cut initial aneurysm surface with create plane
                cutWithPlane = vtk.vtkCutter()
                cutWithPlane.SetInputData(anerysm_sac)
                cutWithPlane.SetCutFunction(plane)
                cutWithPlane.Update()
                
                contour = _connected_region(cutWithPlane.GetOutput(),'largest')
                
                try:
                    contourPoints  = contour.GetPoints()
                    nContourPoints = contour.GetNumberOfPoints()

                    if _contour_is_closed(contour) and nContourPoints != intZero:
                        
                        # Update minmum area
                        if   min_variable == 'area': 
                            variable = _contour_plane_area(contour)
                            
                        elif min_variable == 'perimeter': 
                            variable = _contour_perimeter(contour)
                            
                        elif min_variable == 'hyd_diameter': 
                            variable = intFour*area/perimeter
                            
                        else: print('Minimizing variable not recognized!' \
                                    'Choose area, perimeter or hyd_diameter.')
                            
                        if variable < minVariable: 
                            minVariable = variable
                            minPlane    = plane
                except:
                    continue


        # Write to array min area and its surface plane  
        if minVariable != HUGE:
            if minVariable > previousVariable:
                break

            sectionInfo.append([minVariable, minPlane])
            
    sectionInfo = np.array(sectionInfo)

    # Get local minimum area
    areas     = sectionInfo[:,intZero]
    
    minimumId = _local_minimum(areas)

    return sectionInfo[minimumId,intOne]


def generateCenterline(surface):
    """Compute centerline and resampling it."""
    centerlines = vmtkscripts.vmtkCenterlines()
    centerlines.Surface = surface
    centerlines.Execute()

    # Resampling
    centerlineResampling = vmtkscripts.vmtkCenterlineResampling()
    centerlineResampling.Centerlines = centerlines.Centerlines
    centerlineResampling.Length      = intOne/intTen
    centerlineResampling.Execute()
    
    if _inspect:
        tools.viewSurface(centerlineResampling.Centerlines)
    
    return centerlineResampling.Centerlines


def aneurysmNeckPlane(surface_model,
                      parent_centerlines,
                      clipping_points,
                      min_variable='perimeter'):
    """Extracts the aneurysm neck plane.
    
    Procedure based on Piccinelli's pipeline, which is based on 
    the surface model with the aneurysm and its parent vasculature 
    reconstruction. The single difference is the variable which the
    algorithm minimizes to search for the neck plane: the default 
    is the neck perimeter, whereas in the default procedure is the
    neck section area; this can be controlled by the optional
    argument 'min_variable'.

    It returns the clipped aneurysm surface from the original
    vasculature.
    
    Arguments
    ---------
    surface_model : the original vasculature surface with the 
                    aneurysm
    parent_centerlines : the centerlines of the reconstructed 
                         parent vasculature
    clipping_points : points where the vasculature will be 
                      clipped.
                      
    Optional args
    min_variable : the varible by which the neck will be searched
                   (default 'perimeter'; options 'perimeter' 'area')
    """ 
    # Variables
    tubeToAneurysmDistance = 'ClippedTubeToAneurysmDistanceArray'
    
    # Initiate neck plane detection pipeline
    VoronoiDiagram   = _compute_Voronoi(surface_model)

    parentTube       = _tube_surface(parent_centerlines)

    clippedTube      = _clip_tube(parentTube,
                                  parent_centerlines,
                                  clipping_points)

    aneurysmVoronoi  = _clip_aneurysm_Voronoi(VoronoiDiagram, parentTube)

    aneurysmEnvelope = _Voronoi_envelope(aneurysmVoronoi)

    initialAneurysm  = _clip_initial_aneurysm(surface_model,
                                              aneurysmEnvelope,
                                              parentTube)
    
    initialAneurysmSurface = _compute_surfaces_distance(initialAneurysm, 
                                                        clippedTube, 
                                                        array_name=tubeToAneurysmDistance, 
                                                        signed_array=False)
    
    barycenters,normals = _sac_centerline(initialAneurysmSurface,
                                          tubeToAneurysmDistance)

    # Search for neck plane
    neckPlane  = _search_neck_plane(initialAneurysmSurface,
                                    barycenters,
                                    normals,
                                    min_variable=min_variable)
    
    neckCenter = neckPlane.GetOrigin()
    neckNormal = neckPlane.GetNormal()

    # Remove distance array
    initialAneurysmSurface.GetPointData().RemoveArray(tubeToAneurysmDistance) 
    
    # Clip final aneurysm surface: when clipped, two surfaces the aneurysm (desired) and
    # the rest (not desired) which is closest to the clipped tube
    surf1 = _clip_with_plane(initialAneurysmSurface,neckCenter,neckNormal)
    surf2 = _clip_with_plane(initialAneurysmSurface,neckCenter,neckNormal,inside_out=True)

    # Check which output is farthest from clipped tube 
    tubePoints  = _vtk_vertices_to_numpy(clippedTube)
    surf1Points = _vtk_vertices_to_numpy(surf1)
    surf2Points = _vtk_vertices_to_numpy(surf2)

    tubeCentroid  = tubePoints.mean(axis=0)
    surf1Centroid = surf1Points.mean(axis=0)
    surf2Centroid = surf2Points.mean(axis=0)

    surf1Distance = vtk.vtkMath.Distance2BetweenPoints(tubeCentroid,surf1Centroid)
    surf2Distance = vtk.vtkMath.Distance2BetweenPoints(tubeCentroid,surf2Centroid)

    if surf1Distance > surf2Distance:
        aneurysmSurface = surf1
    else:
        aneurysmSurface = surf2 
    
    return aneurysmSurface
