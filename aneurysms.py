import sys
import vtk
import numpy as np

from vmtk import vmtkscripts
from vmtk import vtkvmtk
from scipy import interpolate
from scipy.spatial import ConvexHull

from vtk.numpy_interface import dataset_adapter as dsa

import vmtkfunctions as vf

# Flags
_write = False
_debug = False
_inspect = False

# Constants
_intZero  = 0
_intOne   = 1
_intTwo   = 2
_intThree = 3
_intFour  = 4
_intFive  = 5

_intTen = 10
_intHundred = 100

INCR = 0.01 
HUGE = 1e30

_dimensions = _intThree
_xComp = _intZero
_yComp = _intOne
_zComp = _intThree

degToRad = np.pi/180.0


def _id_min_dist_to_point(point, array):
    """ 
        Get the id of a point in an
        array which is the closest from a point
        given by the user that does not necessarely
        belongs to the array.
    """
    
    # Computes distance vector array between point and array
    arrDistToPoint = array - point
    arrDistance    = np.linalg.norm(arrDistToPoint, axis=1)

    minIds = np.where(arrDistance == min(arrDistance))
    
    return minIds[_intZero][_intZero]      # to get first of tuple and the first of list in case of equispaced points


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
    return np.array([[             np.cos(azim),             -np.sin(azim),      _intZero],
                     [np.sin(azim)*np.cos(tilt), np.cos(azim)*np.cos(tilt), -np.sin(tilt)],
                     [np.sin(azim)*np.sin(tilt), np.cos(azim)*np.sin(tilt),  np.cos(tilt)]])



def _read_surface(file_name):
    """Read surface file to VTK object.
               
    Arguments: 
    file_name -- complete path with file name
    """
    reader = vmtkscripts.vmtkSurfaceReader()
    reader.InputFileName = file_name
    reader.Execute()
    
    return reader.Surface


# Viewing surface
def _view_surface(surface):
    """View surface vtkPolyData objects.
    
    Arguments:
    surface -- the surface to be displayed.
    """
    viewer = vmtkscripts.vmtkSurfaceViewer()
    viewer.Surface = surface
    viewer.Execute()

    
# Writing a surface
def _write_surface(surface, file_name, mode='binary'):
    """Write surface vtkPolyData. 
        
    Arguments:
    surface -- surface vtkPolyData object
    file_name -- output file name with full path
    
    Optional arguments:
    mode -- mode to write file (ASCII or binary, default binary)
    """
    writer = vmtkscripts.vmtkSurfaceWriter()
    writer.Surface = surface
    writer.Mode    = mode
    writer.OutputFileName = file_name
    writer.Execute()


def _smooth_surface(surface):
    """Smooth surface based on Taubin's algorithm."""
    smoother = vmtkscripts.vmtkSurfaceSmoothing()
    smoother.Surface = surface
    smoother.Method  = 'taubin'
    smoother.NumberOfIterations = _intThree*_intTen
    smoother.PassBand = _intOne/_intTen
    smoother.Execute()
    
    return smoother.Surface


def _connected_region(regions,method,closest_point=None):
    """Extract the largest or closest to point patch of a disconnected domain."""
    triangulator = vmtkscripts.vmtkSurfaceTriangle()
    triangulator.Surface = regions
    triangulator.Execute()
    
    connectivity = vmtkscripts.vmtkSurfaceConnectivity()
    connectivity.Surface     = triangulator.Surface
    connectivity.Method      = method
    connectivity.CleanOutput = True
    
    if method == 'closest':
        connectivity.ClosestPoint = closest_point
    
    connectivity.Execute()
    
    return connectivity.Surface


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
    
    Arguents:
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
        _view_surface(tube)
    
    if smooth:
        return _smooth_surface(tube)
    else:
        return tube


def _compute_surfaces_distance(isurface, rsurface, array_name='DistanceArray', signed_array=True):
    """
    Internal function to compute distance between a reference
    surface, rsurface, and an input surface, isurface, with 
    the resulting array written in the isurface. 
    """
    
    surfaceDistance = vmtkscripts.vmtkSurfaceDistance()
    surfaceDistance.Surface = isurface
    surfaceDistance.ReferenceSurface = rsurface
    
    # Check if signed array or not
    if signed_array:
        surfaceDistance.SignedDistanceArrayName = array_name
    else:
        surfaceDistance.DistanceArrayName = array_name
         
    surfaceDistance.Execute()    
    
    return surfaceDistance.Surface


def _clip_aneurysm_Voronoi(VoronoiSurface, tubeSurface):
    """
    Procedure to clip the voronoi diagram only of 
    the aneurysm portion of the original surface.
    """

    # Compute distance between complete Voronoi and the parent vessel tube surface 
    DistanceArrayName = 'DistanceToTubeArray'
    VoronoiDistance   = _compute_surfaces_distance(VoronoiSurface, tubeSurface, array_name=DistanceArrayName)

    # Clip the original voronoi diagram at the zero distance (intersection)
    VoronoiClipper = vmtkscripts.vmtkSurfaceClipper()
    VoronoiClipper.Surface = VoronoiDistance
    VoronoiClipper.Interactive   = False
    VoronoiClipper.ClipArrayName = DistanceArrayName
    VoronoiClipper.ClipValue     = _intZero
    VoronoiClipper.InsideOut     = True
    VoronoiClipper.Execute()

    aneurysmVoronoi = _connected_region(VoronoiClipper.Surface,'largest')
    
    if _inspect:
        _view_surface(aneurysmVoronoi)
        
    return aneurysmVoronoi


def _Voronoi_envelope(Voronoi):
    """Compute the envelope surface of a Voronoi diagram."""
    # List to collect objects 
    objects = []
    
    radiusArray   = 'MaximumInscribedSphereRadius'
    boxDimensions = _dimensions*[64]
    
    # Building the envelope image function
    envelopeFunction = vmtkscripts.vmtkPolyBallModeller()
    envelopeFunction.Surface = Voronoi
    envelopeFunction.RadiusArrayName  = radiusArray
    envelopeFunction.SampleDimensions = boxDimensions
    envelopeFunction.Execute()
    objects.append(envelopeFunction)
    
    # Get level zero surface
    envelopeSurface = vmtkscripts.vmtkMarchingCubes()
    envelopeSurface.Image = envelopeFunction.Image
    envelopeSurface.Execute()
    objects.append(envelopeSurface)

    envelope = _connected_region(envelopeSurface.Surface,'largest')
    
    if _debug:
        for obj in objects:
            obj.PrintInputMembers()
            obj.PrintOutputMembers()
            
    if _inspect:
        _view_surface(envelope)
        
    return envelope


def _clip_initial_aneurysm(surface_model, aneurysm_envelope, parent_tube):
    """Clip initial aneurysm surface from the original vascular model.
    
    Compute distance between the aneurysm envelope and parent vasculature 
    tube function from the original vascular surface model. Clip the surface
    at the zero value of the difference between these two fields.
    
    Arguments:
    surface_model --  the original vascular surface
    aneuysm_envelope -- the aneurysm surface computed from its Voronoi
    parent_tube -- tube surface of the parent vessel
    """

    # Array names
    clipAneurysmArray    = 'ClipInitialAneurysmArray'
    tubeToModelArray     = 'ParentTubeModelDistanceArray'
    envelopeToModelArray = 'AneurysmEnvelopeModelDistanceArray'

    # Computes distance between original surface model and the aneurysm envelope, 
    # and from the parent tube surface
    aneurysmEnvelopeDistance = _compute_surfaces_distance(surface_model, aneurysm_envelope, array_name=envelopeToModelArray) 
    modelSurfaceWithDistance = _compute_surfaces_distance(aneurysmEnvelopeDistance, parent_tube, array_name=tubeToModelArray)


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
    clipAneurysm.Surface = clippingArray.Surface
    clipAneurysm.ClipArrayName = clippingArray.ResultArrayName
    clipAneurysm.ClipValue   = _intZero
    clipAneurysm.Interactive = False
    clipAneurysm.InsideOut   = False
    clipAneurysm.Execute()

    aneurysm = _connected_region(clipAneurysm.Surface,'largest')
    
    # Remove fields
    aneurysm.GetPointData().RemoveArray(clipAneurysmArray)
    aneurysm.GetPointData().RemoveArray(tubeToModelArray)
    aneurysm.GetPointData().RemoveArray(envelopeToModelArray)
    
    return aneurysm

def _clip_tube(parent_tube, parent_centerline, clipping_points):
    """Clip the tube surface portion between clipping points."""
    # Get parent centerline points 
    vertices = _vtk_vertices_to_numpy(parent_centerline)

    # Search clipping points in the new centerlines
    clipCenters = []
    clipNormals = []
    
    clipPointsArray = _vtk_vertices_to_numpy(clipping_points)
    
    for point in clipPointsArray:

        # Get id of point with min distance
        idMinDist = int(_id_min_dist_to_point(point, vertices))

        # Get plane centers
        center = vertices[idMinDist]
        clipCenters.append(center)

        # Build normals to each clipping point by getting next abscissa coordinates
        nextCenter = vertices[idMinDist + 1]
        normal     = nextCenter - center

        clipNormals.append(normal)

        
    clipCenters = np.array(clipCenters)
    clipNormals = np.array(clipNormals)

    # Clip the parent artery first
    center = clipCenters[_intZero]
    normal = clipNormals[_intZero]

    # Get barycenter of clipping points
    clipPointsPoints     = clipping_points.GetPoints()    
    clipPointsBarycenter = _dimensions * [_intZero]
    vtkvmtk.vtkvmtkBoundaryReferenceSystems.ComputeBoundaryBarycenter(clipPointsPoints,clipPointsBarycenter)    
    
    
    clippedTube = _connected_region(_clip_with_plane(parent_tube,center,normal),
                                    'closest',
                                    clipPointsBarycenter)
    
    if _inspect:
        _view_surface(clippedTube)
        
        
    # Update clip points
    clipCenters = np.delete(clipCenters, _intZero, axis=0)
    clipNormals = np.delete(clipNormals, _intZero, axis=0)

    # Clip remaining daughters
    for center, normal in zip(clipCenters,clipNormals):
        
        clipDaughter = _clip_with_plane(clippedTube,center,normal,inside_out=True)       
        clippedTube  = _connected_region(clipDaughter,'closest',clipPointsBarycenter)

        
    if _inspect:
        _view_surface(clippedTube)
    
    return clippedTube


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
    
    minTubeDist = distanceArray.min()
    maxTubeDist = distanceArray.max()
    
    # Build spline along with to perform the neck search
    
    nPoints       = _intTwo * _intHundred   # number of points to perform search: 200
    barycenters   = []                      # barycenter's list    
    
    aneurysm_sac.GetPointData().SetActiveScalars(distance_array)

    # Get barycenters of iso-contours
    limitFraction = _intFive/_intTen
    
    for isovalue in np.linspace(minTubeDist, limitFraction*maxTubeDist, nPoints):

        # Get isocontour polyline
        isoContour = vtk.vtkContourFilter()
        isoContour.SetInputData(aneurysm_sac)
        isoContour.ComputeScalarsOff()
        isoContour.ComputeNormalsOff()
        isoContour.SetValue(_intZero,isovalue)
        isoContour.Update()

        # Get largest connected contour
        contour = _connected_region(isoContour.GetOutput(),'largest')

        try:
            contourPoints  = contour.GetPoints()
            nContourPoints = contour.GetNumberOfPoints()

            if _contour_is_closed(contour) and nContourPoints != _intZero:
                barycenter = _dimensions*[_intZero]
                vtkvmtk.vtkvmtkBoundaryReferenceSystems.ComputeBoundaryBarycenter(contourPoints,barycenter)
                barycenters.append(barycenter)

        except:
            continue

    return np.array(barycenters)


def _search_neck_plane(anerysm_sac,centers,normals):
    """Search neck plane of aneurysm
    
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
    tiltIncr = _intOne
    azimIncr = _intTen
    tiltMax  = 32
    azimMax  = 360
    
    tilts = np.arange(_intZero, tiltMax, tiltIncr) * degToRad
    azims = np.arange(_intZero, azimMax, azimIncr) * degToRad

    # Minimum area seacrh
    sectionInfo  = []
    previousArea = HUGE

    # Iterate over barycenters and clip surface with closed surface
    for index, (center, normal) in enumerate(zip(centers,normals)): 

        # Store previous area to compare and find local minimum
        if index > _intZero:
            previousArea = minArea

        # Iterate over rotated planes
        # identifies the minimum area
        minArea  = HUGE
        minPlane = None

        for tilt in tilts:
            for azim in azims:

                # Set plane with barycenters and normals as tangent to spline
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

                    if _contour_is_closed(contour) and nContourPoints != _intZero:
                        fillContour = vtk.vtkContourTriangulator()
                        fillContour.SetInputData(contour)
                        fillContour.Update()

                        computeArea = vtk.vtkMassProperties()
                        computeArea.SetInputData(fillContour.GetOutput())
                        computeArea.Update()

                        # Get area
                        area = computeArea.GetSurfaceArea()

                        # Update minmum area 
                        if area < minArea: 
                            minArea  = area
                            minPlane = plane
                
                except:
                    continue


        # Write to array min area and its surface plane  
        if minArea != HUGE:
            if minArea > previousArea:
                break

            sectionInfo.append([minArea, minPlane])

            
    sectionInfo = np.array(sectionInfo)

    # Get local minimum area
    areas  = sectionInfo[:,_intZero]    
    minimumArea = np.r_[True, areas[1:] < areas[:-1]] & \
                  np.r_[areas[:-1] < areas[1:], True]

    # Local minima index    
    minimumId = int(np.where(minimumArea == True)[_intZero][_intZero])

    return sectionInfo[minimumId,_intOne]


def generateCenterline(surface):
    """Compute centerline and resampling it."""
    centerlines = vmtkscripts.vmtkCenterlines()
    centerlines.Surface = surface
    centerlines.Execute()

    # Resampling
    centerlineResampling = vmtkscripts.vmtkCenterlineResampling()
    centerlineResampling.Centerlines = centerlines.Centerlines
    centerlineResampling.Length      = _intOne/_intTen
    centerlineResampling.Execute()
    
    if _inspect:
        _view_surface(centerlineResampling.Centerlines)
    
    return centerlineResampling.Centerlines


def aneurysmNeckPlane(surface_model,parent_centerlines,clipping_points):
    """Extracts the aneurysm neck plane.
    
    Procedure based on Piccinelli's pipeline, which is based on 
    the surface model with the aneurysm and its parent vasculature 
    reconstruction.

    It returns the clipped aneurysm surface from the original
    vasculature, and not the plane surface itself.
    """ 
    
    
    VoronoiDiagram   = _compute_Voronoi(surface_model)

    parentTube       = _tube_surface(parent_centerlines)

    clippedTube      = _clip_tube(parentTube,parent_centerlines,clipping_points)

    aneurysmVoronoi  = _clip_aneurysm_Voronoi(VoronoiDiagram,parentTube)

    aneurysmEnvelope = _Voronoi_envelope(aneurysmVoronoi)

    initialAneurysm  = _clip_initial_aneurysm(surface_model,aneurysmEnvelope,parentTube)


    # Computes distance between initial aneurysm surface and the clipped tube surface
    tubeToAneurysmDistance = 'ClippedTubeToAneurysmDistanceArray'
    
    initialAneurysmSurface = _compute_surfaces_distance(initialAneurysm, 
                                                        clippedTube, 
                                                        array_name=tubeToAneurysmDistance, 
                                                        signed_array=False)

    # Inspect if initial aneurysm surface is correct
    if _inspect:
        _view_surface(clippedTube)
        _view_surface(initialAneurysmSurface)

        
    barycenters = _sac_centerline(initialAneurysmSurface,tubeToAneurysmDistance)

    # Compute list with distance coordinate along spline
    distance = _intZero                      
    previous = barycenters[_intZero]
    distanceCoord = []                      # path distanec coordinate
    
    for index, center in enumerate(barycenters):
        
        if index > _intZero:
            previous = barycenters[index - _intOne]
        
        
        # Interpoint distance
        increment  = center - previous
        distance  += np.linalg.norm(increment,_intTwo)  
        distanceCoord.append(distance)

    distanceCoord = np.array(distanceCoord)

    # Find spline of barycenters and get derivative == normals
    tck, u  = interpolate.splprep(barycenters.T, u=distanceCoord)
    deriv   = interpolate.splev(u, tck, der=1)
    normals = np.array(deriv).T

    # searches for neck plane
    neckPlane  = _search_neck_plane(initialAneurysmSurface,barycenters,normals) 
    neckCenter = neckPlane.GetOrigin()
    neckNormal = neckPlane.GetNormal()

    # Remove distance array
    initialAneurysmSurface.GetPointData().RemoveArray(tubeToAneurysmDistance) 

    return _clip_with_plane(initialAneurysmSurface,neckCenter,neckNormal)


class Aneurysm:  
    """Class representing saccular intracranial aneurysms.
        
    Made internal use of VTK and VMTK's classes for 
    vtkPolyData manipulations. Its main input is the
    aneurysm sac as a vtkPolyData object.
        
    The surface must be open for correct computation of
    its surface area.
    """
    
    # Constructor
    def __init__(self, surface, typeOfAneurysm, status, label):
        self.aneurysmType   = typeOfAneurysm
        self.aneurysmLabel  = label
        self.aneurysmStatus = status
        
        # Triangulate vtkPolyData surface
        triangulate = vtk.vtkTriangleFilter()
        triangulate.SetInputData(surface)
        triangulate.Update()
        
        self.aneurysmSurface = triangulate.GetOutput()
        
        # Cap aneurysm surface
        # (needed for correct calculation of aneurysm volume)
        cappedSurface = self._cap_aneurysm()
        
        # Compute attributes (3D size indices)
        # Surface area is computed with the open surface
        self.surfaceArea = self._surface_area(self.aneurysmSurface)
        self.volume      = self._surface_volume(cappedSurface)
        
        # Compute neck surface area
        self.neckPlaneArea = self._surface_area(self._neck_surface())
        
        # Computing hull properties
        self.hullSurface = self._aneurysm_convex_hull()

     
    def _cap_aneurysm(self):
        """ 
            Returns aneurysm surface capped with a plane
            of triangles. Uses VMTK's script 'vmtksurfacecapper'. 
        """
        
        capper = vmtkscripts.vmtkSurfaceCapper()
        capper.Surface = self.aneurysmSurface
        capper.Interactive = False
        capper.Execute()
        
        return capper.Surface
    

    def _surface_area(self, surf):
        """ 
            Computes the surface area of an input surface. 
        """

        surface_area = vtk.vtkMassProperties()
        surface_area.SetInputData(surf)
        surface_area.Update()
        
        return surface_area.GetSurfaceArea()

    
    def _surface_volume(self, surf):
        """ Computes the volume of an assumed orientable 
            surface. Works internally with VTK, so it 
            assumes that the surface is closed. 
        """

        volume = vtk.vtkMassProperties()
        volume.SetInputData(surf)
        volume.Update()
        
        return volume.GetVolume()

    
    def _make_vtk_id_list(self,it):
        vil = vtk.vtkIdList()
        for i in it:
            vil.InsertNextId(int(i))
        return vil    
    
    def _aneurysm_convex_hull(self):
        """
            This function computes the convex hull set of an
            aneurysm surface provided as a polyData set of VTK.
            It uses internally the scipy.spatial package.
        """

        # Convert surface points to numpy array
        nPoints = self.aneurysmSurface.GetNumberOfPoints()
        vertices  = []
        
        for index in range(nPoints):
            vertex = self.aneurysmSurface.GetPoint(index)
            vertices.append(list(vertex))

        vertices = np.array(vertices)

        # Compute convex hull of points
        aneurysmHull = ConvexHull(vertices)

        # Get hull properties
        self.hullVolume = aneurysmHull.volume
        
        # Need to subtract neck area to 
        # compute correct hull surface area
        self.hullArea   = aneurysmHull.area - self.neckPlaneArea

        # Intantiate poly data
        polyData = vtk.vtkPolyData()

        # Get points
        points = vtk.vtkPoints()

        for xyzPoint in aneurysmHull.points:
            points.InsertNextPoint(xyzPoint)

        polyData.SetPoints(points)

        # Get connectivity matrix
        cellDataArray = vtk.vtkCellArray()

        for cellId in aneurysmHull.simplices:
            if type(cellId) is np.ndarray:
                cellDataArray.InsertNextCell(self._make_vtk_id_list(cellId))
            else:
                for cell in cellId:
                    cellDataArray.InsertNextCell(self._make_vtk_id_list(cell)) 

        polyData.SetPolys(cellDataArray)         

        return polyData
    
    
    
    def _neck_contour(self):
        """
            Get boundary of aneurysm surface (== neck contour)
        """
        boundaryExtractor = vtkvmtk.vtkvmtkPolyDataBoundaryExtractor()
        boundaryExtractor.SetInputData(self.aneurysmSurface)
        boundaryExtractor.Update()

        return boundaryExtractor.GetOutput()


    def _neck_barycenter(self):
        """
            Computes and return the neck line barycenter
            as a Numpy array.
        """
        # Get neck contour
        neckContour = self._neck_contour()
        neckPoints  = neckContour.GetPoints()

        barycenter  = np.zeros(_intThree)
        vtkvmtk.vtkvmtkBoundaryReferenceSystems.ComputeBoundaryBarycenter(neckPoints,barycenter)

        return barycenter


    def _neck_surface(self):
        """
            Compputes aneurysm neck plane polyData
        """
        
        neckIndex = _intTwo
        
        # Use thrshold filter to get neck plane
        getNeckPlane = vmtkscripts.vmtkThreshold()
        getNeckPlane.Surface = self._cap_aneurysm()
        getNeckPlane.LowThreshold  = neckIndex
        getNeckPlane.HighThreshold = neckIndex
        getNeckPlane.Execute()
        
        return getNeckPlane.Surface
        

    def _max_height_vector(self):
        """ 
            Function to compute the vector from the neck 
            contour barycenter and the fartest point
            on the aneurysm surface
        """

        neckContour = self._neck_contour()
        barycenter  = self._neck_barycenter()

        # Get point in which distance to neck line baricenter is maximum
        maxDistance = float(_intZero)
        maxVertex   = None

        nVertices   = self.aneurysmSurface.GetPoints().GetNumberOfPoints()

        for index in range(nVertices):
            vertex = self.aneurysmSurface.GetPoint(index)

            # Compute distance between point and neck barycenter
            distanceSquared = vtk.vtkMath.Distance2BetweenPoints(barycenter, vertex)
            distance = np.sqrt(distanceSquared)

            if distance > maxDistance: 
                maxDistance = distance
                maxVertex = vertex

        return np.array(maxVertex) - barycenter


    def _contour_perimeter(self,contour):
        """
            Returns the perimeter of a vtkPolyData 1D 
            contour defined in 3D space.
        """

        nContourVertices = contour.GetNumberOfPoints()

        # Compute neck perimeter
        perimeter = _intZero
        previous  = contour.GetPoint(_intZero)

        for index in range(nContourVertices):
            if index > _intZero:
                previous = contour.GetPoint(index - _intOne)

            vertex = contour.GetPoint(index)

            # Compute distance between two consecutive points
            distanceSquared = vtk.vtkMath.Distance2BetweenPoints(previous, vertex)
            increment = np.sqrt(distanceSquared)    
            
            perimeter += increment
    
        return perimeter
    
    
    def _contour_hydraulic_diameter(self,contour):
        """
            Computes the hydraulic diameter of a cross 
            section, provided its poly line contour.
        """
        
        contourPerimeter = self._contour_perimeter(contour)

            
        # Compute contour fill area
        fillContour = vtk.vtkContourTriangulator()
        fillContour.SetInputData(contour)
        fillContour.Update()

        computeArea = vtk.vtkMassProperties()
        computeArea.SetInputData(fillContour.GetOutput())
        computeArea.Update()
        
        contourArea = computeArea.GetSurfaceArea()

        # Compute hydraulic diameter of neck
        return _intFour * contourArea/contourPerimeter


    def _neck_plane_normal_vector(self):
        """
            Returns the normal vector to the aneurysm neck plane,
            as a Numpy array
        """
        # Get neck plane surface
        neckPlaneSurface = self._neck_surface()
        
        # Compute neck plane normal
        normals = vtk.vtkPolyDataNormals()
        normals.SetInputData(neckPlaneSurface)
        normals.ComputeCellNormalsOn()
        normals.ComputePointNormalsOff()
        normals.Update()

        xNormal = normals.GetOutput().GetCellData().GetNormals().GetRange(0)[0]
        yNormal = normals.GetOutput().GetCellData().GetNormals().GetRange(1)[0]
        zNormal = normals.GetOutput().GetCellData().GetNormals().GetRange(2)[0]

        return np.array([xNormal, yNormal, zNormal])



    
    
    # 1D Size Indices
    def neckDiameter(self):
        """
            Compute neck diameter, defined as the hydraulic diameter
            of the neck plane section:

                Dn = 4*An/pn

            where An is the aneurysm neck section area and pn is its
            perimeter.
         """   

        # Get lenght of boundary neck (validate in ParaView)
        neckContour   = self._neck_contour()
        neckPerimeter = self._contour_perimeter(neckContour)
        
        # Compute hydraulic diameter of neck
        return _intFour * self.neckPlaneArea/neckPerimeter


    def maximumHeight(self):
        """ 
            Computation of the maximum aneurysm height, 
            defined as the maximum distance between the 
            neck barycenter and the aneurysm surface.
        """
        # Get neck contour
        vec = self._max_height_vector()
        return np.linalg.norm(vec)


    def maximumNormalHeight(self):
        """ 
            Computation of the maximum NORMAL aneurysm 
            height, defined as the maximum distance between 
            the neck barycenter and the aneurysm surface.
        """

        # Get max height vector and neck plane normal vector 
        vecMaxHeight = self._max_height_vector()
        vecNormal    = self._neck_plane_normal_vector()

        return abs(vtk.vtkMath.Dot(vecMaxHeight,vecNormal))


    def maximumDiameter(self):
        """
            Computattion of the maximum section diameter of the aneurysm,
            defined as the maximum diameter of the aneurysm cross sections
            that are parallel to the neck plane.
        """
        # Compute neck contour barycenter and normal vector
        normal     = self._neck_plane_normal_vector()
        barycenter = self._neck_barycenter()

        
        # Get maximum normal height
        Hnmax = self.maximumNormalHeight()

        # Form points of perpendicular line to neck plane
        nPoints    = _intThree * _intTen
        dimensions = _intThree

        t = np.linspace(0, Hnmax, nPoints)
        parameters = np.array([t]*dimensions).T

        points = barycenter + parameters * normal

        # Computes minimum hydraulic diameter
        maxDiameter = 0.0

        for center in points:
            plane = vtk.vtkPlane()
            plane.SetOrigin(center)
            plane.SetNormal(normal)

            # Cut initial aneurysm surface with create plane
            cutWithPlane = vtk.vtkCutter()
            cutWithPlane.SetInputData(self.aneurysmSurface)
            cutWithPlane.SetCutFunction(plane)
            cutWithPlane.Update()

            nVertices = cutWithPlane.GetOutput().GetNumberOfPoints()

            # Compute diamenetr if contour is not empty
            if nVertices > _intZero:

                # Compute hydraulic diameter of cut line
                hydraulicDiameter = self._contour_hydraulic_diameter(cutWithPlane.GetOutput())

                # Update minmum area 
                if hydraulicDiameter > maxDiameter: 
                    maxDiameter = hydraulicDiameter

        return maxDiameter    
    
    
    # 2D Shape indices
    def aspectRatio(self):
        """
            Computes the aneurysm aspect ratio, defined as the 
            ratio between the maximum perpendicular height and
            the neck diameter. 

        """
        
        return self.maximumNormalHeight()/self.neckDiameter()
    
    
    def bottleneckFactor(self):
        """
            Computes the bottleneck factor, defined as the 
            ratio between the maximum diameter and the neck
            diameter. This index represents the level 
            to which the neck acts as a bottleneck to entry of 
            blood during normal physiological function and to 
            coils during endovascular procedures. 
        """
        
        return self.maximumDiameter()/self.neckDiameter()
    
    
    
    # 3D Shape indices
    def nonSphericityIndex(self):
        """ Computes the non-sphericity index of an aneurysm 
            surface, given by:

                NSI = 1 - (18pi)^(1/3) * Va^(2/3)/Sa

            where Va and Sa are the volume and surface area of the
            aneurysm.
        """
        factor = (18*np.pi)**(1./3.)
        
        return 1 - factor/self.surfaceArea*(self.volume**(2./3.))
    
    
    def ellipticityIndex(self):
        """ Computes the ellipiticity index of an aneurysm 
            surface, given by:

                EI = 1 - (18pi)^(1/3) * Vch^(2/3)/Sch

            where Vch and Sch are the volume and surface area 
            of the aneurysm convex hull.
        """
        
        
        factor = (18*np.pi)**(1./3.)
        
        return _intOne - factor/self.hullArea*(self.hullVolume**(2./3.))
    
    def undulationIndex(self):
        """
            Computes the undulation index of an aneurysm,
            defined as:
                
                UI = 1 - Va/Vch
            
            where Va is the aneurysm volume and Vch the
            volume of its convex hull.
        """
        
        return _intOne - self.volume/self.hullVolume



if __name__ == '__main__':
    
    # Input arguments
    writeDir = sys.argv[1]        #parentVesselDir + caseId+"/"
    caseId   = sys.argv[2]

    # Input files
    surfaceModelFile   = writeDir + caseId+"_model.vtp"
    parentVesselFile   = writeDir + caseId+"_reconstructedmodel.vtp"
    clippingPointsFile = writeDir + caseId+'_clippingpoints.vtp'

    # Filenames of intermediate structures
    modelVoronoiFileName = writeDir + caseId+"_voronoi.vtp"
    neckPlaneFileName    = writeDir + caseId+'_aneurysm_neckplane.vtp'
    aneurysmPlaneClipped = writeDir + caseId+'_aneurysm_plane_clipped.vtp'


    aneurysmSurfaceModel = _read_surface(surfaceModelFile)
    parentVesselSurface  = _read_surface(parentVesselFile)
    clippingPointsData   = _read_surface(clippingPointsFile)

    # Smooth
    parentVesselSmooth = _smooth_surface(parentVesselSurface)
    parentCenterlines  = _generate_centerline(parentVesselSmooth)    

    aneurysmPlaneClippedSurface = aneurysmNeckPlane(aneurysmSurfaceModel,
                                                    parentCenterlines,
                                                    clippingPointsData)

    _write_surface(aneurysmPlaneClippedSurface,aneurysmPlaneClipped)