"""Provide function to calculate the aneurysm neck plane.

The module provides a function to compute the aneurysm neck plane,
as defined by Piccinelli et al. (2009).
"""

import sys
import vtk
import numpy as np
import centerlines
import morphman

from vmtk import vtkvmtk
from vmtk import vmtkscripts
from scipy import interpolate

from vtk.numpy_interface import dataset_adapter as dsa

# Local modules
from constants import *
import polydatatools as tools
import polydatageometry as geo

# Flags
_write = False
_debug = False
_inspect = False

_write_dir = '/home/iagolessa/'

INCR = 0.01
HUGE = 1e30

_dimensions = intThree
_xComp = intZero
_yComp = intOne
_zComp = intThree


def _vtk_vertices_to_numpy(polydata):
    """Convert vtkPolyData object to Numpy nd-array.

    Return the point coordinates of a vtkPolyData object as a n-dimensional
    Numpy nd-array. It only extracts its points, ignoring any array or field in
    the vtkPolyData.

    Arguments:
    polydata -- the vtkPolyData
    """
    vertices = []
    nPoints = polydata.GetNumberOfPoints()

    for index in range(nPoints):
        vertex = polydata.GetPoint(index)
        vertices.append(vertex)

    return np.array(vertices)


def _rotate3d_matrix(tilt, azim):
    """Rotation matrix traformation for a vector in 3D space."""
    return np.array([[np.cos(azim),             -np.sin(azim),      intZero],
                     [np.sin(azim)*np.cos(tilt), np.cos(azim)
                      * np.cos(tilt), -np.sin(tilt)],
                     [np.sin(azim)*np.sin(tilt), np.cos(azim)*np.sin(tilt),  np.cos(tilt)]])


def _compute_Voronoi(surface_model):
    """Compute Voronoi diagram of a vascular surface."""
    voronoiDiagram = vmtkscripts.vmtkDelaunayVoronoi()
    voronoiDiagram.Surface = surface_model
    voronoiDiagram.CheckNonManifold = True
    voronoiDiagram.Execute()

    return voronoiDiagram.Surface


def _tube_surface(centerline, smooth=True):
    """Reconstruct tube surface of a given vascular surface.

    The tube surface is the maximum tubular structure ins- cribed in the
    vasculature. 

    Arguments:
        centerline -- the centerline to compute the tube surface with the radius
            array.

    Keyword arguments:
        smooth -- to smooth tube surface (default True)
    """
    # List to collect objects
    objects = []

    radiusArray = 'MaximumInscribedSphereRadius'

    # TODO: adapt the bpox dimension to the input geometry
    boxDimensions = _dimensions*[128]

    # Build tube function of the centerlines
    # TODO: what happens if the centerline does not have a radius array?
    tubeImage = vmtkscripts.vmtkCenterlineModeller()
    tubeImage.Centerlines = centerline
    tubeImage.RadiusArrayName = radiusArray
    tubeImage.SampleDimensions = boxDimensions
    tubeImage.Execute()
    objects.append(tubeImage)

    # Convert tube function to surface
    tubeSurface = vmtkscripts.vmtkMarchingCubes()
    tubeSurface.Image = tubeImage.Image
    tubeSurface.Execute()
    objects.append(tubeSurface)

    tube = tools.ExtractConnectedRegion(tubeSurface.Surface, 'largest')

    if _debug:
        for obj in objects:
            obj.PrintInputMembers()
            obj.PrintOutputMembers()

    if _inspect:
        tools.ViewSurface(tube)

    if smooth:
        return tools.SmoothSurface(tube)
    else:
        return tube


def _clip_aneurysm_Voronoi(VoronoiSurface, tubeSurface):
    """Extract the Voronoi diagram of the aneurysmal portion."""

    # Compute distance between complete Voronoi
    # and the parent vessel tube surface
    DistanceArrayName = 'DistanceToTubeArray'
    VoronoiDistance = tools.ComputeSurfacesDistance(
        VoronoiSurface,
        tubeSurface,
        array_name=DistanceArrayName
    )

    # Clip the original voronoi diagram at
    # the zero distance (intersection)
    VoronoiClipper = vmtkscripts.vmtkSurfaceClipper()
    VoronoiClipper.Surface = VoronoiDistance
    VoronoiClipper.Interactive = False
    VoronoiClipper.ClipArrayName = DistanceArrayName
    VoronoiClipper.ClipValue = intZero
    VoronoiClipper.InsideOut = True
    VoronoiClipper.Execute()

    aneurysmVoronoi = tools.ExtractConnectedRegion(VoronoiClipper.Surface, 'largest')

    if _inspect:
        tools.ViewSurface(aneurysmVoronoi)

    return tools.Cleaner(aneurysmVoronoi)


def _Voronoi_envelope(Voronoi):
    """Compute the envelope surface of a Voronoi diagram."""

    # List to collect objects
    objects = list()

    radiusArray = 'MaximumInscribedSphereRadius'

    VoronoiBounds = Voronoi.GetBounds()
    radiusArrayBounds = Voronoi.GetPointData().GetArray(radiusArray).GetValueRange()
    maxSphereRadius = radiusArrayBounds[intOne]
    enlargeBoxBounds = (intFour/intTen)*maxSphereRadius

    modelBounds = np.array(VoronoiBounds) + \
        np.array(_dimensions*[-enlargeBoxBounds, enlargeBoxBounds])

    # Building the envelope image function
    envelopeFunction = vmtkscripts.vmtkPolyBallModeller()
    envelopeFunction.Surface = Voronoi
    envelopeFunction.RadiusArrayName = radiusArray
    envelopeFunction.ModelBounds = list(modelBounds)
    envelopeFunction.Execute()
    objects.append(envelopeFunction)

    # Get level zero surface
    envelopeSurface = vmtkscripts.vmtkMarchingCubes()
    envelopeSurface.Image = envelopeFunction.Image
    envelopeSurface.Execute()
    objects.append(envelopeSurface)

    envelope = tools.ExtractConnectedRegion(envelopeSurface.Surface, 'largest')

    if _debug:
        for obj in objects:
            obj.PrintInputMembers()
            obj.PrintOutputMembers()

    if _inspect:
        tools.ViewSurface(envelope)

    return tools.SmoothSurface(envelope)


def _clip_initial_aneurysm(surface_model, aneurysm_envelope, parent_tube):
    """Clip initial aneurysm surface from the original vascular model.

    Compute distance between the aneurysm envelope and parent vasculature tube
    function from the original vascular surface model. Clip the surface at the
    zero value of the difference between these two fields.

    Arguments:
        surface_model --  the original vascular surface
        aneuysm_envelope -- the aneurysm surface computed from its Voronoi
        parent_tube -- tube surface of the parent vessel
    """

    # Array names
    clipAneurysmArray = 'ClipInitialAneurysmArray'
    tubeToModelArray = 'ParentTubeModelDistanceArray'
    envelopeToModelArray = 'AneurysmEnvelopeModelDistanceArray'

    # Computes distance between original surface model
    # and the aneurysm envelope, and from the parent
    # tube surface
    aneurysmEnvelopeDistance = tools.ComputeSurfacesDistance(
                                    surface_model,
                                    aneurysm_envelope,
                                    array_name=envelopeToModelArray
                                )

    modelSurfaceWithDistance = tools.ComputeSurfacesDistance(
                                    aneurysmEnvelopeDistance,
                                    parent_tube,
                                    array_name=tubeToModelArray
                                )

    # Compute difference between the arrays
    clippingArray = vmtkscripts.vmtkSurfaceArrayOperation()
    clippingArray.Surface = modelSurfaceWithDistance
    clippingArray.Operation = 'subtract'
    clippingArray.InputArrayName = envelopeToModelArray
    clippingArray.Input2ArrayName = tubeToModelArray
    clippingArray.ResultArrayName = clipAneurysmArray
    clippingArray.Execute()

    # Clip the model surface at the zero level of the difference array
    clipAneurysm = vmtkscripts.vmtkSurfaceClipper()
    clipAneurysm.Surface = clippingArray.Surface
    clipAneurysm.ClipArrayName = clippingArray.ResultArrayName
    clipAneurysm.ClipValue = intZero
    clipAneurysm.Interactive = False
    clipAneurysm.InsideOut = False
    clipAneurysm.Execute()

    aneurysm = tools.ExtractConnectedRegion(clipAneurysm.Surface, 'largest')

    # Remove fields
    aneurysm.GetPointData().RemoveArray(clipAneurysmArray)
    aneurysm.GetPointData().RemoveArray(tubeToModelArray)
    aneurysm.GetPointData().RemoveArray(envelopeToModelArray)

    aneurysm = tools.Cleaner(aneurysm)

    if _inspect:
        tools.ViewSurface(aneurysm)

    if _write:
        tools.WriteSurface(aneurysm, _write_dir+'initial_aneurysm_surface.vtp')

    return aneurysm


def _sac_centerline(aneurysm_sac, distance_array):
    """Compute aneurysm sac centerline.

    Compute spline that travels alongs the aneurysm sac from the intersection
    with the pa- rent vessel tube. Its points are defined by the geometric
    place of the barycenters of iso- contours of a distance_array defined on
    the aneurysm surface.

    The function returns the spline vertices in a Numpy nd-array.
    """

    # Get wrapper object of vtk numpy interface
    surfaceWrapper = dsa.WrapDataObject(aneurysm_sac)
    distanceArray = np.array(surfaceWrapper.PointData.GetArray(distance_array))

    minTubeDist = float(distanceArray.min())
    maxTubeDist = float(distanceArray.max())

    # Build spline along with to perform the neck search

    nPoints = intOne*intHundred   # number of points to perform search
    barycenters = list()              # barycenter's list

    aneurysm_sac.GetPointData().SetActiveScalars(distance_array)

    # Get barycenters of iso-contours
    for isovalue in np.linspace(minTubeDist, maxTubeDist, nPoints):

        # Get isocontour polyline
        isoContour = vtk.vtkContourFilter()
        isoContour.SetInputData(aneurysm_sac)
        isoContour.ComputeScalarsOff()
        isoContour.ComputeNormalsOff()
        isoContour.SetValue(intZero, isovalue)
        isoContour.Update()

        # Get largest connected contour
        contour = tools.ExtractConnectedRegion(isoContour.GetOutput(), 'largest')

        try:
            contourPoints = contour.GetPoints()
            nContourPoints = contour.GetNumberOfPoints()

            if geo.ContourIsClosed(contour) and nContourPoints != intZero:
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
        increment = center - previous
        distance += np.linalg.norm(increment, intTwo)
        distanceCoord.append(distance)

    distanceCoord = np.array(distanceCoord)

    # Find spline of barycenters and get derivative == normals
    limitFraction = intSeven/intTen

    tck, u = interpolate.splprep(barycenters.T, u=distanceCoord)

    # Max and min t of spline
    minSplineDomain = min(u)-intFive/intTen
    maxSplineDomain = limitFraction*max(u)

    domain = np.linspace(minSplineDomain, maxSplineDomain, intTwo*nPoints)

    deriv0 = interpolate.splev(domain, tck, der=0)
    deriv1 = interpolate.splev(domain, tck, der=1)

    points = np.array(deriv0).T            # spline points
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

        tools.WriteSurface(spline, _write_dir+'sac_spline.vtp')

    return points, tangents

def _local_minimum(array):
    """Find local minimum closest to beginning of array.

    Given an array of real numbers, return the id of the smallest value closest
    to the beginning of the array.
    """
    minimum = np.r_[True, array[1:] < array[:-1]] & \
              np.r_[array[:-1] < array[1:], True]

    # Local minima index
    return int(np.where(minimum == True)[intZero][intZero])


def _search_neck_plane(anerysm_sac, centers, normals, min_variable='area'):
    """Search neck plane of aneurysm by minimizing a contour variable.

    This function effectively searches for the aneurysm neck plane: it
    interactively cuts the aneurysm surface with planes defined by the vertices
    and normals to a spline travelling through the aneurysm sac. 

    The cut plane is further precessed by a tilt and azimuth angle and the
    minimum search between them, as originally proposed by Piccinelli et al.
    (2009).

    It returns the local minimum solution: the neck plane as a vtkPlane object.
    """
    # Rotation angles
    tiltIncr = intOne
    azimIncr = intTen
    tiltMax = 32
    azimMax = 360

    tilts = np.arange(intZero, tiltMax, tiltIncr) * degToRad
    azims = np.arange(intZero, azimMax, azimIncr) * degToRad

    # Minimum area seacrh
    sectionInfo = list()
    previousVariable = HUGE

    # Iterate over barycenters and clip surface with closed surface
    for index, (center, normal) in enumerate(zip(centers, normals)):

        # Store previous area to compare and find local minimum
        if index > intZero:
            previousVariable = minVariable

        # Iterate over rotated planes
        # identifies the minimum area
        minArea = HUGE
        minPlane = None

        minHDiameter = HUGE
        minPerimeter = HUGE
        minVariable = HUGE

        for tilt in tilts:
            for azim in azims:

                # Set plane with barycenters and
                # normals as tangent to spline
                plane = vtk.vtkPlane()
                plane.SetOrigin(tuple(center))

                # Rotate normal
                matrix = _rotate3d_matrix(tilt, azim)
                rNormal = np.dot(matrix, normal)

                # Set rotate plane normal
                plane.SetNormal(tuple(rNormal))

                # Cut initial aneurysm surface with create plane
                cutWithPlane = vtk.vtkCutter()
                cutWithPlane.SetInputData(anerysm_sac)
                cutWithPlane.SetCutFunction(plane)
                cutWithPlane.Update()

                contour = tools.ExtractConnectedRegion(
                    cutWithPlane.GetOutput(), 'largest')

                try:
                    contourPoints = contour.GetPoints()
                    nContourPoints = contour.GetNumberOfPoints()

                    if geo.ContourIsClosed(contour) and nContourPoints != intZero:

                        # Update minmum area
                        if min_variable == 'area':
                            variable = geo.ContourPlaneArea(contour)

                        elif min_variable == 'perimeter':
                            variable = geo.ContourPerimeter(contour)

                        elif min_variable == 'hyd_diameter':
                            variable = intFour*area/perimeter

                        else:
                            print('Minimizing variable not recognized!'
                                  'Choose area, perimeter or hyd_diameter.')

                        if variable < minVariable:
                            minVariable = variable
                            minPlane = plane
                except:
                    continue

        # Write to array min area and its surface plane
        if minVariable != HUGE:
            if minVariable > previousVariable:
                break

            sectionInfo.append([minVariable, minPlane])

    sectionInfo = np.array(sectionInfo)

    # Get local minimum area
    areas = sectionInfo[:, intZero]

    minimumId = _local_minimum(areas)

    return sectionInfo[minimumId, intOne]


def aneurysmNeckPlane(surface_model,
                      parent_centerlines=None,
                      clipping_points=None,
                      min_variable='perimeter'):
    """Extracts the aneurysm neck plane.

    Procedure based on Piccinelli's pipeline, which is based on the surface
    model with the aneurysm and its parent vasculature reconstruction. The
    single difference is the variable which the algorithm minimizes to search
    for the neck plane: the default is the neck perimeter, whereas in the
    default procedure is the neck section area; this can be controlled by the
    optional argument 'min_variable'.

    It returns the clipped aneurysm surface from the original vasculature.

    Arguments
    ---------
        surface_model -- the original vasculature surface with the 
            aneurysm
        parent_centerlines -- the centerlines of the reconstructed 
            parent vasculature
        clipping_points -- points where the vasculature will be 
            clipped.

    Optional args
        min_variable -- the varible by which the neck will be searched
            (default 'perimeter'; options 'perimeter' 'area')
    """
    # Variables
    tubeToAneurysmDistance = 'ClippedTubeToAneurysmDistanceArray'

    # Compute vasculature centerline
    # TODO: update this to use the parent centerlines with the radius array
    parent_centerlines = centerlines.GenerateCenterlines(surface_model)

    # Get clipping and diverging data
    divergingData = centerlines.GetDivergingPoints(surface_model)

    # Reconstruct tube functions
    parentTube = _tube_surface(parent_centerlines)

    # Clip centerlines between clipping points
    clipped_centerline = morphman.get_centerline_between_clipping_points(
                            parent_centerlines, 
                            divergingData
                        )

    clippedTube = _tube_surface(clipped_centerline)

    # Clip aneurysm Voronoi
    VoronoiDiagram = _compute_Voronoi(surface_model)
    aneurysmVoronoi = _clip_aneurysm_Voronoi(VoronoiDiagram, parentTube)

    aneurysmEnvelope = _Voronoi_envelope(aneurysmVoronoi)

    initialAneurysm = _clip_initial_aneurysm(surface_model,
                                             aneurysmEnvelope,
                                             parentTube)

    initialAneurysmSurface = tools.ComputeSurfacesDistance(initialAneurysm,
                                                        clippedTube,
                                                        array_name=tubeToAneurysmDistance,
                                                        signed_array=False)

    barycenters, normals = _sac_centerline(initialAneurysmSurface,
                                           tubeToAneurysmDistance)

    # Search for neck plane
    neckPlane = _search_neck_plane(initialAneurysmSurface,
                                   barycenters,
                                   normals,
                                   min_variable=min_variable)

    neckCenter = neckPlane.GetOrigin()
    neckNormal = neckPlane.GetNormal()

    # Remove distance array
    initialAneurysmSurface.GetPointData().RemoveArray(tubeToAneurysmDistance)

    # Clip final aneurysm surface: when clipped, two surfaces the aneurysm (desired) and
    # the rest (not desired) which is closest to the clipped tube
    surf1 = tools.ClipWithPlane(initialAneurysmSurface, neckCenter, neckNormal)
    surf2 = tools.ClipWithPlane(initialAneurysmSurface,
                             neckCenter, neckNormal, inside_out=True)

    # Check which output is farthest from clipped tube
    tubePoints = _vtk_vertices_to_numpy(clippedTube)
    surf1Points = _vtk_vertices_to_numpy(surf1)
    surf2Points = _vtk_vertices_to_numpy(surf2)

    tubeCentroid = tubePoints.mean(axis=0)
    surf1Centroid = surf1Points.mean(axis=0)
    surf2Centroid = surf2Points.mean(axis=0)

    surf1Distance = vtk.vtkMath.Distance2BetweenPoints(
        tubeCentroid, surf1Centroid)
    surf2Distance = vtk.vtkMath.Distance2BetweenPoints(
        tubeCentroid, surf2Centroid)

    if surf1Distance > surf2Distance:
        aneurysmSurface = surf1
    else:
        aneurysmSurface = surf2

    return aneurysmSurface
