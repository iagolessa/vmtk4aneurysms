"""Provide function to calculate the aneurysm neck plane.

The module provides a function to compute the aneurysm neck plane, as defined
by Piccinelli et al. (2009).
"""

import sys
import vtk
import numpy as np
import morphman as mp

from vmtk import vtkvmtk
from vmtk import vmtkscripts
from scipy import interpolate

from vtk.numpy_interface import dataset_adapter as dsa

# Local modules
from .lib import names
from .lib import centerlines as cnt
from .lib import constants as const
from .lib import polydatatools as tools
from .lib import polydatageometry as geo

INCR = 0.01
HUGE = 1e30
_dimensions = int(const.three)

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
    return np.array([[np.cos(azim),             -np.sin(azim),      const.zero],
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
    radiusArray = 'MaximumInscribedSphereRadius'

    # Get bounds of model
    centerlineBounds = centerline.GetBounds()
    radiusArrayBounds = centerline.GetPointData().GetArray(radiusArray).GetValueRange()
    maxSphereRadius = radiusArrayBounds[1]
    enlargeBoxBounds = (const.ten/const.ten)*maxSphereRadius

    modelBounds = np.array(centerlineBounds) + \
                  np.array(_dimensions*[-enlargeBoxBounds, enlargeBoxBounds])

    # Extract image with tube function from model
    modeller = vtkvmtk.vtkvmtkPolyBallModeller()
    modeller.SetInputData(centerline)
    modeller.SetRadiusArrayName(radiusArray)

    # This needs to be 'on' for centerline
    modeller.UsePolyBallLineOn()

    modeller.SetModelBounds(list(modelBounds))
    modeller.SetNegateFunction(0)
    modeller.Update()

    tubeImage = modeller.GetOutput()

    # Convert tube function to surface
    tubeSurface = vmtkscripts.vmtkMarchingCubes()
    tubeSurface.Image = tubeImage
    tubeSurface.Execute()

    tube = tools.ExtractConnectedRegion(tubeSurface.Surface, 'largest')

    if smooth:
        return tools.SmoothSurface(tube)
    else:
        return tube

# This function was adapted from the Morphman library,
# available at https://github.com/KVSlab/morphMan
def _get_diverging_point(
        centerline: names.polyDataType,
        tol: float
    )   -> tuple:
    """Get diverging point on centerline bifurcation.

    Args:
        centerline1 (vtkPolyData): Centerline with two segments.
        tol (float): Tolerance.
    Returns:
        point (tuple): ID at diverging point.
    """
    line0 = mp.extract_single_line(centerline, 0)
    line1 = mp.extract_single_line(centerline, 1)

    # Find clipping points
    nPoints = min(line0.GetNumberOfPoints(), line1.GetNumberOfPoints())

    bifPointIndex = None
    getPoint0 = line0.GetPoints().GetPoint
    getPoint1 = line1.GetPoints().GetPoint

    for index in range(0, nPoints):
        distance = geo.Distance(getPoint0(index), getPoint1(index))

        if distance > tol:
            bifPointIndex = index
            break

    return getPoint0(bifPointIndex)

def _bifurcation_aneurysm_influence_region(
        vascular_surface: names.polyDataType,
        aneurysm_point: tuple
    )   -> names.polyDataType:
    """Extract vessel portion where a bifurcation aneurysm grew.

    Given the vascular model surface with open inlet and outlet profiles,
    extract the portion of the tube surface where the aneurysm grew by
    calculating the divegence points of the centerlines. The user must select a
    point on the aneurysm's dome surface.

    Note that the algorithm will use the two first outlets to compute the
    centerlines, so avoid any outlet profile between the inlet and the
    aneurysm.
    """
    inlets, outlets = cnt.ComputeOpenCenters(vascular_surface)

    # Tolerance distance to identify the bifurcation
    divTolerance = 0.01

    # One inlet and two outlets, bifurcation with the aneurysm
    # 1 -> centerline of the branches only
    # 2 -> centerline of the first outlet to the aneurysm and inlet
    # 3 -> centerline of the second outlet to the aneurysm and inlet
    relevantOutlets = outlets[0:2]

    clWithoutAneurysm = cnt.GenerateCenterlines(
                            vascular_surface,
                            inlets,
                            relevantOutlets
                        )

    bifClippingPoint = _get_diverging_point(clWithoutAneurysm, divTolerance)

    lines = []
    for cl_id, outlet in enumerate(relevantOutlets):
        daughterCenterline = cnt.GenerateCenterlines(
                                 vascular_surface,
                                 [outlet],
                                 inlets + [aneurysm_point]
                             )
        # Get clipping point on this branch
        dauClippingPoint = _get_diverging_point(daughterCenterline, divTolerance)

        # Then clip the parent centerline
        line = mp.extract_single_line(clWithoutAneurysm, cl_id)

        loc = mp.get_vtk_point_locator(line)

        #Find closest points to clipping on parent centerline
        dauId = loc.FindClosestPoint(dauClippingPoint)
        bifId = loc.FindClosestPoint(bifClippingPoint)

        lines.append(
            mp.extract_single_line(
                line,
                0,
                start_id=bifId,
                end_id=dauId
            )
        )

    aneurysmInceptionClPortion = mp.vtk_merge_polydata(lines)

    return _tube_surface(aneurysmInceptionClPortion)

def _lateral_aneurysm_influence_region(
        vascular_surface: names.polyDataType,
        aneurysm_point: tuple
    )   -> names.polyDataType:
    """Extract vessel portion where a lateral aneurysm grew.

    Given the vascular model surface with open inlet and outlet
    profiles, extract the portion of the tube surface where the
    aneurysm grew by calculating the divegence points of the
    centerlines. The user must select a point on the aneurysm's
    dome surface.

    Note that the algorithm will use the first outlet to
    compute the centerlines, so avoid any outlet profile
    between the inlet and the aneurysm region.
    """
    inlets, outlets = cnt.ComputeOpenCenters(vascular_surface)

    # Tolerance distance to identify the bifurcation
    divTolerance = 0.01

    # One inlet and one outlet (although the model can have more than one outlet),
    # lateral aneurysm
    # 1 -> "forward" centerline, inlet -> outlet, and aneurysm
    # 2 -> "backward" centerline, outlet -> inlet, with aneurysm
    # Note: the aneurysm is like a bifurcation, in this case

    relevantOutlets = outlets[0:1]

    forwardCenterline = cnt.GenerateCenterlines(
                            vascular_surface,
                            inlets,
                            relevantOutlets + [aneurysm_point]
                        )

    backwardCenterline = cnt.GenerateCenterlines(
                            vascular_surface,
                            relevantOutlets,
                            inlets + [aneurysm_point]
                        )

    upstreamClipPoint   = _get_diverging_point(forwardCenterline, divTolerance)
    downstreamClipPoint = _get_diverging_point(backwardCenterline, divTolerance)

    # Clip centerline portion of the forward centerline
    line = mp.extract_single_line(forwardCenterline, 0)
    loc  = mp.get_vtk_point_locator(line)

    #Find closest points to clipping on parent centerline
    upstreamId   = loc.FindClosestPoint(upstreamClipPoint)
    downstreamId = loc.FindClosestPoint(downstreamClipPoint)

    aneurysmInceptionClPortion =  mp.extract_single_line(
                                        line,
                                        0,
                                        start_id=upstreamId,
                                        end_id=downstreamId
                                    )

    return _tube_surface(aneurysmInceptionClPortion)


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

    # Clip the original voronoi diagram at the zero distance (intersection)
    # VoronoiClipper = vmtkscripts.vmtkSurfaceClipper()
    # VoronoiClipper.Surface = VoronoiDistance
    # VoronoiClipper.Interactive = False
    # VoronoiClipper.ClipArrayName = DistanceArrayName
    # VoronoiClipper.ClipValue = const.zero
    # VoronoiClipper.InsideOut = True
    # VoronoiClipper.Execute()

    aneurysmVoronoi = tools.ClipWithScalar(
                        VoronoiDistance,
                        DistanceArrayName,
                        const.zero
                    )

    aneurysmVoronoi = tools.ExtractConnectedRegion(
                        aneurysmVoronoi,
                        'largest'
                    )

    return tools.Cleaner(aneurysmVoronoi)


def _Voronoi_envelope(Voronoi):
    """Compute the envelope surface of a Voronoi diagram."""

    radiusArray = 'MaximumInscribedSphereRadius'

    VoronoiBounds = Voronoi.GetBounds()
    radiusArrayBounds = Voronoi.GetPointData().GetArray(radiusArray).GetValueRange()
    maxSphereRadius = radiusArrayBounds[1]
    enlargeBoxBounds = (const.four/const.ten)*maxSphereRadius

    modelBounds = np.array(VoronoiBounds) + \
                  np.array(_dimensions*[-enlargeBoxBounds, enlargeBoxBounds])

    # Building the envelope image function
    modeller = vtkvmtk.vtkvmtkPolyBallModeller()
    modeller.SetInputData(Voronoi)
    modeller.SetRadiusArrayName(radiusArray)

    # This needs to be off for surfaces
    modeller.UsePolyBallLineOff()

    modeller.SetModelBounds(list(modelBounds))
    modeller.SetNegateFunction(0)
    modeller.Update()

    envelopeImage = modeller.GetOutput()

    # Get level zero surface
    envelopeSurface = vmtkscripts.vmtkMarchingCubes()
    envelopeSurface.Image = envelopeImage
    envelopeSurface.Execute()

    envelope = tools.ExtractConnectedRegion(
                    envelopeSurface.Surface,
                    'largest'
                )

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

    # Computes distance between original surface model and the aneurysm
    # envelope, and from the parent tube surface
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

    clippedAneurysm = tools.ClipWithScalar(
                        clippingArray.Surface,
                        clippingArray.ResultArrayName,
                        const.zero,
                        inside_out=False
                    )

    aneurysm = tools.ExtractConnectedRegion(clippedAneurysm, 'largest')

    # Remove fields
    aneurysm.GetPointData().RemoveArray(clipAneurysmArray)
    aneurysm.GetPointData().RemoveArray(tubeToModelArray)
    aneurysm.GetPointData().RemoveArray(envelopeToModelArray)

    return tools.Cleaner(aneurysm)


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
    nPoints = int(const.one)*int(const.oneHundred)
    barycenters = []

    aneurysm_sac.GetPointData().SetActiveScalars(distance_array)

    # Get barycenters of iso-contours
    for isovalue in np.linspace(minTubeDist, maxTubeDist, nPoints):

        # Get isocontour polyline
        isoContour = vtk.vtkContourFilter()
        isoContour.SetInputData(aneurysm_sac)
        isoContour.ComputeScalarsOff()
        isoContour.ComputeNormalsOff()
        isoContour.SetValue(0, isovalue)
        isoContour.Update()

        # Get largest connected contour
        contour = tools.ExtractConnectedRegion(
                        isoContour.GetOutput(),
                        'largest'
                    )

        try:
            contourPoints = contour.GetPoints()
            nContourPoints = contour.GetNumberOfPoints()

            if geo.ContourIsClosed(contour) and nContourPoints != 0:
                barycenter = _dimensions*[0]
                vtkvmtk.vtkvmtkBoundaryReferenceSystems.ComputeBoundaryBarycenter(
                    contourPoints,
                    barycenter
                )

                barycenters.append(barycenter)

        except:
            continue

    barycenters = np.array(barycenters)

    # Compute list with distance coordinate along spline
    distance = const.zero
    previous = barycenters[0]
    distanceCoord = list()                    # path distance coordinate

    for index, center in enumerate(barycenters):

        if index > 0:
            previous = barycenters[index - 1]

        # Interpoint distance
        increment = center - previous
        distance += np.linalg.norm(increment, 2)
        distanceCoord.append(distance)

    distanceCoord = np.array(distanceCoord)

    # Find spline of barycenters and get derivative == normals
    limitFraction = const.seven/const.ten

    tck, u = interpolate.splprep(barycenters.T, u=distanceCoord)

    # Max and min t of spline
    # Note that we decrease by two units the start of the spline
    # because I noticed that for some cases the initial point of the spline
    # might be pretty inside the aneurysm, skipping the "neck region"
    minSplineDomain = min(u) - const.one #intFive/const.ten
    maxSplineDomain = limitFraction*max(u)

    domain = np.linspace(minSplineDomain, maxSplineDomain, 2*nPoints)

    deriv0 = interpolate.splev(domain, tck, der=0)
    deriv1 = interpolate.splev(domain, tck, der=1)

    points = np.array(deriv0).T            # spline points
    tangents = np.array(deriv1).T            # spline tangents

    return points, tangents

def _local_minimum(array):
    """Find local minimum closest to beginning of array.

    Given an array of real numbers, return the id of the smallest value closest
    to the beginning of the array.
    """
    minimum = np.r_[True, array[1:] < array[:-1]] & \
              np.r_[array[:-1] < array[1:], True]

    # Local minima index
    return int(np.where(minimum == True)[0][0])


# TODO: this function is the bottleneck of the algorithm. I think there is
# a lot of space to optimize it (to many explicit loops and checks inside
# the loops).
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
    tiltIncr = const.one
    azimIncr = const.ten
    tiltMax = 32
    azimMax = 360

    tilts = np.arange(const.zero, tiltMax, tiltIncr) * const.degToRad
    azims = np.arange(const.zero, azimMax, azimIncr) * const.degToRad

    # Minimum area seacrh
    sectionInfo = []
    previousVariable = HUGE

    # Function to calculate the value of minimizing function per plane
    def minimizingVariable(contour):
        if min_variable == 'area':
            return geo.ContourPlaneArea(contour)

        elif min_variable == 'perimeter':
            return geo.ContourPerimeter(contour)

        else:
            print('Minimizing variable not recognized!'
                  'Choose area, perimeter.')

    # Iterate over barycenters and clip surface with closed surface
    for index, (center, normal) in enumerate(zip(centers, normals)):

        # Store previous area to compare and find local minimum
        if index > 0:
            previousVariable = minVariable

        # Iterate over rotated planes
        # identifies the minimum area
        minArea = HUGE
        minPlane = None

        minHDiameter = HUGE
        minPerimeter = HUGE
        minVariable = HUGE

        for tilt, azim in zip(tilts, azims):
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

                if geo.ContourIsClosed(contour) and nContourPoints != 0:

                    # Update minmum area
                    variable = minimizingVariable(contour)

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
    areas = sectionInfo[:, 0]

    minimumId = _local_minimum(areas)

    return sectionInfo[minimumId, 1]


def AneurysmNeckPlane(
        vascular_surface: names.polyDataType,
        parent_vascular_surface: names.polyDataType = None,
        min_variable: str = "area",
        aneurysm_type: str = "",
        aneurysm_point: tuple = None
    )   -> names.polyDataType:
    """Search the aneurysm neck plane and clip the aneurysm.

    Procedure based on Piccinelli's pipeline, which is based on the surface
    model with the aneurysm and its parent vasculature reconstruction. The
    single difference is the variable which the algorithm minimizes to search
    for the neck plane: the default is the neck perimeter, whereas in the
    default procedure is the neck section area; this can be controlled by the
    optional argument 'min_variable'.

    It returns the clipped aneurysm surface from the original vasculature.

    Arguments
    ---------
        surface_model -- the original vasculature surface with the aneurysm
        parent_centerlines -- the centerlines of the reconstructed parent
            vasculature
        clipping_points -- points where the vasculature will be clipped.

    Optional args
        min_variable -- the varible by which the neck will be searched (default
            'perimeter'; options 'perimeter' 'area')
    """
    # Variables
    tubeToAneurysmDistance = 'ClippedTubeToAneurysmDistanceArray'

    # Compute vasculature Voronoi
    vascularVoronoi = _compute_Voronoi(vascular_surface)

    # Compute vasculature centerline
    # TODO: update this to use the parent centerlines with the radius array
    if parent_vascular_surface is None:
        # Use the centerline to build the parent tube
        parentCenterlines = cnt.GenerateCenterlines(vascular_surface)

    else:
        parentCenterlines = cnt.GenerateCenterlines(parent_vascular_surface)

    # Reconstruct tube functions and build aneurysm Voronoi
    parentTubeSurface = _tube_surface(parentCenterlines)
    aneurysmVoronoi   = _clip_aneurysm_Voronoi(
                            vascularVoronoi,
                            parentTubeSurface
                        )

    aneurysmEnvelope  = _Voronoi_envelope(aneurysmVoronoi)

    # Extract region of aneurysm influence on the parent artery
    aneurysmPoint = tools.SelectSurfacePoint(vascular_surface) \
                    if aneurysm_point is None \
                    else aneurysm_point

    if aneurysm_type == "bifurcation":
        aneurysmInceptionPortion = _bifurcation_aneurysm_influence_region(
                                       vascular_surface,
                                       aneurysmPoint
                                   )

    elif aneurysm_type == "lateral":
        aneurysmInceptionPortion = _lateral_aneurysm_influence_region(
                                       vascular_surface,
                                       aneurysmPoint
                                   )

    else:
        sys.exit("I do not know the aneurysm type {}".format(aneurysm_type))

    aneurysmalSurface = _clip_initial_aneurysm(
                            vascular_surface,
                            aneurysmEnvelope,
                            parentTubeSurface
                        )

    # Compute distance to aneurysm and tube clipped at diverging points
    aneurysmalSurface = tools.ComputeSurfacesDistance(
                            aneurysmalSurface,
                            aneurysmInceptionPortion,
                            array_name=tubeToAneurysmDistance,
                            signed_array=False
                        )

    # Create sac centerline and search plane along it
    barycenters, normals = _sac_centerline(
                                aneurysmalSurface,
                                tubeToAneurysmDistance
                            )

    neckPlane = _search_neck_plane(
                    aneurysmalSurface,
                    barycenters,
                    normals,
                    min_variable=min_variable
                )

    neckCenter = neckPlane.GetOrigin()
    neckNormal = neckPlane.GetNormal()

    # Remove distance array
    aneurysmalSurface.GetPointData().RemoveArray(tubeToAneurysmDistance)

    # Clip final aneurysm surface: when clipped, two surfaces the aneurysm
    # (desired) and the rest (not desired) which is closest to the clipped tube
    surf1 = tools.ClipWithPlane(
                aneurysmalSurface,
                neckCenter,
                neckNormal
            )

    surf2 = tools.ClipWithPlane(
                aneurysmalSurface,
                neckCenter,
                neckNormal,
                inside_out=True
            )

    # Check which output is farthest from clipped tube (the actual aneurysm
    # surface should be farther)
    tubePoints = _vtk_vertices_to_numpy(aneurysmInceptionPortion)
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
