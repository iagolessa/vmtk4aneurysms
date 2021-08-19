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

from .lib import names
from .lib import centerlines as cl
from .lib import constants as const
from .lib import polydatatools as tools
from .lib import polydatageometry as geo

_dimensions = int(const.three)
_clipInitialAneurysmArrayName = "ClipInitialAneurysmArray"

def _transf_normal(
        normal: tuple,
        tilt: float,
        azim: float
    )   -> tuple:
    """Rotates a normal vector to plane by a tilt and azimuth angles."""

    matrix = np.array([[ np.cos(azim),
                        -np.sin(azim),
                        const.zero],
                       [np.sin(azim)*np.cos(tilt),
                        np.cos(azim)*np.cos(tilt),
                        -np.sin(tilt)],
                       [np.sin(azim)*np.sin(tilt),
                        np.cos(azim)*np.sin(tilt),
                        np.cos(tilt)]])

    return tuple(np.dot(matrix, normal))


def _compute_Voronoi(
        surface_model: names.polyDataType
    )   -> names.polyDataType:
    """Compute Voronoi diagram of a vascular surface."""

    voronoiDiagram = vmtkscripts.vmtkDelaunayVoronoi()
    voronoiDiagram.Surface = surface_model
    voronoiDiagram.CheckNonManifold = True
    voronoiDiagram.Execute()

    return voronoiDiagram.Surface


def _tube_surface(
        centerline: names.polyDataType,
        smooth: bool = True
    )   -> names.polyDataType:
    """Reconstruct tube surface of a given vascular surface.

    The tube surface is the maximum tubular structure inscribed in the
    vasculature.

    Arguments:
        centerline -- the centerline to compute the tube surface with the radius
            array.

    Keyword arguments:
        smooth -- to smooth tube surface (default True)
    """

    radiusArray = 'MaximumInscribedSphereRadius'

    # Get bounds of model
    centerlineBounds  = centerline.GetBounds()
    radiusArrayBounds = centerline.GetPointData().GetArray(radiusArray).GetValueRange()
    maxSphereRadius   = radiusArrayBounds[1]

    # To enlarge the box: could be a fraction of maxSphereRadius
    # tests show that the whole radius is appropriate
    enlargeBoxBounds  = maxSphereRadius

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
    inlets, outlets = cl.ComputeOpenCenters(vascular_surface)

    # Tolerance distance to identify the bifurcation
    divTolerance = 0.01

    # One inlet and two outlets, bifurcation with the aneurysm
    # 1 -> centerline of the branches only
    # 2 -> centerline of the first outlet to the aneurysm and inlet
    # 3 -> centerline of the second outlet to the aneurysm and inlet
    relevantOutlets = outlets[0:2]

    clWithoutAneurysm = cl.GenerateCenterlines(
                            vascular_surface,
                            inlets,
                            relevantOutlets
                        )

    bifClippingPoint = cl.GetDivergingPoint(clWithoutAneurysm, divTolerance)

    lines = []
    for cl_id, outlet in enumerate(relevantOutlets):
        daughterCenterline = cl.GenerateCenterlines(
                                 vascular_surface,
                                 [outlet],
                                 inlets + [aneurysm_point]
                             )
        # Get clipping point on this branch
        dauClippingPoint = cl.GetDivergingPoint(
                               daughterCenterline,
                               divTolerance
                           )

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

    return cl.ComputeTubeSurface(aneurysmInceptionClPortion)

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
    inlets, outlets = cl.ComputeOpenCenters(vascular_surface)

    # Tolerance distance to identify the bifurcation
    divTolerance = 0.01

    # One inlet and one outlet (although the model can have more than one outlet),
    # lateral aneurysm
    # 1 -> "forward" centerline, inlet -> outlet, and aneurysm
    # 2 -> "backward" centerline, outlet -> inlet, with aneurysm
    # Note: the aneurysm is like a bifurcation, in this case

    relevantOutlets = outlets[0:1]

    forwardCenterline = cl.GenerateCenterlines(
                            vascular_surface,
                            inlets,
                            relevantOutlets + [aneurysm_point]
                        )

    backwardCenterline = cl.GenerateCenterlines(
                            vascular_surface,
                            relevantOutlets,
                            inlets + [aneurysm_point]
                        )

    upstreamClipPoint   = cl.GetDivergingPoint(
                              forwardCenterline,
                              divTolerance
                          )

    downstreamClipPoint = cl.GetDivergingPoint(
                              backwardCenterline,
                              divTolerance
                          )

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

    return cl.ComputeTubeSurface(aneurysmInceptionClPortion)


def _clip_aneurysm_Voronoi(
        VoronoiSurface: names.polyDataType,
        tubeSurface: names.polyDataType
    )   -> names.polyDataType:
    """Extract the Voronoi diagram of the aneurysmal portion."""

    # Compute distance between complete Voronoi and the parent vessel tube
    # surface
    DistanceArrayName = 'DistanceToTubeArray'

    VoronoiSurface  = tools.ComputeSurfacesDistance(
                          VoronoiSurface,
                          tubeSurface,
                          array_name=DistanceArrayName
                      )

    aneurysmVoronoi = tools.ClipWithScalar(
                          VoronoiSurface,
                          DistanceArrayName,
                          const.zero
                      )

    aneurysmVoronoi = tools.ExtractConnectedRegion(
                          aneurysmVoronoi,
                          'largest'
                      )

    return tools.Cleaner(aneurysmVoronoi)

def _clip_initial_aneurysm(
        surface_model: names.polyDataType,
        aneurysm_envelope: names.polyDataType,
        parent_tube: names.polyDataType,
        result_clip_array: str = _clipInitialAneurysmArrayName
    )   -> names.polyDataType:
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
    tubeToModelArray     = 'ParentTubeModelDistanceArray'
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
    clippingArray.Surface         = modelSurfaceWithDistance
    clippingArray.Operation       = 'subtract'
    clippingArray.InputArrayName  = envelopeToModelArray
    clippingArray.Input2ArrayName = tubeToModelArray
    clippingArray.ResultArrayName = result_clip_array
    clippingArray.Execute()

    clippedAneurysm = tools.ClipWithScalar(
                          clippingArray.Surface,
                          clippingArray.ResultArrayName,
                          const.zero,
                          inside_out=False
                      )

    aneurysm = tools.ExtractConnectedRegion(
                   clippedAneurysm,
                   'largest'
               )

    # Remove fields
    aneurysm.GetPointData().RemoveArray(tubeToModelArray)
    aneurysm.GetPointData().RemoveArray(envelopeToModelArray)
    aneurysm.GetPointData().RemoveArray(result_clip_array)

    return tools.Cleaner(aneurysm)


def _sac_centerline(
        aneurysm_sac: names.polyDataType,
        distance_array: str
    )   -> tuple:
    """Compute aneurysm sac centerline.

    Compute spline that travels alongs the aneurysm sac from the intersection
    with the pa- rent vessel tube. Its points are defined by the geometric
    place of the barycenters of iso- contours of a distance_array defined on
    the aneurysm surface.

    The function returns the spline vertices in a Numpy nd-array.
    """

    # Get wrapper object of vtk numpy interface
    surfaceWrapper = dsa.WrapDataObject(aneurysm_sac)
    distanceArray  = np.array(
                         surfaceWrapper.PointData.GetArray(distance_array)
                     )

    minTubeDist = float(distanceArray.min())
    maxTubeDist = float(distanceArray.max())

    # Build spline along with to perform the neck search
    nPoints     = int(const.oneHundred)
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

        contour = isoContour.GetOutput()

        contourPoints  = contour.GetPoints()
        nContourPoints = contour.GetNumberOfPoints()
        nContourCells  = contour.GetNumberOfCells()

        if nContourPoints > 0 and nContourCells > 0:
            barycenter = _dimensions*[0]

            vtkvmtk.vtkvmtkBoundaryReferenceSystems.ComputeBoundaryBarycenter(
                contourPoints,
                barycenter
            )

            barycenters.append(barycenter)

    if barycenters:

        # Shift centers to compute interpoint distance
        shiftedBarycenters = [barycenters[0]] + barycenters[0:-1]

        barycenters        = np.array(barycenters)
        shiftedBarycenters = np.array(shiftedBarycenters)

        # Compute distance coordinates
        incrDistances = np.linalg.norm(
                            shiftedBarycenters - barycenters,
                            axis=1
                        )

        distanceCoord = np.cumsum(incrDistances)

        # Find spline of barycenters and get derivative == normals
        limitFraction = const.seven/const.ten

        tck, u = interpolate.splprep(barycenters.T, u=distanceCoord)

        # Max and min t of spline
        # Note that we decrease by two units the start of the spline because I
        # noticed that for some cases the initial point of the spline might be
        # pretty inside the aneurysm, skipping the "neck region"
        minSplineDomain = min(u)
        maxSplineDomain = limitFraction*max(u)

        domain = np.linspace(minSplineDomain, maxSplineDomain, 2*nPoints)

        deriv0 = interpolate.splev(domain, tck, der=0)
        deriv1 = interpolate.splev(domain, tck, der=1)

        # Spline points
        points = np.array(deriv0).T

        # Spline tangents
        tangents = np.array(deriv1).T

        return points, tangents

    else:
        sys.exit("No barycenters found for sac centerline construction.")


def _search_neck_plane(
        aneurysm_sac: names.polyDataType,
        centers: np.ndarray,
        normals: np.ndarray,
        min_variable="area"
    )   -> names.planeType:
    """Search neck plane of aneurysm by minimizing a contour variable.

    This function effectively searches for the aneurysm neck plane: it
    interactively cuts the aneurysm surface with planes defined by the vertices
    and normals to a spline travelling through the aneurysm sac.

    The cut plane is further precessed by a tilt and azimuth angle and the
    minimum search between them, as originally proposed by Piccinelli et al.
    (2009).

    It returns the local minimum solution: the neck plane as a vtkPlane object.
    """

    # For each center on the sac centerline (list), create the rotated and
    # tilted plane normals (list) and compute its area (or min_variable)

    # Rotation angles: from original work
    tiltIncr = const.two
    azimIncr = const.ten
    tiltMax = 32
    azimMax = 360

    tilts = np.arange(const.zero, tiltMax, tiltIncr)*const.degToRad
    azims = np.arange(const.zero, azimMax, azimIncr)*const.degToRad

    globalMinimumAreas = {} # can be used for debug
    previousArea = 0.0

    for center, normal in zip(map(tuple, centers), map(tuple, normals)):

        # More readable option
        planeContours = {(tilt, azim): tools.ContourCutWithPlane(
                                          aneurysm_sac,
                                          center,
                                          _transf_normal(normal, tilt, azim)
                                      )
                         for tilt in tilts for azim in azims}

        # Compute area of the closed contours for each normal direction
        planeSectionAreas = {key: geo.ContourPerimeter(contour) \
                                 if min_variable == "perimeter" \
                                 else geo.ContourPlaneArea(contour)
                             for key, contour in planeContours.items()
                             if contour.GetNumberOfCells() > 0 and \
                                geo.ContourIsClosed(contour)}

        if planeSectionAreas:
            # Get the normal direction of max. area
            minCenter    = center
            minDirection = min(planeSectionAreas, key=planeSectionAreas.get)
            minPlaneArea = min(planeSectionAreas.values())
            minPlaneNormal = _transf_normal(normal, *minDirection)

            # Associate this with each center
            # globalMinimumAreas.update({
            #     center: {
            #         "normal": minPlaneNormal,
            #         "area"  : minPlaneArea
            #     }
            # })

            if minPlaneArea <= previousArea:
                previousArea = minPlaneArea
                continue

            else:
                break

        else:
            continue

    # Create plane
    neckPlane = vtk.vtkPlane()
    neckPlane.SetOrigin(minCenter)
    neckPlane.SetNormal(minPlaneNormal)

    return neckPlane


def AneurysmNeckPlane(
        vascular_surface: names.polyDataType,
        aneurysm_type: str,
        parent_vascular_surface: names.polyDataType = None,
        min_variable: str = "area",
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
        vascular_surface (names.polyDataType)
            the original vasculature surface with the aneurysm

        aneurysm_type (str)
            the aneurysm type, bifurcation or lateral

    Optional
        parent_vascular_surface (names.polyDataType, default: None)
            reconstructed parent vasculature

        min_variable (str, 'area' or 'perimeter', default: 'area')
            the varible by which the neck will be searched

        aneurysm_point (tuple)
            point at the tip of the aneurysm dome, for aneurysm identification.
            If none, the user is prompted to select it.

    Return
        aneurysm_surface (names.polyDataType)
            returns the aneurysm clipped at the neck plane
    """

    # Clean up any arrays on the surface
    vascular_surface = tools.Cleaner(vascular_surface)
    vascular_surface = tools.CleanupArrays(vascular_surface)

    # The authors of the study used the distance to the clipped tube
    # surface to compute the sac centerline. I am currently using
    # the same array used to clip the aneurysmal region
    tubeToAneurysmDistance = "ClippedTubeToAneurysmDistanceArray"


    # 1) Compute vasculature's Voronoi
    vascularVoronoi = cl.ComputeVoronoiDiagram(vascular_surface)


    # 2) Compute parent vasculature centerline tube
    if parent_vascular_surface is None:
        parentCenterlines = cl.GenerateCenterlines(vascular_surface)

    else:
        parentCenterlines = cl.GenerateCenterlines(parent_vascular_surface)

    parentTubeSurface = cl.ComputeTubeSurface(parentCenterlines)


    # 3) Aneurysm Voronoi isolation
    aneurysmVoronoi   = _clip_aneurysm_Voronoi(
                            vascularVoronoi,
                            parentTubeSurface
                        )

    aneurysmEnvelope  = cl.ComputeVoronoiEnvelope(aneurysmVoronoi)


    # 4) Extraction aneurysmal-inception region
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
        sys.exit(
            "I do not know the aneurysm type {}".format(aneurysm_type)
        )


    # 5) Aneurysmal surface isolation
    aneurysmalSurface = _clip_initial_aneurysm(
                            vascular_surface,
                            aneurysmEnvelope,
                            parentTubeSurface
                        )

    aneurysmalSurface = tools.ComputeSurfacesDistance(
                            aneurysmalSurface,
                            aneurysmInceptionPortion,
                            array_name=tubeToAneurysmDistance,
                            signed_array=False
                        )


    # 6) Create sac centerline
    barycenters, normals = _sac_centerline(
                                aneurysmalSurface,
                                tubeToAneurysmDistance
                            )

    aneurysmalSurface.GetPointData().RemoveArray(
        tubeToAneurysmDistance
    )


    # 7) Search neck plane
    neckPlane = _search_neck_plane(
                    aneurysmalSurface,
                    barycenters,
                    normals,
                    min_variable=min_variable
                )

    # 8) Detach aneurysm sac from parent vasculature
    neckCenter = neckPlane.GetOrigin()
    neckNormal = neckPlane.GetNormal()

    # Clip final aneurysm surface: the side to where the normal point
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
    tubePoints  = dsa.WrapDataObject(aneurysmInceptionPortion).GetPoints()
    surf1Points = dsa.WrapDataObject(surf1).GetPoints()
    surf2Points = dsa.WrapDataObject(surf2).GetPoints()

    tubeCentroid  = tubePoints.mean(axis=0)
    surf1Centroid = surf1Points.mean(axis=0)
    surf2Centroid = surf2Points.mean(axis=0)

    surf1Distance = vtk.vtkMath.Distance2BetweenPoints(
                        tubeCentroid,
                        surf1Centroid
                    )

    surf2Distance = vtk.vtkMath.Distance2BetweenPoints(
                        tubeCentroid,
                        surf2Centroid
                    )

    return surf1 if surf1Distance > surf2Distance else surf2
