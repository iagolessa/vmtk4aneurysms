# Copyright (C) 2022, Iago L. de Oliveira

# vmtk4aneurysms is free software: you can redistribute it and/or modify
# it under the terms of the GNU General Public License as published by
# the Free Software Foundation, either version 3 of the License, or
# (at your option) any later version.

# This program is distributed in the hope that it will be useful,
# but WITHOUT ANY WARRANTY; without even the implied warranty of
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
# GNU General Public License for more details.

# You should have received a copy of the GNU General Public License
# along with this program.  If not, see <http://www.gnu.org/licenses/>.

"""Collection of tools to manipulate vascular models and aneurysms.

The two main functions provided by the module compute the aneurysm neck plane,
as defined by Piccinelli et al. (2009), and extracts the hypothetically healthy
vasculature.
"""

import sys
import vtk
import numpy as np
import morphman.common as mplib
from morphman.manipulate_curvature import extract_single_line

from vmtk import vtkvmtk
from vmtk import vmtkscripts
from scipy import interpolate

from vtk.numpy_interface import dataset_adapter as dsa

from .lib import names
from .lib import centerlines as cl
from .lib import constants as const
from .lib import polydatatools as tools
from .lib import polydatageometry as geo

from . import wallmotion as wm

_dimensions = int(const.three)
_initialAneurysmArrayName = names.AneurysmalRegionArrayName

def _bifurcation_aneurysm_clipping_points(
        vascular_surface: names.polyDataType,
        aneurysm_point: tuple
    )   -> dict:
    """Extract vessel portion where a bifurcation aneurysm grew.

    Given the vascular model surface with open inlet and outlet profiles,
    extract the portion of the tube surface where the aneurysm grew by
    calculating the divegence points of the centerlines. The user must select a
    point on the aneurysm's dome surface.

    Note that the algorithm will use the two first outlets to compute the
    centerlines, so avoid any outlet profile between the inlet and the
    aneurysm.
    """
    clippingPoints  = {}
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

    clippingPoints["bif"] = cl.GetDivergingPoint(
                                clWithoutAneurysm,
                                divTolerance
                            )

    for cl_id, outlet in enumerate(relevantOutlets):
        daughterCenterline = cl.GenerateCenterlines(
                                 vascular_surface,
                                 [outlet],
                                 inlets + [aneurysm_point]
                             )
        # Get clipping point on this branch
        clippingPoints["dau" + str(cl_id)] = cl.GetDivergingPoint(
                                                 daughterCenterline,
                                                 divTolerance
                                             )


    return clippingPoints

def _lateral_aneurysm_clipping_points(
        vascular_surface: names.polyDataType,
        aneurysm_point: tuple
    )   -> dict:
    """Extract vessel portion where a lateral aneurysm grew.

    Given the vascular model surface with open inlet and outlet profiles,
    extract the portion of the tube surface where the aneurysm grew by
    calculating the divegence points of the centerlines. The user must select a
    point on the aneurysm's dome surface.

    Note that the algorithm will use the first outlet to compute the
    centerlines, so avoid any outlet profile between the inlet and the aneurysm
    region.
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

    return {"upstream": upstreamClipPoint, "downstream": downstreamClipPoint}

def _set_portion_in_cl_patch(
        surface: names.polyDataType,
        patch_centerline: names.polyDataType,
        patch_id: int,
        filter_array_name: str
    ):
    """Marks a surface's portions that are within a centerline patch.

    Given a surface with an array of zeros defined on it, change its
    values to one where the surface lies within the bound of the
    polyball function defined on a centerlines patch of the same surface.
    The radius array must also be defined on the surface.
    """

    # Convert surface and create array
    nPoints   = surface.GetNumberOfPoints()
    npSurface = dsa.WrapDataObject(surface)
    pointData = npSurface.GetPointData()

    if filter_array_name not in tools.GetPointArrays(surface):
        pointData.append(
            dsa.VTKArray(np.zeros(nPoints, dtype=int)),
            filter_array_name
        )

    inPatchArray = pointData.GetArray(filter_array_name)

    # Compute cylinder params of centerline patch
    tangent, center, radius = cl.ComputeClPatchEndPointParameters(
                                  patch_centerline,
                                  patch_id
                              )

    # Extract patch
    patch = extract_single_line(patch_centerline, patch_id)

    tubeFunction = vtkvmtk.vtkvmtkPolyBallLine()
    tubeFunction.SetInput(patch)
    tubeFunction.SetPolyBallRadiusArrayName(cl._radiusArrayName)

    lastSphere = vtk.vtkSphere()
    lastSphere.SetRadius(radius*1.5)
    lastSphere.SetCenter(center)

    for index, point in enumerate(npSurface.GetPoints()):
        voronoiVector    = point - center
        voronoiVectorDot = vtk.vtkMath.Dot(voronoiVector, tangent)

        tubevalue   = tubeFunction.EvaluateFunction(point)
        spherevalue = lastSphere.EvaluateFunction(point)

        # If outside the patch region
        if spherevalue < 0.0 and voronoiVectorDot < 0.0:
            continue

        # If inside the patch region and inside the tube
        elif tubevalue <= 0.0:
            inPatchArray[index] = 1

    return npSurface.VTKObject

def _is_bifurcation_aneurysm(
        aneurysm_type: str
    )   -> bool:
    """Return True if aneurysm is bifurcation, else return False."""

    if   aneurysm_type == "bifurcation":
        return True

    elif aneurysm_type == "lateral":
        return False

    else:
        raise NameError(
                "Aneurysm type either 'bifurcation' or 'lateral'."\
                " {} passed".format(aneurysm_type)
              )

def HealthyVesselReconstruction(
        vascular_surface: names.polyDataType,
        aneurysm_type: str,
        dome_point: tuple=None
    )   -> names.polyDataType:
    """Given vasculature model with aneurysm, extract vessel without aneurysm.

    Based on the procedure proposed by

    Ford et al. An objective approach to digital removal of saccular aneurysms:
    technique and applications. The British Journal of Radiology.
    2009;82:S55–61

    and implemented in VMTK by Ms. Piccinelli, this function extracts the
    'hypothetical healthy' vessel of a vascular model with an intracranial
    aneurysm.

    .. warning::
        It is important to "reduce" the vascular surface to only the region
        where the aneurysm is, ie clip the surface so only the parent vessel
        and the daughter branches are left.

    .. warning::
        Try to select, or pass, a dome point that lies on the farthest location
        form the neck and that is centered to the neck.

    Arguments
    vascular_surface (vtkPolyData) -- the vascular surface model clipped at the
    inlet and two outlets (if a bifurcation aneurysm) or one outlet (if a
    lateral aneurysm)

    aneurysm_type (str) -- the type of aneurysm ("bifurcation" or "lateral")

    Optional
    dome_point (tuple) -- a point on the aneurysm dome surface. It is used
    to identify the aneurysm, and must be located at the tip of the aneurysm
    dome, preferentially. If None is passed, the user is prompted to select one
    interactively (default None).

    Returns
    healthy_surface (vtkPolyData) -- the surface model without the aneurysm.
    """
    aneurysmInBifurcation = _is_bifurcation_aneurysm(aneurysm_type)

    voronoi       = cl.ComputeVoronoiDiagram(vascular_surface)
    centerlines   = cl.GenerateCenterlines(vascular_surface)

    # Smooth the Voronoi diagram
    smoothedVoronoi = mplib.voronoi_operations.smooth_voronoi_diagram(
                          voronoi,
                          centerlines,
                          0.25 # Smoothing factor, recommended by Ms. Piccinelli
                      )

    # 1) Compute parent centerline reconstruction
    dome_point = tools.SelectSurfacePoint(vascular_surface) \
                 if dome_point is None \
                 else dome_point

    # Get clipping points on model centerlines and order them correctly
    if aneurysmInBifurcation:
        dictClipPoints = _bifurcation_aneurysm_clipping_points(
                             vascular_surface,
                             dome_point
                         )

        orderedClipPoints = [dictClipPoints.get("bif",  None),
                             dictClipPoints.get("dau0", None),
                             dictClipPoints.get("dau1", None)]

    else:
        dictClipPoints = _lateral_aneurysm_clipping_points(
                             vascular_surface,
                             dome_point
                         )

        orderedClipPoints = [dictClipPoints.get("upstream",   None),
                             dictClipPoints.get("downstream", None)]

    # Store as VTk points
    clippingPoints = vtk.vtkPoints()

    for point in orderedClipPoints:
        clippingPoints.InsertNextPoint(point)

    # Extract patch centerlines
    isSiphon = not aneurysmInBifurcation

    patchCenterlines = mplib.vessel_reconstruction_tools.create_parent_artery_patches(
                            centerlines,
                            clippingPoints,
                            siphon=isSiphon,
                            bif=aneurysmInBifurcation
                        )


    # 2) Interpolate patch centerlines using splines
    parentCenterlines = mplib.vessel_reconstruction_tools.interpolate_patch_centerlines(
                            patchCenterlines,
                            centerlines,
                            additionalPoint=None,
                            lower='bif', # ... Investigate this param
                            version=True
                        )


    # 3) Clip Voronoi Diagram along centerline patches
    filterArrayName =  "InPatchArray"

    for cl_id in range(patchCenterlines.GetNumberOfCells()):

        # Mark points on the Voronoi that are only on the patched centerlines
        markedVoronoi = _set_portion_in_cl_patch(
                            smoothedVoronoi,
                            patchCenterlines,
                            cl_id,
                            filterArrayName
                        )

    # Apply filter and get portion with values == 1
    clippedVoronoi = tools.ExtractPortion(
                         markedVoronoi,
                         filterArrayName,
                         int(const.one)
                     )

    # As required by Morphman, also pass the clipping points as a Numpy
    # array
    clipPointsArray = np.array([clippingPoints.GetPoint(i)
                                for i in range(clippingPoints.GetNumberOfPoints())])

    # 4) Interpolate Voronoi diagram along interpolated centerline
    newVoronoi = mplib.vessel_reconstruction_tools.interpolate_voronoi_diagram(
                    parentCenterlines,
                    patchCenterlines,
                    clippedVoronoi,
                    [clippingPoints, clipPointsArray],
                    bif=[],
                    cylinder_factor=1.0
                )

    # 5) Compute parent surface from new Voronoi
    parentSurface = cl.ComputeVoronoiEnvelope(newVoronoi)

    # Clip the parent vascular surface
    # Does not work well with the vascular cases
    # parentSurface = vscop.ClipVasculature(parentSurface)

    clipper = vmtkscripts.vmtkSurfaceClipper()
    clipper.Surface = parentSurface
    clipper.InsideOut = False
    clipper.Execute()

    return clipper.Surface


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

    # Get the clipping points
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
        line = extract_single_line(clWithoutAneurysm, cl_id)

        loc = mplib.vtk_wrapper.get_vtk_point_locator(line)

        #Find closest points to clipping on parent centerline
        dauId = loc.FindClosestPoint(dauClippingPoint)
        bifId = loc.FindClosestPoint(bifClippingPoint)

        lines.append(
            extract_single_line(
                line,
                0,
                start_id=bifId,
                end_id=dauId
            )
        )

    aneurysmInceptionClPortion = mplib.vtk_wrapper.vtk_merge_polydata(lines)

    return cl.ComputeTubeSurface(aneurysmInceptionClPortion)

def _lateral_aneurysm_influence_region(
        vascular_surface: names.polyDataType,
        aneurysm_point: tuple
    )   -> names.polyDataType:
    """Extract vessel portion where a lateral aneurysm grew.

    Given the vascular model surface with open inlet and outlet profiles,
    extract the portion of the tube surface where the aneurysm grew by
    calculating the divegence points of the centerlines. The user must select a
    point on the aneurysm's dome surface.

    Note that the algorithm will use the first outlet to compute the
    centerlines, so avoid any outlet profile between the inlet and the aneurysm
    region.
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
    line = extract_single_line(forwardCenterline, 0)
    loc  = mplib.vtk_wrapper.get_vtk_point_locator(line)

    #Find closest points to clipping on parent centerline
    upstreamId   = loc.FindClosestPoint(upstreamClipPoint)
    downstreamId = loc.FindClosestPoint(downstreamClipPoint)

    aneurysmInceptionClPortion =  extract_single_line(
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

def _mark_aneurysmal_region(
        surface_model: names.polyDataType,
        aneurysm_envelope: names.polyDataType,
        parent_tube: names.polyDataType,
        result_clip_array: str = _initialAneurysmArrayName
    )   -> names.polyDataType:
    """Compute the aneurysmal region surface from the original vascular model.

    Compute distance between the aneurysm envelope and parent vasculature tube
    function from the original vascular surface model. Mark the surface with a
    field such that its zero value is the difference between those two fields.
    This represents an approximation of the aneurysm neck contour.

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
    markedAneurysmalRegion = vmtkscripts.vmtkSurfaceArrayOperation()
    markedAneurysmalRegion.Surface         = modelSurfaceWithDistance
    markedAneurysmalRegion.Operation       = 'subtract'
    markedAneurysmalRegion.InputArrayName  = tubeToModelArray
    markedAneurysmalRegion.Input2ArrayName = envelopeToModelArray
    markedAneurysmalRegion.ResultArrayName = result_clip_array
    markedAneurysmalRegion.Execute()

    aneurysmalSurface = markedAneurysmalRegion.Surface

    # Remove unnecessary fields
    aneurysmalSurface.GetPointData().RemoveArray(tubeToModelArray)
    aneurysmalSurface.GetPointData().RemoveArray(envelopeToModelArray)

    return aneurysmalSurface

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

def _extract_aneurysm_inception_region(
        vascular_surface: names.polyDataType,
        aneurysm_type: str,
        aneurysm_point: tuple=None
    )   -> names.polyDataType:

    aneurysmPoint = tools.SelectSurfacePoint(vascular_surface) \
                    if aneurysm_point is None \
                    else aneurysm_point

    aneurysmInBifurcation = _is_bifurcation_aneurysm(aneurysm_type)

    if aneurysmInBifurcation:
        aneurysmInceptionPortion = _bifurcation_aneurysm_influence_region(
                                       vascular_surface,
                                       aneurysmPoint
                                   )

    else:
        aneurysmInceptionPortion = _lateral_aneurysm_influence_region(
                                       vascular_surface,
                                       aneurysmPoint
                                   )

    return aneurysmInceptionPortion

def _extract_aneurysmal_region(
        vascular_surface: names.polyDataType,
        parent_vascular_surface: names.polyDataType=None,
        parent_vascular_centerline: names.polyDataType=None,
        aneurysmal_region_array: str=_initialAneurysmArrayName,
        aneurysm_type: str=""
    )   -> names.polyDataType:
    """Marks the aneurysmal region with an array and extract the vessel portion
    where the aneurysm grew.

    Based on the five first steps of Piccinelli's procedure, this function
    marks the vascular model passed with an array whose zero value marks the
    contour of the aneurysmal region. It also returns the surface portion of
    the vascular model where the aneurysm grew.
    """

    # 1) Compute vasculature's Voronoi
    vascularVoronoi = cl.ComputeVoronoiDiagram(vascular_surface)

    # 2) Compute parent vasculature centerline tube
    if parent_vascular_centerline is None and \
       parent_vascular_surface is None:

        # Compute healthy vasculature
        parent_vascular_surface = HealthyVesselReconstruction(
                                      vascular_surface,
                                      aneurysm_type
                                  )

        parentCenterlines = cl.GenerateCenterlines(parent_vascular_surface)

    elif parent_vascular_centerline is None and \
         parent_vascular_surface is not None:

        # Compute the centerline of the parent vascular surface
        parentCenterlines = cl.GenerateCenterlines(parent_vascular_surface)

    elif parent_vascular_centerline is not None:
        # The best alternative actually
        # By adding this option, the centerline generated by
        # HealthyVesselReconstruction could be directly used
        parentCenterlines = parent_vascular_centerline

    parentTubeSurface = cl.ComputeTubeSurface(parentCenterlines)

    # 3) Aneurysm Voronoi isolation
    aneurysmVoronoi   = _clip_aneurysm_Voronoi(
                            vascularVoronoi,
                            parentTubeSurface
                        )

    aneurysmEnvelope  = cl.ComputeVoronoiEnvelope(aneurysmVoronoi)

    # 4) Aneurysmal surface isolation
    aneurysmalSurface = _mark_aneurysmal_region(
                            vascular_surface,
                            aneurysmEnvelope,
                            parentTubeSurface,
                            result_clip_array=aneurysmal_region_array
                        )

    return aneurysmalSurface

def MarkAneurysmSacManually(
        surface: names.polyDataType,
        aneurysm_neck_array_name: str=names.DistanceToNeckArrayName
    )   -> names.polyDataType:
    """Manually select the aneurysm neck contour and compute the distance to
    it.

    Given a vasculature with an aneurysm, prompt the user to manually draw the
    aneurysm neck on the surface. An scalar array (field) is then defined on
    the surface with value 0 on the aneurysm neck contour defined and its other
    values as the geodesic distance to the neck contour.
    """

    # Get ids of contour
    getContour = tools.SelectContourPointsIds()
    getContour.Surface = surface
    getContour.Execute()

    # Compute the geodesic distance  from the approximate neck contour
    surface = geo.SurfaceGeodesicDistanceToContour(
                  surface,
                  getContour.ContourIds,
                  gdistance_array_name=aneurysm_neck_array_name
              )

    return surface

def ClipVasculature(
        vascular_surface: names.polyDataType
    )   -> names.polyDataType:
    """Clip a vascular surface segment, by selecting end points.

    Given a vascular surface, the user is prompted to select points
    on the surface that 1) identifies the surface's bulk and 2) where the
    vasculature should be clipped. Uses, internally, the
    'vmtksurfaceendclipper' script.
    """

    centerlines = cl.GenerateCenterlines(vascular_surface)
    geoCenterlines = cl.ComputeCenterlineGeometry(centerlines)

    FrenetTangentArrayName = "FrenetTangent"

    surfaceEndClipper = vmtkscripts.vmtkSurfaceEndClipper()
    surfaceEndClipper.Surface = vascular_surface
    surfaceEndClipper.CenterlineNormals = 1
    surfaceEndClipper.Centerlines = geoCenterlines
    surfaceEndClipper.FrenetTangentArrayName = FrenetTangentArrayName
    surfaceEndClipper.Execute()

    return surfaceEndClipper.Surface

def MarkAneurysmalRegion(
        vascular_surface: names.polyDataType,
        parent_vascular_surface: names.polyDataType=None,
        parent_vascular_centerline: names.polyDataType=None,
        gdistance_to_neck_array_name: str=names.DistanceToNeckArrayName,
        aneurysm_point: tuple=None
    )   -> names.polyDataType:
    """Marks the aneurysmal region with an array of (geodesic) distances to the
    neck contour.

    Based on the five first steps of Piccinelli's procedure, this function
    marks the vascular model passed with an array whose zero value marks the
    contour of the aneurysmal region. This may be used for an initial
    approximation of the aneurysm surface itself. The rest of the array is
    given by the geodesic distance of the point to the neck contour.

    You may optionally pass a point located at the dome point of the aneurysm
    so the algorithm more easily identifies the aneurysm region.

    .. warning::
        Negative distance values are used inside the aneurysm neck contour
        (i.e., it marks the aneurysm sac) and positive values elsewhere.

    .. warning::
        Better results are expected if you "reduce" the vascular surface to
        only the region where the aneurysm is, ie clip the surface so only the
        parent vessel and the daughter branches are left.

    .. warning::
        This function destroys any arrays in the passing surface.

    Arguments
    ---------
    vascular_surface (names.polyDataType) -- the original vasculature surface
    with the aneurysm

    Optional
    parent_vascular_surface (names.polyDataType, default: None) --
    reconstructed parent vasculature

    parent_vascular_centerline (names.polyDataType, default: None) -- instead
    of the parent (hypothetically healthy) vascular surface, its centerline can
    be passed

    gdistance_to_neck_array_name (str) -- name of the array defined on the
    surface to mark the aneurysm

    aneurysm_point (tuple) -- point at the tip of the aneurysm dome, for
    aneurysm identification.

    Return
    surface (vtkPolyData) -- vascular surface with an array defined on it
    marking the aneurysmal region contour.
    """

    # Clean up any arrays on the surface
    vascular_surface = tools.Cleaner(vascular_surface)

    # Copy the original surface and store it so the final array is interpolated
    # back to it
    copiedSurface = tools.CopyVtkObject(vascular_surface)

    # Perform the procedure on a clean surface
    vascular_surface = tools.CleanupArrays(vascular_surface)

    # Perform first five steps of Piccinelli's procedure, returning the
    # vascular surface marked with the aneurysmal region via an array and the
    # vascular rerion where the aneurysm has grown
    vascular_surface = _extract_aneurysmal_region(
                           vascular_surface,
                           parent_vascular_surface=parent_vascular_surface,
                           parent_vascular_centerline=parent_vascular_centerline,
                           aneurysmal_region_array=names.AneurysmalRegionArrayName
                       )

    # Add a little bit of smoothing to the Distance field to remove corner
    # discontnuities
    vascular_surface = tools.SmoothSurfacePointField(
                           vascular_surface,
                           names.AneurysmalRegionArrayName,
                           niterations=5
                       )

    # The best approach I found to extract the closest path with the surface
    # model points was through the clip: the clip used subsequentely wtih the
    # boundary extractor provides a set of points that are ORIENTED along the
    # neck line. On the other hand, The initial tests I did were with the
    # contour filter, which, as far as I could assess, generates a polyline
    # that does not have its points oriented along the path, which inhibited
    # the use of the selection filter to get the aneurysmal region and change
    # the sign of the geodeseic distance to neck array (note, the coumputation
    # of the geodesic distance per se did not require the points to be
    # oriented).
    aneurysmalSurface = tools.ClipWithScalar(
                            vascular_surface,
                            names.AneurysmalRegionArrayName,
                            const.zero
                        )

    # Get the largest region or closest to aneurysm dome point
    if aneurysm_point:
        aneurysmalSurface = tools.ExtractConnectedRegion(
                                aneurysmalSurface,
                                "closest",
                                closest_point=aneurysm_point
                            )

    else:
        aneurysmalSurface = tools.ExtractConnectedRegion(
                                aneurysmalSurface,
                                "largest"
                            )

    # Extract the bounday of the cutted cells
    # This provides a rough approximation of where the neck contour
    # cuts the surface
    boundaryExtractor = vtkvmtk.vtkvmtkPolyDataBoundaryExtractor()
    boundaryExtractor.SetInputData(aneurysmalSurface)
    boundaryExtractor.Update()

    neckContour = boundaryExtractor.GetOutput()

    # Locator to find closest points
    locator = vtk.vtkPointLocator()
    locator.SetDataSet(vascular_surface)
    locator.BuildLocator()
    locator.Update()

    # Get points on the surface that are closest to the neck points
    allClosestPointsIds = [locator.FindClosestPoint(
                               neckContour.GetPoint(pointId)
                           )
                           for pointId in range(neckContour.GetNumberOfPoints())]

    # Remove duplicates while keeping its order
    closestPointsIds = sorted(
                           set(allClosestPointsIds),
                           key=lambda x: allClosestPointsIds.index(x)
                       )

    # Build ID list of points on the surface
    pointIds = vtk.vtkIdList()

    for pointId in closestPointsIds:
        pointIds.InsertNextId(pointId)

    # Compute the geodesic distance  from the approximate neck contour
    vascular_surface = geo.SurfaceGeodesicDistanceToContour(
                           vascular_surface,
                           pointIds,
                           gdistance_array_name=gdistance_to_neck_array_name
                       )

    # Interpolate the distance array to the original surface
    vascular_surface = tools.ProjectPointArray(
                           copiedSurface,
                           vascular_surface,
                           gdistance_to_neck_array_name
                       )

    return vascular_surface

def ComputeAneurysmNeckPlane(
        vascular_surface: names.polyDataType,
        aneurysm_type: str,
        parent_vascular_surface: names.polyDataType=None,
        parent_vascular_centerline: names.polyDataType=None,
        min_variable: str="area",
        aneurysm_point: tuple=None
    )   -> tuple:
    """Search the aneurysm neck plane.

    Procedure based on Piccinelli's pipeline, which is based on the surface
    model with the aneurysm and its parent vasculature reconstruction. The
    single difference is the variable which the algorithm minimizes to search
    for the neck plane: the default is the neck perimeter, whereas in the
    default procedure is the neck section area; this can be controlled by the
    optional argument 'min_variable'.

    It returns the information necessary to construct the neck plane: its
    center and its normal, based on the coordinate system of the input vascular
    surface model.

    .. warning::
        The parent vessel surface can be computed with the
        HealthyVesselReconstruction function also provided with this module. If
        the parent vessel is not provided, this function uses the vascular
        surface itself to perform the procedure (tube function reconstruction),
        which may impair the results.

    .. warning::
        Better results are expected if you "reduce" the vascular surface to
        only the region where the aneurysm is, ie clip the surface so only the
        parent vessel and the daughter branches are left.

    .. warning::
        Try to select, or pass, a dome point that lies on the farthest location
        form the neck and that is centered to the neck.

    Arguments
    ---------
    vascular_surface (names.polyDataType) -- the original vasculature surface
    with the aneurysm

    aneurysm_type (str) -- the aneurysm type, bifurcation or lateral

    Optional
    parent_vascular_surface (names.polyDataType, default: None) --
    reconstructed parent vasculature

    parent_vascular_centerline (names.polyDataType, default: None) -- instead
    of the parent (hypothetically healthy) vascular surface, its centerline can
    be passed

    min_variable (str, 'area' or 'perimeter', default: 'area') -- the varible
    by which the neck will be searched

    aneurysm_point (tuple) -- point at the tip of the aneurysm dome, for
    aneurysm identification.  If none, the user is prompted to select it.

    Return
    plane center and normal (tuple) -- returns the center of the neck plane and
    its normal vector
    """

    # Clean up any arrays on the surface
    vascular_surface = tools.Cleaner(vascular_surface)
    vascular_surface = tools.CleanupArrays(vascular_surface)

    # Perform first five steps of Piccinelli's procedure, returning the
    # vascular surface marked with the aneurysmal region via an array and the
    # vascular rerion where the aneurysm has grown
    aneurysmalSurface = _extract_aneurysmal_region(
                            vascular_surface,
                            parent_vascular_surface=parent_vascular_surface,
                            parent_vascular_centerline=parent_vascular_centerline,
                            aneurysmal_region_array=_initialAneurysmArrayName,
                            aneurysm_type=aneurysm_type
                        )

    # Clip aneurysmal portion (scalars < 0)
    clippedAneurysmalSurface = tools.ClipWithScalar(
                                   aneurysmalSurface,
                                   _initialAneurysmArrayName,
                                   const.zero
                               )

    clippedAneurysmalSurface = tools.ExtractConnectedRegion(
                                   clippedAneurysmalSurface,
                                   'largest'
                               )

    clippedAneurysmalSurface.GetPointData().RemoveArray(
        _initialAneurysmArrayName
    )

    # Get the portion where the aneurysm grew
    aneurysmInceptionPortion = _extract_aneurysm_inception_region(
                                   vascular_surface,
                                   aneurysm_type,
                                   aneurysm_point=aneurysm_point
                               )

    # The authors of the study used the distance to the clipped tube
    # surface to compute the sac centerline. I am currently using
    # the same array used to clip the aneurysmal region
    tubeToAneurysmDistance = "ClippedTubeToAneurysmDistanceArray"

    aneurysmalSurface = tools.ComputeSurfacesDistance(
                            tools.Cleaner(clippedAneurysmalSurface),
                            aneurysmInceptionPortion,
                            array_name=tubeToAneurysmDistance,
                            signed_array=False
                        )


    # Create sac centerline
    barycenters, normals = _sac_centerline(
                                aneurysmalSurface,
                                tubeToAneurysmDistance
                            )

    aneurysmalSurface.GetPointData().RemoveArray(
        tubeToAneurysmDistance
    )


    # Search neck plane
    neckPlane = _search_neck_plane(
                    aneurysmalSurface,
                    barycenters,
                    normals,
                    min_variable=min_variable
                )

    # 8) Detach aneurysm sac from parent vasculature
    neckCenter = neckPlane.GetOrigin()
    neckNormal = neckPlane.GetNormal()

    # # Clip final aneurysm surface: the side to where the normal point
    # surf1 = tools.ClipWithPlane(
    #             aneurysmalSurface,
    #             neckCenter,
    #             neckNormal
    #         )

    # surf2 = tools.ClipWithPlane(
    #             aneurysmalSurface,
    #             neckCenter,
    #             neckNormal,
    #             inside_out=True
    #         )

    # # Check which output is farthest from clipped tube (the actual aneurysm
    # # surface should be farther)
    # tubePoints  = dsa.WrapDataObject(aneurysmInceptionPortion).GetPoints()
    # surf1Points = dsa.WrapDataObject(surf1).GetPoints()
    # surf2Points = dsa.WrapDataObject(surf2).GetPoints()

    # tubeCentroid  = tubePoints.mean(axis=0)
    # surf1Centroid = surf1Points.mean(axis=0)
    # surf2Centroid = surf2Points.mean(axis=0)

    # surf1Distance = vtk.vtkMath.Distance2BetweenPoints(
    #                     tubeCentroid,
    #                     surf1Centroid
    #                 )

    # surf2Distance = vtk.vtkMath.Distance2BetweenPoints(
    #                     tubeCentroid,
    #                     surf2Centroid
    #                 )

    # aneurysmPlaneNeckSurface = surf1 if surf1Distance > surf2Distance else surf2

    return neckCenter, neckNormal

def ExtractAneurysmSacSurface(
        vascular_surface: names.polyDataType,
        mode: str="interactive",
        parent_vascular_surface: names.polyDataType=None,
        parent_vascular_centerline: names.polyDataType=None,
        aneurysm_type: str=""
    )   -> names.polyDataType:
    """Clip the aneurysm sac surface from the vascular surface model.

    Given the vascular model with an aneurysm, clip the aneurysm sac surface
    with one of three methods:

        - Manually, by interactively placing seeds that form the aneurysm neck
          contour;

        - Automatically, by using Piccinelli's procedure to clip the aneurysm
          neck. This option extracts the aneurysmal portion;

        - Automatically but using the neck plane procedure by Piccinelli et al.

    Arguments
    ---------
    vascular_surface (names.polyDataType) -- the original vasculature surface
    with the aneurysm

    Optional
    --------
    mode (str, default: 'interactive') -- the method to clip the aneurysm:
    'interactive', 'automatic', or 'plane'

    parent_vascular_surface (names.polyDataType, default: None) --
    reconstructed parent vasculature

    parent_vascular_centerline (names.polyDataType, default: None) -- instead
    of the parent (hypothetically healthy) vascular surface, its centerline can
    be passed

    aneurysm_type (str, default: "", ["lateral", "bifurcation"]): mandatory if
    'mode' is 'automatic' and 'parent_vascular_surface' is not passed, because
    it is used in its computation

    Return
    aneurysm sac surface (names.polyDataType) -- the surface of the aneurysm
    sac clipped from the vascular surface model
    """

    # Clean up any arrays on the surface
    vascular_surface = tools.Cleaner(vascular_surface)
    # vascular_surface = tools.CleanupArrays(vascular_surface)

    # Based on the available methods, mark the surface with the neck array
    markedSurface = ComputeDistanceToAneurysmNeck(
                        vascular_surface,
                        mode=mode,
                        parent_vascular_surface=parent_vascular_surface,
                        aneurysm_type=aneurysm_type
                    )

    # Add a little bit of smoothing on the neck distance field
    markedSurface = tools.SmoothSurfacePointField(
                        markedSurface,
                        names.DistanceToNeckArrayName,
                        niterations=10
                    )

    # Clip the aneurysm sac (aneurysm marked with negative values)
    clippedAneurysmSurface = tools.ClipWithScalar(
                                   markedSurface,
                                   names.DistanceToNeckArrayName,
                                   const.zero
                               )

    clippedAneurysmSurface.GetPointData().RemoveArray(
        names.DistanceToNeckArrayName
    )

    return clippedAneurysmSurface

def _compute_local_wlr(diameter):
    if diameter > const.VesselLargeDiameter:
        return const.WlrLarge

    elif diameter < const.VesselMediumDiameter:
        return const.WlrMedium

    else:
        # Linear threshold
        deltaWlr = const.WlrLarge - const.WlrMedium
        deltaDiameter = const.VesselLargeDiameter - const.VesselMediumDiameter
        angCoeff = deltaWlr/deltaDiameter

        return const.WlrMedium + angCoeff*(diameter - const.VesselMediumDiameter)

def ComputeVasculatureThickness(
        vascular_surface: names.polyDataType,
        centerlines: names.polyDataType=None,
        thickness_field_name: str=names.ThicknessArrayName,
        set_uniform_wlr: bool=False,
        uniform_wlr_value: float=const.WlrMedium
    )   -> names.polyDataType:
    """Compute thickness of a vasculature based on its diameter and WLR.

    Given input surface with the radius array, computes the thickness by
    multiplying by the wall-to-lumen ration. The aneurysm portion is also
    multiplyed.
    """

    # Compute centerlines
    if not centerlines:
        centerlines = cl.GenerateCenterlines(vascular_surface)

    # Compute distance to centerlines
    # It will hold the thickness field at the end
    distanceToCenterlines = vtkvmtk.vtkvmtkPolyDataDistanceToCenterlines()
    distanceToCenterlines.SetInputData(vascular_surface)
    distanceToCenterlines.SetCenterlines(centerlines)

    distanceToCenterlines.SetUseRadiusInformation(True)
    distanceToCenterlines.SetEvaluateCenterlineRadius(True)
    distanceToCenterlines.SetEvaluateTubeFunction(False)
    distanceToCenterlines.SetProjectPointArrays(False)

    distanceToCenterlines.SetDistanceToCenterlinesArrayName(
        thickness_field_name
    )

    distanceToCenterlines.SetCenterlineRadiusArrayName(names.VascularRadiusArrayName)
    distanceToCenterlines.Update()

    # use numpy interface with VTK
    npSurface = dsa.WrapDataObject(distanceToCenterlines.GetOutput())

    distanceArray = npSurface.GetPointData().GetArray(thickness_field_name)
    radiusArray   = npSurface.GetPointData().GetArray(names.VascularRadiusArrayName)

    # This portion evaluates if distance is much higher
    # than the actual radius array
    # This necessarily will need some smoothing

    # Set high and low threshold factors
    # Are they arbitrary?
    highRadiusThresholdFactor = 1.4
    lowRadiusThresholdFactor  = 0.9

    npMaxRadiusLim = highRadiusThresholdFactor*radiusArray
    npMinRadiusLim = lowRadiusThresholdFactor*radiusArray

    distanceArray = np.where(
                        distanceArray > npMaxRadiusLim,
                        npMaxRadiusLim,
                        distanceArray
                    )

    distanceArray = np.where(
                        distanceArray < npMinRadiusLim,
                        radiusArray,
                        distanceArray
                    )

    # Smooth the distance to centerline array to avoid sudden changes of
    # thickness in certain regions
    surface = tools.SmoothSurfacePointField(
                  npSurface.VTKObject,
                  thickness_field_name,
                  niterations=5
              )

    npSurface = dsa.WrapDataObject(surface)

    # Multiply by WLR to have a prelimimar thickness array
    # I assume that the WLR is the same for medium sized arteries
    # but I can change this in a point-wise manner based on
    # the local radius array by using the algorithm contained
    # in the vmtksurfacearrayoperation script
    distanceArray = npSurface.GetPointData().GetArray(thickness_field_name)
    radiusArray   = npSurface.GetPointData().GetArray(names.VascularRadiusArrayName)

    if set_uniform_wlr:

        npSurface.PointData.append(
            dsa.VTKArray([
                uniform_wlr_value*(2.0*r)
                for r in distanceArray
            ]),
            thickness_field_name
        )

    else:
        print("Using non uniform WLR", end="\n")

        # Compute are store local WLR for debug
        localWLRArray = dsa.VTKArray([
                            _compute_local_wlr(2.0*r)
                            for r in distanceArray
                        ])

        npSurface.PointData.append(
            localWLRArray,
            "LocalWLR"
        )

        # Compute thickness array and replace the thickness array
        # (originally stored as the distance to neck array with a new one)
        npSurface.PointData.append(
            localWLRArray*(2.0*distanceArray),
            thickness_field_name
        )

    vascular_surface = npSurface.VTKObject
    vascular_surface.GetPointData().RemoveArray(names.VascularRadiusArrayName)

    return vascular_surface

def UpdateAbnormalHemodynamicsRegions(
        vascular_surface: names.polyDataType,
        field_name: str,
        atherosclerotic_factor: float=1.20,
        red_regions_factor: float=0.95
    )   -> names.polyDataType:
    """Update fields on an aneurysm surface based on adjacent hemodynamics."""

    # Factor array: compute WallTypeArrayName if not yet on the surface
    if names.WallTypeArrayName not in tools.GetCellArrays(vascular_surface):
        vascular_surface = wm.WallTypeClassification(vascular_surface)

    npSurface = dsa.WrapDataObject(vascular_surface)

    wallTypeArray = npSurface.GetCellData().GetArray(
                        names.WallTypeArrayName
                    )

    # Add abnormal factor array
    # This is important to have a smooth field to multiply with the
    # thickness array (scale factor can be viewed as a continous
    # distribution in contrast to the WallType array that is discrete)
    abnormalFactorArray = dsa.VTKArray(
                              np.ones(shape=wallTypeArray.shape)
                          )

    # update with scale factors
    abnormalFactorArray[
        wallTypeArray == wm.IaWallTypes["AtheroscleroticWall"]
    ] = atherosclerotic_factor

    abnormalFactorArray[
        wallTypeArray == wm.IaWallTypes["RedWall"]
    ] = red_regions_factor

    npSurface.CellData.append(
        abnormalFactorArray,
        names.AbnormalFactorArrayName
    )

    vascular_surface = npSurface.VTKObject

    # Interpolate AbnormalFactorArray cell data to point data
    vascular_surface = tools.CellFieldToPointField(
                           vascular_surface,
                           names.AbnormalFactorArrayName
                       )

    npSurface = dsa.WrapDataObject(vascular_surface)

    abnormalFactorArray = npSurface.GetPointData().GetArray(
                              names.AbnormalFactorArrayName
                          )

    fieldToBeUpdated = npSurface.GetPointData().GetArray(
                           field_name
                       )

    npSurface.PointData.append(
        abnormalFactorArray*fieldToBeUpdated,
        field_name
    )

    vascular_surface = npSurface.VTKObject
    vascular_surface.GetCellData().RemoveArray(names.AbnormalFactorArrayName)

    return vascular_surface

def ComputeVasculatureThicknessWithAneurysm(
        vascular_surface: names.polyDataType,
        centerlines: names.polyDataType=None,
        thickness_field_name: str=names.ThicknessArrayName,
        set_uniform_wlr: bool=False,
        uniform_wlr_value: float=const.WlrMedium,
        neck_comp_mode: str="interactive",
        gdistance_to_neck_array_name: str=names.DistanceToNeckArrayName,
        aneurysm_type: str="",
        aneurysm_influence_dist: float=0.5,
        scale_factor: float=0.75,
        parent_vascular_surface: names.polyDataType=None,
        dome_point: tuple=None,
        abnormal_thickness: bool=False,
        atherosclerotic_factor: float=1.20,
        red_regions_factor: float=0.95,
        nsmooth_iterations: float=5
    )   -> names.polyDataType:
    """Calculate and set aneurysm thickness.

    Based on the vasculature thickness distribution, defined as the outside
    portion of the complete geometry from the neck selected by the user,
    estimates an aneurysm thickness by averaging the vasculature thickness
    using as weight function the inverse distance to the
    "aneurysm-influenced" region line. The estimated aneurysm thickness is,
    then, set on the aneurysm surface in the thickness array.

    The aneurysm-influenced neck line is defined as the region between the
    neck line (provided by the user or computed automatically) and the path
    that is at a distance of 'AneurysmInfluencedRegionDistance' value (in
    mm; default 0.5 mm) from the neck line.  This strip around the aneurysm
    is imagined as a region of the original vasculature that had its
    thickness changed by the aneurysm growth.

    If the surface does not already have the 'DistanceToNeckArray' scalar, then
    it will prompt the user to select the neck line, which will be stored on
    the surface. Alternatively, the user may select the option "neck_comp_mode"
    as 'automatic', which estimates a neck line (see function
    'MarkAneurysmalRegion').

    The aneurysm sac thickness may be estimated as 'uniform', the default
    behavior, or using the abnormal wall thickness based on the adjacent
    hemodynamics to the aneurysm wall: the TAWSS and OSI fields (controlled by
    setting the option 'abnormal_thickness' to True). In this last case, the
    passed suface must have these two field from a CFD simulation.

    The aneurysm abnormal thickness is computed based on a 'wall type array',
    and hence increase or deacrease the sac thickness. The procedure is as
    follows: With a global thickness array already defined on the surface,
    update the thickness based on the wall type array created based on the
    hemodynamics variables, by multiplying it by a factor defined below. As
    explained in the function WallTypeCharacterization of wallmotion.py, the
    three types of wall and the operation performed here for each are:

    .. table:: Local wall type characterization
        :widths: auto

        =====   =============== =========
        Label   Wall Type       Operation
        =====   =============== =========
            0   Normal wall     Nothing (default = 1)
            1   Atherosclerotic Increase thickness (default factor = 1.20)
            2   "Red" wall      Decrease thickness (default factor = 0.95)
        =====   =============== =========

    The multiplying factors for the atherosclerotic and red wall must be
    provided, with default values given above. The function will look for the
    array named "WallType" for defining its operation or compute it on the fly.
    """

    # Compute thickness of the vascular tree portion
    vascular_surface = ComputeVasculatureThickness(
                            vascular_surface,
                            centerlines,
                            thickness_field_name=thickness_field_name,
                            set_uniform_wlr=set_uniform_wlr,
                            uniform_wlr_value=uniform_wlr_value
                        )

    # Compute the distance to neck array
    if gdistance_to_neck_array_name not in tools.GetPointArrays(vascular_surface):
        vascular_surface = ComputeDistanceToAneurysmNeck(
                               vascular_surface,
                               mode=neck_comp_mode,
                               aneurysm_type=aneurysm_type,
                               parent_vascular_surface=parent_vascular_surface
                           )

    # Surface with thickness and distnce to neck
    npDistanceSurface = dsa.WrapDataObject(vascular_surface)

    # Update both fields with selection
    thicknessArray = npDistanceSurface.GetPointData().GetArray(
                         thickness_field_name
                     )

    distanceToNeckArray = npDistanceSurface.GetPointData().GetArray(
                              gdistance_to_neck_array_name
                          )

    # First compute aneurysm thickness based on vasculature thickness
    # the vasculature is selection value > 0
    onVasculature = distanceToNeckArray > aneurysm_influence_dist

    # Filter thickness and neckScalars
    vasculatureThicknesses = onVasculature*thicknessArray
    vasculatureDistances   = onVasculature*distanceToNeckArray

    # Aneurysm thickness as weighted average
    aneurysmThickness = scale_factor*np.average(
                            vasculatureThicknesses,
                            weights=np.array([
                                1.0/x if x != 0.0 else 0.0
                                for x in vasculatureDistances
                            ])
                        )

    print(
        "Aneurysm thickness computed: {}".format(
            aneurysmThickness
        ),
        end="\n"
    )

    # Then, substitute thickness array by aneurysmThickness
    thicknessArray[vasculatureThicknesses == 0.0] = aneurysmThickness

    vascular_surface = npDistanceSurface.VTKObject

    if abnormal_thickness:
        vascular_surface = UpdateAbnormalHemodynamicsRegions(
                               vascular_surface,
                               field_name=thickness_field_name,
                               atherosclerotic_factor=atherosclerotic_factor,
                               red_regions_factor=red_regions_factor
                           )

    # After array created, smooth it hard
    vascular_surface = tools.SmoothSurfacePointField(
                           vascular_surface,
                           thickness_field_name,
                           niterations=nsmooth_iterations
                       )

    return vascular_surface


def ComputeVasculatureThicknessWithNAneurysms(
        vascular_surface: names.polyDataType,
        centerlines: names.polyDataType=None,
        thickness_field_name: str=names.ThicknessArrayName,
        set_uniform_wlr: bool=False,
        uniform_wlr_value: float=const.WlrMedium,
        naneurysms: int=1,
        aneurysm_type: str="",
        gdistance_to_neck_array_name: str=names.DistanceToNeckArrayName,
        neck_comp_mode: str="interactive",
        parent_vascular_surface: names.polyDataType=None,
        dome_point: tuple=None
    )   -> names.polyDataType:
    #this version will account for more than one aneurysm case
    raise NotImplementedError("Not yet implemented.")

    # import re
    # # Check if there is any 'DistanceToNeck<i>' array in points arrays
    # # where 'i' indicates that more than one aneurysm are present on
    # # the surface.
    # r = re.compile(gdistance_to_neck_array_name + ".*")

    # distanceToNeckArrayNames = list(filter(r.match, pointArrays))

    # for id_ in range(naneurysms):

    #     # Update neck array name if more than one aneurysm
    #     arrayName = gdistance_to_neck_array_name + str(id_ + 1) \
    #                 if naneurysms > 1 \
    #                 else gdistance_to_neck_array_name

def ComputeVasculatureElasticityWithAneurysm(
        vascular_surface: names.polyDataType,
        elasticity_field_name: str=names.ElasticityArrayName,
        aneurysm_elasticity_mode: str="uniform",
        arteries_elasticity: float=5e6,
        aneurysm_elasticity: float=2e6,
        neck_comp_mode: str="interactive",
        gdistance_to_neck_array_name: str=names.DistanceToNeckArrayName,
        aneurysm_type: str="",
        parent_vascular_surface: names.polyDataType=None,
        dome_point: tuple=None,
        abnormal_elasticity: bool=False,
        atherosclerotic_factor: float=1.20,
        red_regions_factor: float=0.95,
        nsmooth_iterations: float=5
    )   -> names.polyDataType:
    """Calculate and set aneurysm and vascular elasticity.

    Based on a value for the aneurysm elasticity and the arterial elasticity,
    set them on the vascular surface. The arterial elasticity is considered to
    be uniform, whereas the aneurysm elasticity accepts two modes:

        * 'uniform': uniform elasticity;
        * 'linear': elasticity linearly varying from the arterial value to a
            value set by the user too.

    The neck contour that devides the aneurysm sac is either provided by the
    user or computed automatically, through the array 'DistanceToNeck' that
    marks the neck contour with zero values. If the surface does not already
    have the 'DistanceToNeck' scalar array, then it will prompt the user to
    select the neck line, which will be stored on the surface. Alternatively,
    the user may select the option "neck_comp_mode" as 'automatic', which
    estimates a neck line (see function 'MarkAneurysmalRegion').

    The option 'abnormal_elasticity' allows for the automatic update of the
    aneurysm elasticity based on the adjacent hemodynamics to the aneurysm
    wall: the TAWSS and OSI fields. In this last case, the passed surface must
    have these two field from a CFD simulation.

    The aneurysm abnormal elasticity is computed based on a 'wall type array',
    and hence increase or deacrease the sac elasticity. The procedure is as
    follows: With a global elasticity array already defined on the surface,
    update the elasticity based on the wall type array created based on the
    hemodynamics variables, by multiplying it by a factor defined below. As
    explained in the function WallTypeCharacterization of wallmotion.py, the
    three types of wall and the operation performed here for each are:

    .. table:: Local wall type characterization
        :widths: auto

        =====   =============== =========
        Label   Wall Type       Operation
        =====   =============== =========
            0   Normal wall     Nothing (default = 1)
            1   Atherosclerotic Increase elasticity (default factor = 1.20)
            2   "Red" wall      Decrease elasticity (default factor = 0.95)
        =====   =============== =========

    The multiplying factors for the atherosclerotic and red wall must be
    provided, with default values given above. The function will look for the
    array named "WallType" for defining its operation or compute it on the fly.
    """

    # Compute the distance to neck array (here serving only as a neck contour)
    if gdistance_to_neck_array_name not in tools.GetPointArrays(vascular_surface):
        vascular_surface = ComputeDistanceToAneurysmNeck(
                               vascular_surface,
                               mode=neck_comp_mode,
                               aneurysm_type=aneurysm_type,
                               parent_vascular_surface=parent_vascular_surface
                           )


    # Surface with thickness and distnce to neck
    npDistanceSurface = dsa.WrapDataObject(vascular_surface)

    distanceArray = npDistanceSurface.PointData.GetArray(
                        gdistance_to_neck_array_name
                    )

    # Array to hold the actual elasticity array
    elasticities = dsa.VTKArray(
                        np.zeros(
                            shape=vascular_surface.GetNumberOfPoints()
                        )
                    )

    # Mark regions based on distance array values
    onAneurysm  = distanceArray <= 0.0
    outAneurysm = distanceArray > 0.0

    elasticities[outAneurysm] = arteries_elasticity

    # One single aneurysm expected here
    if aneurysm_elasticity_mode == "uniform":

        elasticities[onAneurysm] = aneurysm_elasticity

    elif aneurysm_elasticity_mode == "linear":

        # Fundus and neck elasticity
        neckElasticity   = arteries_elasticity
        fundusElasticity = aneurysm_elasticity

        # Distances on the aneurysm are negative: max distance is actually min
        maxDistance = -min(distanceArray)

        # Angular coeff. for linear elasticity on the aneurysm sac
        angCoeff = \
            (neckElasticity - fundusElasticity)/maxDistance

        elasticities[onAneurysm] = \
            dsa.VTKArray([
                neckElasticity + angCoeff*distance
                for distance in distanceArray[onAneurysm]
            ])

    else:
        raise ValueError(
                  """Aneurysm elasticity mode either 'uniform'
                  or 'linear'. {} passed.""".format(
                      aneurysm_elasticity_mode
                  )
              )

    npDistanceSurface.PointData.append(
        elasticities,
        elasticity_field_name
    )

    vascular_surface = npDistanceSurface.VTKObject

    if abnormal_elasticity:
        vascular_surface = UpdateAbnormalHemodynamicsRegions(
                               vascular_surface,
                               field_name=elasticity_field_name,
                               atherosclerotic_factor=atherosclerotic_factor,
                               red_regions_factor=red_regions_factor
                           )

    # After array created, smooth it hard
    vascular_surface = tools.SmoothSurfacePointField(
                           vascular_surface,
                           elasticity_field_name,
                           niterations=nsmooth_iterations
                       )

    return vascular_surface

def ComputeDistanceToAneurysmNeck(
        vascular_surface: names.polyDataType,
        mode: str="interactive",
        gdistance_to_neck_array_name: str=names.DistanceToNeckArrayName,
        aneurysm_type: str="",
        parent_vascular_surface: names.polyDataType=None,
        parent_vascular_centerline: names.polyDataType=None
    )   -> names.polyDataType:
    """Compute the geodesic distance to an aneurysm neck.

    Given a vascular surface with an aneurysm, computes the geodesic distance
    to the aneurysm neck by thre different methods:

        *   'interactive': the user is prompted to interactively draw the
            aneurysm contour;

        *   'automatic': automatically marks a 3D contour on the aneurysm
            surface that separates the sac from the vasculature (see
            'MarkAneurysmalRegion' function);

        *   'plane': computes an approximate neck plane, based on Piccinelli's
            publication (see 'ComputeAneurysmNeckPlane' function).

    The automatic mode uses the parent vessels surface to estimate the neck
    contour, if this surface is not passed, it will be computed and the user
    will be prompted to clip its inlet and outlets.
    """

    if mode == "interactive":

        vascular_surface = MarkAneurysmSacManually(
                               vascular_surface,
                               aneurysm_neck_array_name=gdistance_to_neck_array_name
                           )

    elif mode == "automatic":

        vascular_surface = MarkAneurysmalRegion(
                               vascular_surface,
                               parent_vascular_surface=parent_vascular_surface,
                               gdistance_to_neck_array_name=gdistance_to_neck_array_name
                           )

    elif mode == "plane":
        raise NotImplementedError(
                  "Clipping by the neck plane not yet implemented."
              )

    else:
        raise ValueError(
                  """Neck computation mode either 'interactive'
                  or 'automatic'. {} passed.""".format(
                      neck_comp_mode
                  )
              )

    return vascular_surface
