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

from scipy.signal import find_peaks_cwt
from vtk.numpy_interface import dataset_adapter as dsa

from .lib import names
from .lib import centerlines as cl
from .lib import constants as const
from .lib import polydatatools as tools
from .lib import polydatageometry as geo

def _bifurcation_aneurysm_clipping_points(
        vascular_surface: names.polyDataType,
        aneurysm_point: tuple,
        inlet_points: list = None,
        outlet_points: list = None
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

    # Get the clipping points
    noEndPoints = inlet_points == None and outlet_points == None

    if noEndPoints:
        inletRefs, outletRefs = cl.ComputeOpenCenters(vascular_surface)

        inlet_points = list(inletRefs.keys())
        outlet_points = list(outletRefs.keys())

    inlets = inlet_points
    outlets = outlet_points

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
        aneurysm_point: tuple,
        inlet_points: list = None,
        outlet_points: list = None
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
    # Get the clipping points
    noEndPoints = inlet_points == None and outlet_points == None

    if noEndPoints:
        inletRefs, outletRefs = cl.ComputeOpenCenters(vascular_surface)

        inlet_points = list(inletRefs.keys())
        outlet_points = list(outletRefs.keys())

    inlets = inlet_points
    outlets = outlet_points

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
    tubeFunction.SetPolyBallRadiusArrayName(names.VascularRadiusArrayName)

    lastSphere = vtk.vtkSphere()
    lastSphere.SetRadius(radius*1.5)
    lastSphere.SetCenter(center)

    # Compute array on the surface to select point in the tube
    points = npSurface.GetPoints()
    voronoiVectors = points - center

    voronoiVectorDots = dsa.VTKArray(
                            [vtk.vtkMath.Dot(voronoiVector, tangent)
                             for voronoiVector in voronoiVectors]
                        )

    tubeValues = dsa.VTKArray(
                    [tubeFunction.EvaluateFunction(point)
                     for point in points]
                )

    sphereValues = dsa.VTKArray(
                        [lastSphere.EvaluateFunction(point)
                         for point in points]
                    )

    # Set conditions
    inTube = tubeValues <= 0.0
    inSphere = sphereValues < 0.0
    notOnPatch = voronoiVectorDots < 0.0

    # Modify filter array
    inPatchArray[inTube] = 1
    inPatchArray[inSphere & notOnPatch] = 0

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
        dome_point: tuple=None,
        inlet_ref_systems: dict=None,
        outlet_ref_systems: dict=None,
        interactive: bool=False
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

    # Get inlets and outlets ref. systems
    if inlet_ref_systems == None and outlet_ref_systems == None:

        inlet_ref_systems, outlet_ref_systems = cl.ComputeOpenCenters(
                                                    vascular_surface,
                                                    interactive=interactive
                                                )

    inletCenter = list(inlet_ref_systems.keys())
    outletCenters = list(outlet_ref_systems.keys())

    voronoi       = cl.ComputeVoronoiDiagram(vascular_surface)
    centerlines   = cl.GenerateCenterlines(
                        vascular_surface,
                        source_points=inletCenter,
                        target_points=outletCenters
                    )

    # Smooth the Voronoi diagram
    smoothedVoronoi = mplib.voronoi_operations.smooth_voronoi_diagram(
                          voronoi,
                          centerlines,
                          0.25 # Smoothing factor, recommended by Ms. Piccinelli
                      )


    # 1) Compute parent centerline reconstruction
    dome_point = tools.SelectSurfacePoint(
                    vascular_surface,
                    input_text="Select point on the aneurysm surface\n"
                 ) if dome_point is None else dome_point

    # Get clipping points on model centerlines and order them correctly
    if aneurysmInBifurcation:
        dictClipPoints = _bifurcation_aneurysm_clipping_points(
                             vascular_surface,
                             dome_point,
                             inlet_points=inletCenter,
                             outlet_points=outletCenters
                         )

        orderedClipPoints = [dictClipPoints.get("bif",  None),
                             dictClipPoints.get("dau0", None),
                             dictClipPoints.get("dau1", None)]

    else:
        dictClipPoints = _lateral_aneurysm_clipping_points(
                             vascular_surface,
                             dome_point,
                             inlet_points=inletCenter,
                             outlet_points=outletCenters
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
    healthyVessel = cl.ComputeVoronoiEnvelope(newVoronoi)

    # Clip the parent vascular surface
    inlet_ref_systems.update(outlet_ref_systems)

    for center, normal in inlet_ref_systems.items():

        # Invert normal and displace center by one profile diameter
        center = tuple(np.array(center) - np.array(normal))

        healthyVessel = SeamPlaneTubularStructureMarker(
                            healthyVessel,
                            center,
                            normal,
                            seam_scalar_array_name=names.SeamScalarsArrayName
                        )

        healthyVessel = tools.ClipWithScalar(
                            healthyVessel,
                            names.SeamScalarsArrayName,
                            const.zero
                        )

    healthyVessel.GetPointData().RemoveArray(names.SeamScalarsArrayName)

    return healthyVessel

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
        aneurysm_point: tuple,
        inlet_points: list = None,
        outlet_points: list = None
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
    noEndPoints = inlet_points == None and outlet_points == None

    if noEndPoints:
        inletRefs, outletRefs = cl.ComputeOpenCenters(vascular_surface)

        inlet_points = list(inletRefs.keys())
        outlet_points = list(outletRefs.keys())

    inlets = inlet_points
    outlets = outlet_points

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
        aneurysm_point: tuple,
        inlet_points: list = None,
        outlet_points: list = None
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

    # Get the clipping points
    noEndPoints = inlet_points == None and outlet_points == None

    if noEndPoints:
        inletRefs, outletRefs = cl.ComputeOpenCenters(vascular_surface)

        inlet_points = list(inletRefs.keys())
        outlet_points = list(outletRefs.keys())

    inlets = inlet_points
    outlets = outlet_points

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
        result_clip_array: str = names.AneurysmalRegionArrayName
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
            barycenter = const.nSpatialDimensions*[0]

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

    # These normals point to the aneurysm direction
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

    aneurysmPoint = tools.SelectSurfacePoint(
                        vascular_surface,
                        input_text="Select point on the aneurysm surface\n"
                    ) if aneurysm_point is None else aneurysm_point

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
        aneurysmal_region_array: str=names.AneurysmalRegionArrayName,
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

def _mark_aneurysm_sac_interactively(
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

def _geo_distance_to_aneurysmal_region_neck(
        vascular_surface: names.polyDataType,
        parent_vascular_surface: names.polyDataType=None,
        parent_vascular_centerline: names.polyDataType=None,
        gdistance_to_neck_array_name: str=names.DistanceToNeckArrayName,
        aneurysm_type: str="",
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
                           aneurysmal_region_array=names.AneurysmalRegionArrayName,
                           aneurysm_type=aneurysm_type
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
    pointIds = tools.GetClosestContourOnSurface(
                   vascular_surface,
                   neckContour
               )

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

def _geo_distance_to_aneurysm_plane_neck(
        vascular_surface: names.polyDataType,
        parent_vascular_surface: names.polyDataType=None,
        parent_vascular_centerline: names.polyDataType=None,
        gdistance_to_neck_array_name: str=names.DistanceToNeckArrayName,
        aneurysm_type: str="",
        aneurysm_point: tuple=None
    )   -> names.polyDataType:
    """Marks the aneurysmal region with an array of (geodesic) distances to the
    neck plane contour.

    Similar to '_geo_distance_to_aneurysmal_region_neck', this function
    marks the vascular model passed with an array whose zero value marks the
    contour of the neck plane, compute with 'ComputeAneurysmNeckPlane'.  The
    rest of the array is given by the geodesic distance of the point to the
    neck contour.

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

    # Get plane neck and aneurysm clipped
    neckPlane, aneurysmSurface = ComputeAneurysmNeckPlane(
                                     vascular_surface,
                                     aneurysm_type=aneurysm_type,
                                     parent_vascular_surface=parent_vascular_surface,
                                     min_variable="area",
                                     aneurysm_point=aneurysm_point
                                 )

    # The next algorithm needs to use a poly data result of a CLIP!
    # That is the case with the ComputeAneurysmNeckPlane algorithm
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

    # Extract the bounday of the cutted cells
    # This provides a rough approximation of where the neck contour
    # cuts the surface
    boundaryExtractor = vtkvmtk.vtkvmtkPolyDataBoundaryExtractor()
    boundaryExtractor.SetInputData(aneurysmSurface)
    boundaryExtractor.Update()

    neckContour = boundaryExtractor.GetOutput()

    # Locator to find closest points
    pointIds = tools.GetClosestContourOnSurface(
                   vascular_surface,
                   neckContour
               )

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

def SeamPlaneTubularStructureMarker(
        surface: names.polyDataType,
        plane_center: tuple,
        plane_normal: tuple,
        seed_point: tuple=None,
        seam_scalar_array_name: str=names.SeamScalarsArrayName
    )   -> names.polyDataType:

    if seed_point is None:
        locator = vtk.vtkPointLocator()
        locator.SetDataSet(surface)
        locator.BuildLocator()

        seedPointId = locator.FindClosestPoint(plane_center)
        seed_point = surface.GetPoint(seedPointId)

    plane = vtk.vtkPlane()
    plane.SetOrigin(plane_center)
    plane.SetNormal(plane_normal)

    seamFilter = vtkvmtk.vtkvmtkTopologicalSeamFilter()
    seamFilter.SetInputData(surface)
    seamFilter.SetClosestPoint(seed_point)
    seamFilter.SetSeamScalarsArrayName(seam_scalar_array_name)
    seamFilter.SetSeamFunction(plane)
    seamFilter.Update()

    return seamFilter.GetOutput()

def ClipVasculatureWithPlane(
        vascular_surface: names.polyDataType,
        plane_center: tuple,
        plane_normal: tuple
    )   -> names.polyDataType:
    """Clip vascular tree section with a plane.

    Given a plane center and normal, clip the passed vascular surface model at
    the plane. The portion of the surface kept is on the normal direction.
    """

    # Clip vessel at inlet location
    vascular_surface = SeamPlaneTubularStructureMarker(
                           vascular_surface,
                           plane_center=plane_center,
                           plane_normal=plane_normal,
                           seam_scalar_array_name=names.SeamScalarsArrayName
                       )

    # The SeamScalars are positive (1.0) in the region of the positiove
    # direction of the plane normal so use inside_out False.
    return tools.ClipWithScalar(
               vascular_surface,
               names.SeamScalarsArrayName,
               const.zero,
               inside_out=False
           )

def ClipVasculature(
        vascular_surface: names.polyDataType,
        centerlines=None
    )   -> names.polyDataType:
    """Clip a vascular surface segment, by selecting end points.

    Given a vascular surface, the user is prompted to select points
    on the surface that 1) identifies the surface's bulk and 2) where the
    vasculature should be clipped. Uses, internally, the
    'vmtksurfaceendclipper' script.
    """

    if centerlines is None:
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

    # Operate on copy
    vascular_surface = tools.CopyVtkObject(vascular_surface)

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
                            aneurysmal_region_array=names.AneurysmalRegionArrayName,
                            aneurysm_type=aneurysm_type
                        )

    # Clip aneurysmal portion (scalars < 0)
    clippedAneurysmalSurface = tools.ClipWithScalar(
                                   aneurysmalSurface,
                                   names.AneurysmalRegionArrayName,
                                   const.zero
                               )

    clippedAneurysmalSurface = tools.ExtractConnectedRegion(
                                   clippedAneurysmalSurface,
                                   'largest'
                               )

    clippedAneurysmalSurface.GetPointData().RemoveArray(
        names.AneurysmalRegionArrayName
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
    # It is impotant to clip the aneurysm here to clip only the aneurysmal
    # region surface
    neckCenter = neckPlane.GetOrigin()
    neckNormal = neckPlane.GetNormal()

    # How to clip the aneurysm and get the correct plne neck contour to compute
    # the distance to i too I could use what they do in the vmtkbranchclipper
    # script where they clip the surface exaclty on that value wth remeshing of
    # elements there

    # Clip final aneurysm surface: the side to where the normal point
    planeContour = tools.ContourCutWithPlane(
                       aneurysmalSurface,
                       neckCenter,
                       neckNormal
                   )

    # Get point on the plane contour closest to the neck center
    locator = vtk.vtkPointLocator()
    locator.SetDataSet(planeContour)
    locator.BuildLocator()

    seedPointId = locator.FindClosestPoint(neckCenter)
    seedPoint = planeContour.GetPoint(seedPointId)

    aneurysmalSurface = SeamPlaneTubularStructureMarker(
                            aneurysmalSurface,
                            neckCenter,
                            neckNormal,
                            seed_point=seedPoint,
                            seam_scalar_array_name=names.SeamScalarsArrayName
                        )

    aneurysmSacSurface = tools.ClipWithScalar(
                             aneurysmalSurface,
                             names.SeamScalarsArrayName,
                             const.zero,
                             inside_out=False
                         )

    return neckPlane, aneurysmSacSurface

def ComputeGeodesicDistanceToAneurysmNeck(
        vascular_surface: names.polyDataType,
        mode: str="interactive",
        gdistance_to_neck_array_name: str=names.DistanceToNeckArrayName,
        aneurysm_type: str="",
        aneurysm_point: tuple=None,
        parent_vascular_surface: names.polyDataType=None,
        parent_vascular_centerline: names.polyDataType=None,
        nsmoothing_iterations: int=10
    )   -> names.polyDataType:
    """Mark the aneurysm neck contour and compute the geodesic distance to it.

    Given a vascular surface with an aneurysm, computes the geodesic distance
    to the aneurysm neck by three different methods:

        *   'interactive': the user is prompted to interactively draw the
            aneurysm contour;

        *   'automatic': automatically marks a 3D contour on the aneurysm
            surface that separates the sac from the vasculature (see
            '_geo_distance_to_aneurysmal_region_neck' function);

        *   'plane': computes an approximate neck plane, based on Piccinelli's
            publication (see 'ComputeAneurysmNeckPlane' function).

    The automatic mode uses the parent vessels surface to estimate the neck
    contour, if this surface is not passed, it will be computed and the user
    will be prompted to clip its inlet and outlets.

    .. warning::
        Adds a little bit of smoothing in the resulting distance field to avoid
        discontnuities in the original field due to its dependency on the
        underlying discretization of the surface. For the 'plane' mode, it may
        distance the neck contour plane from an actual plane. This may be
        controlled with the 'nsmoothing_iterations' function argument.

    .. warning::
        If using the 'plane' mode, as the smoothing may render the plane
        actually not a perfect plane, if you really need to use a plane, than
        use directly the function 'ComputeAneurysmNeckPlane', which returns the
        neck plane as a vtkPlane and the clipped aneurysm sac by that plane.
    """

    if mode == "interactive":

        vascular_surface = _mark_aneurysm_sac_interactively(
                               vascular_surface,
                               aneurysm_neck_array_name=gdistance_to_neck_array_name
                           )

    elif mode == "automatic":

        vascular_surface = _geo_distance_to_aneurysmal_region_neck(
                               vascular_surface,
                               parent_vascular_surface=parent_vascular_surface,
                               gdistance_to_neck_array_name=gdistance_to_neck_array_name,
                               aneurysm_type=aneurysm_type
                           )

    elif mode == "plane":
        # The smoohing here alters a little bit the 'plane'
        vascular_surface = _geo_distance_to_aneurysm_plane_neck(
                               vascular_surface,
                               parent_vascular_surface=parent_vascular_surface,
                               aneurysm_type=aneurysm_type,
                               aneurysm_point=aneurysm_point
                           )

    else:
        raise ValueError(
                  """Neck computation mode either 'interactive', 'automatic',
                  or 'plane'; {} passed.""".format(
                      neck_comp_mode
                  )
              )

    # Add a little bit of smoothing in the field to avoid discontinuities
    return tools.SmoothSurfacePointField(
               vascular_surface,
               gdistance_to_neck_array_name,
               niterations=nsmoothing_iterations
           )

def ClipAneurysmSacSurface(
        vascular_surface: names.polyDataType,
        mode: str="interactive",
        parent_vascular_surface: names.polyDataType=None,
        parent_vascular_centerline: names.polyDataType=None,
        aneurysm_type: str="",
        aneurysm_point: tuple=None
    )   -> names.polyDataType:
    """Clip the aneurysm sac surface from the vascular surface model.

    Given the vascular model with an aneurysm, clip the aneurysm sac surface
    based on the neck contour computed via the function
    'ComputeGeodesicDistanceToAneurysmNeck' (see its docstring for the
    available methods of defining the aneurysm neck contour). Returns a tuple
    with the aneurysm and the rest of the surface clipped.

    .. warning::
        If using the 'plane' mode, as the smoothing of the distance field may
        render the plane actually not a perfect plane, if you really need to
        use a plane, than use directly the function 'ComputeAneurysmNeckPlane',
        which returns the neck plane as a vtkPlane and the clipped aneurysm sac
        by that plane.

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
    (aneurysm sac surface, clipped surface) (tuple) -- the surface of the
    aneurysm sac clipped from the vascular surface model and the vascular model
    surface clipped.
    """

    # Clean up any arrays on the surface
    vascular_surface = tools.Cleaner(vascular_surface)
    # vascular_surface = tools.CleanupArrays(vascular_surface)

    # Copy the original surface and store it so the final array is interpolated
    # back to it
    copiedSurface = tools.CopyVtkObject(vascular_surface)

    # Perform the procedure on a clean surface
    vascular_surface = tools.CleanupArrays(vascular_surface)

    if mode == "plane":

        # Get plane neck and aneurysm clipped
        neckPlane, clippedAneurysmSurface = ComputeAneurysmNeckPlane(
                                                vascular_surface,
                                                aneurysm_type=aneurysm_type,
                                                parent_vascular_surface=parent_vascular_surface,
                                                min_variable="area",
                                                aneurysm_point=aneurysm_point
                                            )

        # Clip the rest of the surface
        vascularSurfaceNoAneurysm = tools.ClipWithPlane(
                                        vascular_surface,
                                        neckPlane.GetOrigin(),
                                        neckPlane.GetNormal(),
                                        inside_out=True
                                    )

    else:
        # Based on the available methods, mark the surface with the neck array
        markedSurface = ComputeGeodesicDistanceToAneurysmNeck(
                            vascular_surface,
                            mode=mode,
                            parent_vascular_surface=parent_vascular_surface,
                            aneurysm_type=aneurysm_type,
                            aneurysm_point=aneurysm_point
                        )

        # Clip the aneurysm sac (aneurysm marked with negative values)
        clippedAneurysmSurface = tools.ClipWithScalar(
                                     markedSurface,
                                     names.DistanceToNeckArrayName,
                                     const.zero
                                 )

        # Needed for the surface branching procedure
        vascularSurfaceNoAneurysm = tools.ClipWithScalar(
                                        markedSurface,
                                        names.DistanceToNeckArrayName,
                                        const.zero,
                                        inside_out=False
                                    )

        clippedAneurysmSurface.GetPointData().RemoveArray(
            names.DistanceToNeckArrayName
        )
        vascularSurfaceNoAneurysm.GetPointData().RemoveArray(
            names.DistanceToNeckArrayName
        )

    return clippedAneurysmSurface, vascularSurfaceNoAneurysm

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

def WallTypeClassification(
        surface: names.polyDataType,
        low_wss: float=5.0,
        high_wss: float=10.0,
        low_osi: float=0.001,
        high_osi: float=0.01,
        distance_to_neck_array: str=names.DistanceToNeckArrayName,
        neck_iso_value: float=const.NeckIsoValue
    )   -> names.polyDataType:
    """Based on the WSS hemodynamics, characterize an aneurysm wall morphology.

    Based on the TAWSS and OSI fields, identifies the aneurysm regions prone to
    atherosclerotic walls (thicker walls) and red wall (thinner) by adding a
    new array on the passed surface name "WallType" with the following values:

    .. table:: Local wall type characterization
        :widths: auto

        =====   ===============
        Label   Wall Type
        =====   ===============
            0   Normal wall
            1   Atherosclerotic
            2   "Red" wall
        =====   ===============

    Classifications based on the references:

        [1] Furukawa et al. "Hemodynamic characteristics of hyperplastic
        remodeling lesions in cerebral aneurysms". PLoS ONE. 2018 Jan
        16;13:1–11.

        [2] Cebral et al. "Local hemodynamic conditions associated with focal
        changes in the intracranial aneurysm wall". American Journal of
        Neuroradiology.  2019; 40(3):510–6.
    """
    normalWall  = const.IaWallTypes["RegularWall"]
    thickerWall = const.IaWallTypes["AtheroscleroticWall"]
    thinnerWall = const.IaWallTypes["RedWall"]

    # Maybe put this limiting values to be passed by the user
    # for flexibility
    limitHemodynamics = {names.TAWSS: {"low": low_wss, "high": high_wss},
                         names.OSI  : {"low": low_osi, "high": high_osi}#,
                         #names.RRT  : {"low": 0.25,  "high": 0.75}
                        }

    arraysInSurface = tools.GetPointArrays(surface) + \
                      tools.GetCellArrays(surface)

    if distance_to_neck_array not in arraysInSurface:
        print("Distance to neck array name not in surface. Computing it.")

        surface = _mark_aneurysm_sac_interactively(
                      surface,
                      aneurysm_neck_array_name=distance_to_neck_array
                  )

    elif names.TAWSS not in arraysInSurface:
        raise ValueError("TAWSS array not in surface!")

    elif names.OSI not in arraysInSurface:
        raise ValueError("OSI array not in surface!")

    fieldsDf = tools.vtkPolyDataToDataFrame(surface)

    # Add int field which will indicate the thicker regions
    # zero indicates normal wall... the aneuysm portion wil be updated
    fieldsDf[names.WallTypeArrayName] = normalWall

    # Groups of conditions
    isAneurysm = fieldsDf[distance_to_neck_array] < const.NeckIsoValue

    isHighWss = fieldsDf[names.TAWSS] > limitHemodynamics[names.TAWSS]["high"]
    isLowWss  = fieldsDf[names.TAWSS] < limitHemodynamics[names.TAWSS]["low"]

    isHighOsi = fieldsDf[names.OSI] > limitHemodynamics[names.OSI]["high"]
    isLowOsi  = fieldsDf[names.OSI] < limitHemodynamics[names.OSI]["low"]

    # isHighRrt = fieldsDf[names.RRT] > limitHemodynamics[names.RRT]["high"]
    # isLowRrt = fieldsDf[names.RRT] < limitHemodynamics[names.RRT]["low"]

    thickerWallCondition = (isAneurysm) & (isLowWss)  & (isHighOsi)# & (isHighRrt)
    thinnerWallCondition = (isAneurysm) & (isHighWss) & (isLowOsi) # & (isLowRrt)

    # Update wall type array
    fieldsDf.loc[thickerWallCondition, names.WallTypeArrayName] = thickerWall
    fieldsDf.loc[thinnerWallCondition, names.WallTypeArrayName] = thinnerWall

    hemodynamicSurfaceNumpy = dsa.WrapDataObject(surface)

    # Add new field to surface
    hemodynamicSurfaceNumpy.CellData.append(
        dsa.VTKArray(fieldsDf[names.WallTypeArrayName]),
        names.WallTypeArrayName
    )

    return hemodynamicSurfaceNumpy.VTKObject

def UpdateAbnormalHemodynamicsRegions(
        vascular_surface: names.polyDataType,
        field_name: str,
        atherosclerotic_factor: float=1.20,
        red_regions_factor: float=0.95
    )   -> names.polyDataType:
    """Update fields on an aneurysm surface based on adjacent hemodynamics."""

    # Factor array: compute WallTypeArrayName if not yet on the surface
    if names.WallTypeArrayName not in tools.GetCellArrays(vascular_surface):
        vascular_surface = WallTypeClassification(vascular_surface)

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
        wallTypeArray == const.IaWallTypes["AtheroscleroticWall"]
    ] = atherosclerotic_factor

    abnormalFactorArray[
        wallTypeArray == const.IaWallTypes["RedWall"]
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
    '_geo_distance_to_aneurysmal_region_neck').

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
        vascular_surface = ComputeGeodesicDistanceToAneurysmNeck(
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


# def ComputeVasculatureThicknessWithNAneurysms(
#         vascular_surface: names.polyDataType,
#         centerlines: names.polyDataType=None,
#         thickness_field_name: str=names.ThicknessArrayName,
#         set_uniform_wlr: bool=False,
#         uniform_wlr_value: float=const.WlrMedium,
#         naneurysms: int=1,
#         aneurysm_type: str="",
#         gdistance_to_neck_array_name: str=names.DistanceToNeckArrayName,
#         neck_comp_mode: str="interactive",
#         parent_vascular_surface: names.polyDataType=None,
#         dome_point: tuple=None
#     )   -> names.polyDataType:
#     #this version will account for more than one aneurysm case
#     raise NotImplementedError("Not yet implemented.")

#     # import re
#     # # Check if there is any 'DistanceToNeck<i>' array in points arrays
#     # # where 'i' indicates that more than one aneurysm are present on
#     # # the surface.
#     # r = re.compile(gdistance_to_neck_array_name + ".*")

#     # distanceToNeckArrayNames = list(filter(r.match, pointArrays))

#     # for id_ in range(naneurysms):

#     #     # Update neck array name if more than one aneurysm
#     #     arrayName = gdistance_to_neck_array_name + str(id_ + 1) \
#     #                 if naneurysms > 1 \
#     #                 else gdistance_to_neck_array_name

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
    estimates a neck line (see function
    '_geo_distance_to_aneurysmal_region_neck').

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
        vascular_surface = ComputeGeodesicDistanceToAneurysmNeck(
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

    # After array created, smooth it hard to remove discontinuity
    vascular_surface = tools.SmoothSurfacePointField(
                           vascular_surface,
                           elasticity_field_name,
                           niterations=nsmooth_iterations
                       )

    return vascular_surface

def _get_field_value_at_closest_point(
        vtk_object,
        point,
        point_field_name
    ):

    if point_field_name in tools.GetCellArrays(vtk_object):

        vtk_object = tools.CellFieldToPointField(
                        vtk_object,
                        cell_field_name=point_field_name
                    )

    # Get closest point to ICA at the bifurcation found
    bifPoint = tools.LocateClosestPointOnPolyData(
                    vtk_object,
                    point
                )

    npVtkObject = dsa.WrapDataObject(vtk_object)

    # Get ID of ICA bifurcation
    bifId = np.all(
                npVtkObject.Points == bifPoint,
                axis=1
            ).argmax()

    # Get ID of ICA in GroupIds for offset attributes
    bifGroupIds = npVtkObject.PointData.GetArray(point_field_name)

    return bifGroupIds[bifId]

def _split_centerline_object(centerlines):
    """Split tree centerline into dict of its centerlines components."""

    npCenterlines = dsa.WrapDataObject(centerlines)

    centerlineIds = list(
                        set(
                            npCenterlines.CellData.GetArray(
                                names.vmtkCenterlineIdsArrayName
                            )
                        )
                    )

    # Cretae dict to better storing of separate centerlines
    individualCenterlines = {}

    for cl_id in centerlineIds:

        individualCenterlines[cl_id] = {}

        # Extract centerline portion and transfrom GroupId to point for later
        clPortion = tools.CellFieldToPointField(
                        tools.ExtractPortion(
                            centerlines,
                            names.vmtkCenterlineIdsArrayName,
                            cl_id
                        ),
                        cell_field_name=names.vmtkGroupIdsArrayName
                    )

        individualCenterlines[cl_id].update(
            {"object": clPortion}
        )

        # Add total length of each
        individualCenterlines[cl_id].update({
            "length": max(
                          clPortion.GetCellData().GetArray(
                             names. vmtkLengthArrayName
                          ).GetRange()
                      )
        })

    return individualCenterlines

def _robust_offset_centerline(
        centerlines: names.polyDataType,
        ref_systems: names.polyDataType,
        bif_group_id: int
    )   -> names.polyDataType:

    # Get original max length
    # Compute original range of abscissas
    maxLength = cl.CenterlineMaxLength(centerlines)

    # Iterate to avoid spourious errors in offsert computation
    for _ in range(0,1000):

        offsetFilter = vtkvmtk.vtkvmtkCenterlineReferenceSystemAttributesOffset()

        offsetFilter.SetInputData(
            tools.CopyVtkObject(centerlines)
        )

        offsetFilter.SetReferenceSystems(ref_systems)
        offsetFilter.SetAbscissasArrayName(names.vmtkAbscissasArrayName)
        offsetFilter.SetNormalsArrayName(names.vmtkParallelTransportArrayName)

        offsetFilter.SetOffsetAbscissasArrayName(names.vmtkAbscissasArrayName)
        offsetFilter.SetOffsetNormalsArrayName(names.vmtkParallelTransportArrayName)

        offsetFilter.SetGroupIdsArrayName(names.vmtkGroupIdsArrayName)
        offsetFilter.SetCenterlineIdsArrayName(names.vmtkCenterlineIdsArrayName)

        offsetFilter.SetReferenceSystemsNormalArrayName(
            names.vmtkReferenceSystemsNormalArrayName
        )

        offsetFilter.SetReferenceSystemsGroupIdsArrayName(
            names.vmtkGroupIdsArrayName
        )

        offsetFilter.SetReferenceGroupId(bif_group_id)
        offsetFilter.Update()

        offsetCenterlines = offsetFilter.GetOutput()

        newMaxLength = cl.CenterlineMaxLength(offsetCenterlines)

        if not np.abs(newMaxLength - maxLength) > 1.0:
            break

    return offsetCenterlines

def ComputeICABendsLimits(
        centerlines: names.polyDataType,
        abscissas_peak_widths: np.ndarray=np.arange(30,40)
    )   -> list:
    """Given the vascular tree centerline containing the ICA segment with
    Abscissas defined from its bifurcation, compute the intervals of its bends.

    A precise definition of the bends of the internal carotide artery (ICA) was
    provided by the work:

        M. Piccinelli et al., “Geometry of the Internal Carotid Artery and
        Recurrent Patterns in Location, Orientation, and Rupture Status of
        Lateral Aneurysms: An Image-Based Computational Study”, Neurosurgery,
        vol. 68, nº 5, p. 1270–1285, maio 2011, doi:
        10.1227/NEU.0b013e31820b5242.

    which subdivides the ICA intro bends defined by torsion and curvature
    peaks. This functions implements it based on the procedure proposed in the
    paper. However, the procedure is sensitive to some arguments. For example,
    its is recommended that the passed centerline be smoothed with the
    centerline smoothing procedure in VMTK (a function that encapsulates the
    procedure with suitable arguments tuned for this subdivision of the ICA is
    in the lib/centerline.py module, see 'SmoothCenterline').

    Also, the subdivision depends on the identification of peaks of torsion
    sand curvature of the centerline that is normally a relatively noisy field
    for discretized centerline (hence the recommendation to smooth it). In this
    case the peaks are found with the scipy.signal.find_peaks_cwt function
    which depends on the 'width' argument passsed to the wavelets functions.
    These widths are a lista of possible widths between peaks of the torsion
    and curvature signal, which depends on each vascular case. These argument
    can be passed to this function in the 'abscissas_peak_widths' argument. If
    None is passed, then the default is between 30 and 40, which is pretty
    arbitrary, but were found based on testing with the aneurisk repository
    cases and yield the best results.

    .. warning::
        It is highly recommended to smooth the centerline prior to passing it
        to this function.
    """

    individualCenterlines = _split_centerline_object(centerlines)

    # Get longest centerline
    # ID of ICA clip will be identified in this portion
    idLongestCenterline = max(
                              individualCenterlines,
                              key=lambda idx: individualCenterlines[idx]["length"]
                          )

    longestCenterline = individualCenterlines[idLongestCenterline]["object"]

    minAbscissas, _ = longestCenterline.GetPointData().GetArray(
                          names.vmtkAbscissasArrayName
                      ).GetRange()

    icaCenterline = tools.ClipWithScalar(
                        longestCenterline,
                        names.vmtkAbscissasArrayName,
                        const.zero
                    )

    npIcaCenterline = dsa.WrapDataObject(icaCenterline)

    icaTorsionField   = npIcaCenterline.GetPointData().GetArray(
                            names.TorsionArrayName
                        )

    icaCurvatureField = npIcaCenterline.GetPointData().GetArray(
                            names.CurvatureArrayName
                        )

    icaAbscissasField = npIcaCenterline.GetPointData().GetArray(
                            names.vmtkAbscissasArrayName
                        )

    # Find ids of torsion peaks in smoothed centerline
    # Estimate widths between peaks: between a subdivision of 5 and 2
    # the length of the ICA
    torsionPeaksIds = find_peaks_cwt(
                            abs(icaTorsionField),
                            widths=abscissas_peak_widths
                        )

    # Find ids of torsion peaks in smoothed centerline
    curvaturePeaksIds = find_peaks_cwt(
                            icaCurvatureField,
                            widths=abscissas_peak_widths
                        )

    # Get the peaks and add min and max of ICA abscissas
    torsionPeaksAbscissas = sorted(
                                np.append(
                                    icaAbscissasField[torsionPeaksIds],
                                    [const.zero,
                                     minAbscissas]
                                )
                            )

    curvaturePeaksAbscissas = sorted(
                                    np.append(
                                        icaAbscissasField[curvaturePeaksIds],
                                        [const.zero,
                                         minAbscissas]
                                    )
                                )

    # Sort arrays
    torsionPeaksAbscissas = np.array(list(reversed(torsionPeaksAbscissas)))
    curvaturePeaksAbscissas = np.array(list(reversed(curvaturePeaksAbscissas)))

    # Get distal and proximal torsian peaks
    # with curvature peaks within it
    bendLimits = []
    saveValueForNext = []

    curvatureIntervals = zip(
                            curvaturePeaksAbscissas,
                            curvaturePeaksAbscissas[1:],
                            curvaturePeaksAbscissas[2:]
                        )

    # Identify distal and proximal values to curvature peaks
    for max_abs, centre, min_abs in curvatureIntervals:

        # Compute the 2 enclosing tosion peaks (closest)
        # Divide into upstream values and downstream values
        upstreamTorsionPeaks = torsionPeaksAbscissas[
                                   (torsionPeaksAbscissas <= max_abs) &
                                   (torsionPeaksAbscissas >= centre)
                               ]

        downstreamTorsionPeaks = torsionPeaksAbscissas[
                                     (torsionPeaksAbscissas <= centre) &
                                     (torsionPeaksAbscissas >= min_abs)
                                 ]

        # Handle case where there is no torsion peaks between two
        # curvature peaks
        if downstreamTorsionPeaks.size == 0:
            # Store the upstream value only
            # Get upstream proximal value
            saveValueForNext.append(
                upstreamTorsionPeaks[
                    abs(upstreamTorsionPeaks - centre).argmin()
                ]
            )

            continue

        elif upstreamTorsionPeaks.size == 0:

            # this case occurs when a donwtream was empty
            # Get only downstream value to list

            upstreamLimit = saveValueForNext[0]


            downstreamLimit = downstreamTorsionPeaks[
                                  abs(downstreamTorsionPeaks - centre).argmax()
                              ]

        else:
            # Get upstream proximal value
            upstreamLimit = upstreamTorsionPeaks[
                                abs(upstreamTorsionPeaks - centre).argmin()
                            ]

            # Get downstream distal value
            downstreamLimit = downstreamTorsionPeaks[
                                  abs(downstreamTorsionPeaks - centre).argmax()
                              ]

        bendLimits.append((upstreamLimit, downstreamLimit))

    return bendLimits

def ClipVasculatureOffBifurcation(
        vascular_surface: names.polyDataType,
        centerlines: names.polyDataType,
        clip_vessel_field: str=names.vmtkAbscissasArrayName,
        inlet_vessel_clip_value: float=-40.0,
        outlet_vessel_clip_value: str=8.0,
        bif_point: tuple=None,
        aneurysm_point: tuple=None
    )   -> names.polyDataType:
    """Automatically clip a vessel structure away of a specified bifurcation
    and an aneurysm, if it exists.

    Given a vessel surface model and its centerlines, clip the vessel at points
    espefied by a distance away from a selected bifurcation and away from the
    bifurcation closer to a specified aneurysm. The function is intended to
    clip inlet and outlet boundary conditions in vascular models based a
    predefined distance from a bifurcation and an aneurysm. For example, you
    can select the ICA bifurcation and pass two values identified as the
    distance fom the ICA bifurcation where the vascular model surface will be
    clipped, if no aneurysm on the model. If an aneurysm is present and you
    want to clip after the aneurysm, you can pass the ICA bifurcation point and
    the aneurysm dome point (if not passed, the function prompts the user to
    interactively select it), so the function clips the vasculature after the
    aneurysm bifurcation.

    Both the clip values for inlet (before the bifurcation) as the clip value
    of outlets (after the bifurcation can be passed. The default field used to
    clip is the 'Abscissas' field of distance values along the centerline with
    origin its closest point to the selected bifurcation. If an aneurysm
    exists, then the clip arg. 'outlet_vessel_clip_value' is relative to the
    bifurcation closest to the aneurysm.

    The point closest to the selected bifurcation can be passed via the arg.
    'bif_point'. If None, the user is prompted to interactively select it.
    """

    # Compute centerlines abscissas and other attributes
    geoCenterlines = cl.ComputeCenterlineGeometry(centerlines)

    # Perform branching
    geoCenterlines = cl.CenterlineBranching(geoCenterlines)

    # Computing the bifurcation reference system
    referenceSystems = cl.CenterlineReferenceSystems(geoCenterlines)

    # Get ICA-MCA-ACA bifurcation
    if bif_point is None:

        bif_point = tools.SelectSurfacePoint(
                        vascular_surface,
                        input_text="Select point at the ICA bifurcation\n"
                    )

    # Get closest point to ICA at the bifurcation found
    bifGroupId  = int(
                      _get_field_value_at_closest_point(
                          referenceSystems,
                          bif_point,
                          names.vmtkGroupIdsArrayName
                      )
                  )

    # Offset attributes to the bifurcation
    offsetCenterlines = _robust_offset_centerline(
                            geoCenterlines,
                            referenceSystems,
                            bifGroupId
                        )

    # Identify the bifurcation GroupId and its Abscissas closer to the aneurysm
    if aneurysm_point is None:

        aneurysm_point = tools.SelectSurfacePoint(
                             vascular_surface,
                             input_text="Select a  point on the aneurysm surface"
                         )

    # Get center of bifurcation closest to aneurysm point
    iaClosestPointToBif = tools.LocateClosestPointOnPolyData(
                                referenceSystems,
                                aneurysm_point
                            )

    onlyBifurcations = tools.ExtractPortion(
                           offsetCenterlines,
                           # only centerlines potions of bifs.
                           names.vmtkBlankingArrayName,
                           const.one
                       )

    # Get also the group id of the portion where the aneurysm is
    iaAbscissasClosestBif = _get_field_value_at_closest_point(
                                  onlyBifurcations,
                                  iaClosestPointToBif,
                                  names.vmtkAbscissasArrayName
                            )

    # Get also the group id of the portion where the aneurysm is
    iaGroupIdClosestBif = _get_field_value_at_closest_point(
                                onlyBifurcations,
                                aneurysm_point,
                                names.vmtkGroupIdsArrayName
                          )

    # Check whether passed clip values are within the clip field range
    offsetAbscissasRange = offsetCenterlines.GetPointData().GetArray(
                                names.vmtkAbscissasArrayName
                            ).GetRange()

    if inlet_vessel_clip_value < min(offsetAbscissasRange):

        raise ValueError(
                "{} smaller than min of {} (~ {}).\nSpecify higher value.".format(
                    inlet_vessel_clip_value,
                    clip_vessel_field,
                    round(min(offsetAbscissasRange), 3)
                  )
              )

    if outlet_vessel_clip_value + iaAbscissasClosestBif > max(offsetAbscissasRange):

        raise ValueError(
                "{} distance relative to aneurysm bifurcation is larger than max of {} (~ {}).\nSpecify smaller value.".format(
                    outlet_vessel_clip_value,
                    clip_vessel_field,
                    round(max(offsetAbscissasRange), 3)
                  )
              )

    # Cretae dict to better storing of separate centerlines
    individualCenterlines = _split_centerline_object(offsetCenterlines)

    # Get longest centerline
    # ID of ICA clip will be identified in this portion
    idLongestCenterline = max(
                                individualCenterlines,
                                key=lambda idx: individualCenterlines[idx]["length"]
                            )

    longestCenterline = individualCenterlines[idLongestCenterline]["object"]
    npLongestCenterline = dsa.WrapDataObject(longestCenterline)

    # Clip inlet artery
    clipArray = npLongestCenterline.PointData.GetArray(clip_vessel_field)

    # Get id of the point where to clip
    icaClipPointId = (
                        np.abs(clipArray - inlet_vessel_clip_value)
                     ).argmin()

    icaClipPoint = tuple(npLongestCenterline.Points[icaClipPointId])

    icaClipNormal = tuple(
                        npLongestCenterline.PointData.GetArray(
                            names.vmtkFrenetTangentArrayName
                        )[icaClipPointId]
                    )

    # Clip vessel at inlet location
    vascular_surface = ClipVasculatureWithPlane(
                           vascular_surface,
                           plane_center=icaClipPoint,
                           plane_normal=icaClipNormal
                       )

    # Store the group id to whcih the point belong
    outletClipPoints = {}

    for cl_id, dict_ in individualCenterlines.items():

        npClPortion = dsa.WrapDataObject(dict_["object"])
        clClipField = npClPortion.PointData.GetArray(clip_vessel_field)

        clGroups = npClPortion.PointData.GetArray(
                        names.vmtkGroupIdsArrayName
                    )

        clGroupIdsList = list(set(clGroups))

        # Check whether the centerline have abscissas
        if bifGroupId in clGroupIdsList:

            # If the ica bif. and aneurysm group ids are on the centerline,
            # then use the updated clip value to clip (meaning to clip AFTER
            # the aneurysm abscissa)
            if iaGroupIdClosestBif in clGroupIdsList and \
               iaGroupIdClosestBif > bifGroupId:

                clipValue = iaAbscissasClosestBif + \
                            outlet_vessel_clip_value

            else:
                clipValue = outlet_vessel_clip_value

            # Use the GroupIds as keys to dict to avoid repetitive entries
            outletClipPointId = (
                                    np.abs(
                                        clClipField - clipValue
                                    )
                                ).argmin()

            outletClipPoint  = tuple(npClPortion.Points[outletClipPointId])
            outletClipNormal = tuple(
                                   npClPortion.PointData.GetArray(
                                       names.vmtkFrenetTangentArrayName
                                   )[outletClipPointId]
                               )

            outletClipPoints.update({
                clGroups[outletClipPointId]: {
                    "center": outletClipPoint,
                    "normal": outletClipNormal
                }
            })

        else:
            continue

    for dict_ in outletClipPoints.values():

        # Clip at outlets: invert normal at these positions
        vascular_surface = ClipVasculatureWithPlane(
                               vascular_surface,
                               plane_center=dict_["center"],
                               plane_normal=tuple(-val for val in dict_["normal"])
                           )

    return tools.CleanupArrays(vascular_surface)
