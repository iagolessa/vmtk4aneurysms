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

"""Collection of tools that operate or are related to centerlines."""

import vtk
import numpy as np
from morphman.manipulate_curvature import extract_single_line
from vtkmodules.numpy_interface import dataset_adapter as dsa
from scipy.signal import find_peaks_cwt

from vmtk import vtkvmtk
from vmtk import vmtkscripts

from . import names
from . import constants as const
from . import polydatageometry as geo
from . import polydatatools as tools

def ComputeOpenCenters(
        surface: names.polyDataType,
        interactive: bool=False
    )   -> tuple:
    """Compute barycenters outwards normals of inlets and outlets.

    Computes the geometric center and outward normal of each open boundary of
    the model. Computes two dictionaries with the centers as keys (tuples) and
    the normals as values, one for the inlets and another for the outlets. Both
    normals and centers are given as tuples:

    Dict inlet: {(x1, y1, z1): (nx1, ny1, nz1)}

    Dict outlet: {(x1, y1, z1): (nx1, ny1, nz1),
                  (x2, y2, z2): (nx2, ny2, nz2),
                  ...,
                  (xN, yN, zN): (nxN, nyN, nzN)}

    for a model with a single inlet and n outlets. The magnitude of the normals
    is the radius of the open profile. The inlet is defined as the open
    boundary with largest radius.
    """
    # I noticed some weird behavior of the vtkvmtkBoundaryReferenceSystems
    # when using it with a surface that has passed through the
    # vtkPolyDataNormals filter. i couldn't solve the problem, so I am putting
    # a clean-up and copy of the input surface to avoid any problems for safety
    # but keep in mind that this did not solved the problem.

    # Clean up any arrays in surface and make copy of surface
    newSurface = tools.CopyVtkObject(surface)
    newSurface = tools.CleanupArrays(newSurface)

    # Get complete ref systems
    pointArrays  = ['Point1', 'Point2']
    boundaryRadiusArrayName  = 'Radius'
    boundaryNormalsArrayName = 'BoundaryNormals'

    boundarySystems = vtkvmtk.vtkvmtkBoundaryReferenceSystems()
    boundarySystems.SetInputData(newSurface)
    boundarySystems.SetBoundaryRadiusArrayName(boundaryRadiusArrayName)
    boundarySystems.SetBoundaryNormalsArrayName(boundaryNormalsArrayName)
    boundarySystems.SetPoint1ArrayName(pointArrays[0])
    boundarySystems.SetPoint2ArrayName(pointArrays[1])
    boundarySystems.Update()

    referenceSystems = boundarySystems.GetOutput()

    npEndPoints = dsa.WrapDataObject(referenceSystems)
    endCenters  = npEndPoints.GetPoints()
    radiusArray = npEndPoints.PointData.GetArray(boundaryRadiusArrayName)

    # The normal are outward the vascular domain
    outNormalsArray = npEndPoints.PointData.GetArray(boundaryNormalsArrayName)

    # Get inlet and outlet ids
    if interactive:
        capper = vtkvmtk.vtkvmtkCapPolyData()
        capper.SetInputData(newSurface)
        capper.SetDisplacement(0.0)
        capper.SetInPlaneDisplacement(0.0)
        capper.SetCellEntityIdsArrayName(names.CellEntityIdsArrayName)
        capper.Update()

        cappedSurface = capper.GetOutput()

        # Select the inlet point
        inletPickPoint = tools.PickPointSeedSelector()
        inletPickPoint.SetSurface(cappedSurface)
        inletPickPoint.InputInfo("Select a point on the inlet\n")
        inletPickPoint.Execute()

        outletPickPoint = tools.PickPointSeedSelector()
        outletPickPoint.SetSurface(cappedSurface)
        outletPickPoint.InputInfo(
            "Select the aneurysm branch outlets (if branching > 3 branches)\n"
        )
        outletPickPoint.Execute()

        inletSeeds  = inletPickPoint.PickedSeeds
        outletSeeds = outletPickPoint.PickedSeeds

        # Locate selected inlet and outlets ref. systems
        locator = vtk.vtkPointLocator()
        locator.SetDataSet(referenceSystems)
        locator.BuildLocator()

        inletIds = [locator.FindClosestPoint(inletSeeds.GetPoint(pid))
                    for pid in range(inletSeeds.GetNumberOfPoints())]

        outletIds = [locator.FindClosestPoint(outletSeeds.GetPoint(pid))
                     for pid in range(outletSeeds.GetNumberOfPoints())]

    else:
        # Select as inlet the profile with largest section area
        inletIds  = [radiusArray.argmax()]
        outletIds = [idx
                     for idx in range(referenceSystems.GetNumberOfPoints())
                     if idx not in inletIds]


    inletRefSystems = {tuple(c): tuple(r*n)
                       for c, r, n in zip(
                                           endCenters[inletIds],
                                           radiusArray[inletIds],
                                           outNormalsArray[inletIds]
                                       )
                       }

    outletRefSystems = {tuple(c): tuple(r*n)
                        for c, r, n in zip(
                                            endCenters[outletIds],
                                            radiusArray[outletIds],
                                            outNormalsArray[outletIds]
                                        )
                       }

    return inletRefSystems, outletRefSystems

# Code of this functions was based on the vmtkcenterlines.py script of the
# VMTK library: https://github.com/vmtk/vmtk
def GenerateCenterlines(
        surface: names.polyDataType,
        source_points: list = None,
        target_points: list = None,
        append_end_points: bool=True
    )   -> names.polyDataType:
    """Compute centerlines, given source and target points."""

    noEndPoints = source_points == None and target_points == None

    if noEndPoints:
        inletRefs, outletRefs = ComputeOpenCenters(surface)

        source_points = list(inletRefs.keys())
        target_points = list(outletRefs.keys())

    # Get inlet and outlet centers of surface
    CapDisplacement = 0.0
    FlipNormals = 0
    CostFunction = '1/R'
    AppendEndPoints = append_end_points
    CheckNonManifold = 0

    Resampling = 1
    ResamplingStepLength = 0.1
    SimplifyVoronoi = 0

    # Clean and triangulate
    surface = tools.Cleaner(surface)

    surfaceTriangulator = vtk.vtkTriangleFilter()
    surfaceTriangulator.SetInputData(surface)
    surfaceTriangulator.PassLinesOff()
    surfaceTriangulator.PassVertsOff()
    surfaceTriangulator.Update()

    # Cap surface
    surfaceCapper = vtkvmtk.vtkvmtkCapPolyData()
    surfaceCapper.SetInputConnection(surfaceTriangulator.GetOutputPort())
    surfaceCapper.SetDisplacement(CapDisplacement)
    surfaceCapper.SetInPlaneDisplacement(CapDisplacement)
    surfaceCapper.Update()

    centerlineInputSurface = surfaceCapper.GetOutput()

    # Get source and target ids of closest point
    sourceSeedIds = vtk.vtkIdList()
    targetSeedIds = vtk.vtkIdList()

    pointLocator = vtk.vtkPointLocator()
    pointLocator.SetDataSet(centerlineInputSurface)
    pointLocator.BuildLocator()

    for point in source_points:
        id_ = pointLocator.FindClosestPoint(point)
        sourceSeedIds.InsertNextId(id_)

    for point in target_points:
        id_ = pointLocator.FindClosestPoint(point)
        targetSeedIds.InsertNextId(id_)

    # Compute centerlines
    centerlineFilter = vtkvmtk.vtkvmtkPolyDataCenterlines()
    centerlineFilter.SetInputData(centerlineInputSurface)

    centerlineFilter.SetSourceSeedIds(sourceSeedIds)
    centerlineFilter.SetTargetSeedIds(targetSeedIds)

    centerlineFilter.SetRadiusArrayName(names.VascularRadiusArrayName)
    centerlineFilter.SetCostFunction(CostFunction)
    centerlineFilter.SetFlipNormals(FlipNormals)
    centerlineFilter.SetAppendEndPointsToCenterlines(AppendEndPoints)
    centerlineFilter.SetSimplifyVoronoi(SimplifyVoronoi)

    centerlineFilter.SetCenterlineResampling(Resampling)
    centerlineFilter.SetResamplingStepLength(ResamplingStepLength)
    centerlineFilter.Update()

    return centerlineFilter.GetOutput()

# This function was adapted from the Morphman library,
# available at https://github.com/KVSlab/morphMan
def GetDivergingPoint(
        centerline: names.polyDataType,
        tolerance: float
    )   -> tuple:
    """Get diverging point on centerline bifurcation.

    Args:
        centerline (vtkPolyData): centerline of a bifurcation.
        tolerance (float): tolerance.
    Returns:
        point (tuple): diverging point.
    """
    line0 = extract_single_line(centerline, 0)
    line1 = extract_single_line(centerline, 1)

    nPoints = min(line0.GetNumberOfPoints(), line1.GetNumberOfPoints())

    bifPointIndex = None
    getPoint0 = line0.GetPoints().GetPoint
    getPoint1 = line1.GetPoints().GetPoint

    for index in range(0, nPoints):
        distance = geo.Distance(getPoint0(index), getPoint1(index))

        if distance > tolerance:
            bifPointIndex = index
            break

    return getPoint0(bifPointIndex)

def ComputeCenterlineGeometry(centerlines):
    """Compute centerline sections and geometry."""

    calcGeometry = vmtkscripts.vmtkCenterlineGeometry()
    calcGeometry.Centerlines = centerlines
    calcGeometry.Execute()

    # Computation of centerlines attributes (parallel theory)
    calcAttributes = vmtkscripts.vmtkCenterlineAttributes()
    calcAttributes.Centerlines = calcGeometry.Centerlines
    calcAttributes.Execute()

    return calcAttributes.Centerlines

def ComputeVoronoiDiagram(
        vascular_surface: names.polyDataType
    )   -> names.polyDataType:
    """Compute Voronoi diagram of a vascular surface."""

    voronoiDiagram = vmtkscripts.vmtkDelaunayVoronoi()
    voronoiDiagram.Surface = vascular_surface
    voronoiDiagram.CheckNonManifold = True
    voronoiDiagram.Execute()

    return voronoiDiagram.Surface

def ComputeTubeSurface(
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


    # Get bounds of model
    centerlineBounds  = centerline.GetBounds()
    radiusArrayBounds = centerline.GetPointData().GetArray(names.VascularRadiusArrayName).GetValueRange()
    maxSphereRadius   = radiusArrayBounds[1]

    # To enlarge the box: could be a fraction of maxSphereRadius
    # tests show that the whole radius is appropriate
    enlargeBoxBounds  = maxSphereRadius

    modelBounds = np.array(centerlineBounds) + \
                  np.array(const.nSpatialDimensions*[-enlargeBoxBounds, enlargeBoxBounds])

    # Extract image with tube function from model
    modeller = vtkvmtk.vtkvmtkPolyBallModeller()
    modeller.SetInputData(centerline)
    modeller.SetRadiusArrayName(names.VascularRadiusArrayName)

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

def ComputeVoronoiEnvelope(
        voronoi_surface: names.polyDataType,
        smooth: bool=True
    )   -> names.polyDataType:
    """Compute the envelope surface of a Voronoi diagram."""

    VoronoiBounds     = voronoi_surface.GetBounds()
    radiusArrayBounds = voronoi_surface.GetPointData().GetArray(names.VascularRadiusArrayName).GetValueRange()
    maxSphereRadius   = radiusArrayBounds[1]
    enlargeBoxBounds  = maxSphereRadius

    modelBounds = np.array(VoronoiBounds) + \
                  np.array(const.nSpatialDimensions*[-enlargeBoxBounds, enlargeBoxBounds])

    # Building the envelope image function
    modeller = vtkvmtk.vtkvmtkPolyBallModeller()
    modeller.SetInputData(voronoi_surface)
    modeller.SetRadiusArrayName(names.VascularRadiusArrayName)

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

    if smooth:
        return tools.SmoothSurface(envelope)
    else:
        return envelope

# This function was adapted from the Morphman library,
# available at https://github.com/KVSlab/morphMan
def ComputeClPatchEndPointParameters(
        patch_centerlines: names.polyDataType,
        patch_id: int
    )   -> tuple:
    """Compute the tangent, point and radius at the ends of a centerline.

    The result depend on the id of the centerline patch: it returns the
    end points closest to the bifurcation (or patched region) of the original
    centerline. The tangent direction is always towards the path of the
    centerline.
    """

    # Set cell to a vtk cell
    cell = vtk.vtkGenericCell()
    patch_centerlines.GetCell(patch_id, cell)

    if (patch_id == 0):
        # Then get the last point
        lastPointId       = cell.GetNumberOfPoints() - 1
        beforeLastpointId = cell.GetNumberOfPoints() - 2

        point0 = np.array(cell.GetPoints().GetPoint(lastPointId))
        point1 = np.array(cell.GetPoints().GetPoint(beforeLastpointId))

        radius0 = patch_centerlines.GetPointData().GetArray(
                      names.VascularRadiusArrayName
                  ).GetTuple1(cell.GetPointId(lastPointId))

        tan = point1 - point0
        vtk.vtkMath.Normalize(tan)

    else:
        # then get the first point
        point0 = np.array(cell.GetPoints().GetPoint(0))
        point1 = np.array(cell.GetPoints().GetPoint(1))
        radius0 = patch_centerlines.GetPointData().GetArray(
                      names.VascularRadiusArrayName
                  ).GetTuple1(cell.GetPointId(0))

        tan = point1 - point0
        vtk.vtkMath.Normalize(tan)

    return tan, point0, radius0

def CenterlineMaxLength(
        centerlines: names.polyDataType
    )   -> float:
    """Compute max. length of vascular tree centerline."""

    if names.vmtkAbscissasArrayName not in tools.GetPointArrays(centerlines):

        centerlines = ComputeCenterlineGeometry(
                            centerlines
                        )


    abscissasRange = centerlines.GetPointData().GetArray(
                            names.vmtkAbscissasArrayName
                        ).GetRange()

    return max(abscissasRange) - min(abscissasRange)

def CenterlineBranching(
        centerlines: names.polyDataType
    )   -> names.polyDataType:
    """Define centerline branching fields."""

    branches = vmtkscripts.vmtkBranchExtractor()
    branches.Centerlines = centerlines
    branches.Execute()

    return branches.Centerlines

def CenterlineReferenceSystems(
        centerlines: names.polyDataType
    )   -> names.polyDataType:
    """Compute VTK Polydata with reference systems of vasculature
    bifurcations."""

    # Computing the bifurcation reference system
    bifsRefSystem = vmtkscripts.vmtkBifurcationReferenceSystems()

    bifsRefSystem.Centerlines       = centerlines
    bifsRefSystem.RadiusArrayName   = names.VascularRadiusArrayName
    bifsRefSystem.GroupIdsArrayName = names.vmtkGroupIdsArrayName
    bifsRefSystem.ReferenceSystemsNormalArrayName = names.vmtkReferenceSystemsNormalArrayName
    bifsRefSystem.Execute()

    return bifsRefSystem.ReferenceSystems

def SmoothCenterline(
        centerlines: names.polyDataType,
        niterations: int=100,
        smoothing_factor: float=1.25
    )   -> names.polyDataType:
    """Smooth the 3D contour of centerline."""

    smoother = vmtkscripts.vmtkCenterlineSmoothing()

    smoother.Centerlines = centerlines
    smoother.NumberOfSmoothingIterations = niterations
    smoother.SmoothingFactor = smoothing_factor
    smoother.Execute()

    return smoother.Centerlines

def _robust_offset_centerline(
        centerlines: names.polyDataType,
        ref_systems: names.polyDataType,
        bif_group_id: int
    )   -> names.polyDataType:

    # Get original max length
    # Compute original range of abscissas
    maxLength = CenterlineMaxLength(centerlines)

    # TODO: How to better handle this?
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

        newMaxLength = CenterlineMaxLength(offsetCenterlines)

        if not np.abs(newMaxLength - maxLength) > 1.0:
            break

    return offsetCenterlines

def ComputeCenterlinePropertiesOffBifurcation(
        centerlines: names.polyDataType,
        bif_point: tuple
    )   -> names.polyDataType:
    """Compue centerline geometry and branching from ref. bifurcation.

    Given a vascular tree centerline and a point next to the reference
    bifurcation, compute its geometry and branching properties. The Abscissas
    are offset to the bifurcation closest to the ref. point passed.
    """

    geoCenterlines = ComputeCenterlineGeometry(centerlines)
    geoCenterlines = CenterlineBranching(geoCenterlines)

    # Compute ref. systems to get ICA bif
    referenceSystems = CenterlineReferenceSystems(geoCenterlines)

    # Get ICA-MCA-ACA bifurcation
    icaBifGroupId = int(
                        tools.GetFieldValueAtClosestPoint(
                            referenceSystems,
                            bif_point,
                            names.vmtkGroupIdsArrayName
                        )
                    )

    return _robust_offset_centerline(
               geoCenterlines,
               referenceSystems,
               icaBifGroupId
           )

def SplitCenterlineObject(
        centerlines: names.polyDataType
    )   -> dict:
    """Split tree centerline into dict of its centerlines components.

    The dictionary stores the centerline object, its length and the group id in
    the following format:

        {group_id: {"object": centerline, "length": length}}
    """

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
                             names.vmtkLengthArrayName
                          ).GetRange()
                      )
        })

    return individualCenterlines

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

    individualCenterlines = SplitCenterlineObject(centerlines)

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
