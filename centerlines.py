"""Operations on centerlines"""

import vtk
import morphman
from vmtk import vtkvmtk
from vmtk import vmtkscripts

from . import constants as const
from . import polydatatools as tools

def ComputeOpenCenters(surface):
    """Compute barycenters of inlets and outlets.

    Computes the geometric center of each open boundary of the model. Computes
    two lists: one with the inlet coordinates (tuple) and another with the
    outlets coordinates also as tuples of three components:

        Inlet coords:  [(xi, yi, zi)]
        Outlet coords: [(xo1,yo1,zo1),
                        (xo2,yo2,zo2),
                        ...
                        (xon,yon,zon)]

    for a model with a single inlet and n outlets. The inlet is defined as the
    open boundary with largest radius.
    """

    inletCenters = list()
    outletCenters = list()

    radiusArray = 'Radius'
    normalsArray = 'Normals'
    pointArrays = ['Point1', 'Point2']

    referenceSystems = vtkvmtk.vtkvmtkBoundaryReferenceSystems()
    referenceSystems.SetInputData(surface)
    referenceSystems.SetBoundaryRadiusArrayName(radiusArray)
    referenceSystems.SetBoundaryNormalsArrayName(normalsArray)
    referenceSystems.SetPoint1ArrayName(pointArrays[int(const.zero)])
    referenceSystems.SetPoint2ArrayName(pointArrays[int(const.one)])
    referenceSystems.Update()

    openBoundariesRefSystem = referenceSystems.GetOutput()
    nOpenBoundaries = openBoundariesRefSystem.GetPoints().GetNumberOfPoints()

    maxRadius = openBoundariesRefSystem.GetPointData().GetArray(
        radiusArray
    ).GetRange()[int(const.one)]

    for i in range(nOpenBoundaries):
        # Get radius and center
        center = tuple(openBoundariesRefSystem.GetPoints().GetPoint(i))
        radius = openBoundariesRefSystem.GetPointData().GetArray(
            radiusArray
        ).GetValue(i)

        if radius == maxRadius:
            inletCenters.append(center)
        else:
            outletCenters.append(center)

    return inletCenters, outletCenters

def GenerateCenterlines(surface,
                        source_points=None,
                        target_points=None):
    """Compute centerlines, given source and target points."""

    noEndPoints = source_points == None and target_points == None

    if noEndPoints:
        source_points, target_points = ComputeOpenCenters(surface)
    else:
        pass

    # Get inlet and outlet centers of surface
    CapDisplacement = 0.0
    FlipNormals = 0
    CostFunction = '1/R'
    AppendEndPoints = 1
    CheckNonManifold = 0

    Resampling = 1
    ResamplingStepLength = 0.1
    SimplifyVoronoi = 0
    RadiusArrayName = "MaximumInscribedSphereRadius"

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

    centerlineFilter.SetRadiusArrayName(RadiusArrayName)
    centerlineFilter.SetCostFunction(CostFunction)
    centerlineFilter.SetFlipNormals(FlipNormals)
    centerlineFilter.SetAppendEndPointsToCenterlines(AppendEndPoints)
    centerlineFilter.SetSimplifyVoronoi(SimplifyVoronoi)

    centerlineFilter.SetCenterlineResampling(Resampling)
    centerlineFilter.SetResamplingStepLength(ResamplingStepLength)
    centerlineFilter.Update()

    return centerlineFilter.GetOutput()

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

# This class was adapted from the 'patchandinterpolatecenterlines.py' script
# distributed with VMTK in https://github.com/vmtk/vmtk
def GetDivergingPoints(surface):
    """Get diverging data from centerline morphology."""
    inlets, outlets = ComputeOpenCenters(surface)
    aneurysmPoint = tools.SelectSurfacePoint(surface)

    parentCenterlines = GenerateCenterlines(surface, inlets, outlets)

    # Compute daughter centerlines
    # Build daughter centerlines
    daughterCenterlines = list()

    # First outlet centerline
    daughterCenterlines.append(
        GenerateCenterlines(
            surface,
            [outlets[0]], 
            [outlets[1], aneurysmPoint]
        )
    )

    # Compute clipping and diverging points
#     centerlineSpacing = geo.Distance(
#             parentCenterlines.GetPoint(10),
#             parentCenterlines.GetPoint(11)
#         )

    # Origianlly got from the scripts by Picinelli
    divRatioToSpacingTolerance = 2.0
    divTolerance = 0.1/10# centerlineSpacing/divRatioToSpacingTolerance   

    # Compute clipping and divergence points
    divergingData = morphman.get_bifurcating_and_diverging_point_data(
            parentCenterlines, 
            daughterCenterlines[0], 
            divTolerance
        )

    return divergingData

    # Store points 
#     """Compute parent artery centerlines.
#
#     Uses the Morphman library to extract the
#     hypothetical parent artery centerlines.
#     Uses the procedure originally proposed in
#
#         Ford et al. (2009). An objective approach
#         to digital removal of saccular aneurysms:
#         technique and applications.
#         DOI: 10.1259/bjr/67593727
#
#     and improved in
#
#         Bergersen et al. (2019).
#         Automated and Objective Removal of Bifurcation
#         Aneurysms: Incremental Improvements, and Validation
#         Against Healthy Controls.
#         DOI: 10.1016/j.jbiomech.2019.109342
#
#     """
#
#     self._clipping_points = vtk.vtkPoints()
#     self._diverging_points = vtk.vtkPoints()
#
#     for key in self._diverging_data.keys():
#         self._clipping_points.InsertNextPoint(
#                 self._diverging_data[key].get('end_point')
#             )
#
#         self._diverging_points.InsertNextPoint(
#                 self._diverging_data[key].get('div_point')
#             )
#
#     # Compute parent centerline reconstruction
#     patchCenterlines = morphman.create_parent_artery_patches(
#             self._centerlines,
#             self._clipping_points,
#             siphon=True,
#             bif=True
#         )
#
#     tools.WriteSurface(patchCenterlines, '/home/iagolessa/tmp_patch.vtp')
#     self._parent_centerlines = morphman.interpolate_patch_centerlines(
#                                     patchCenterlines,
#                                     self._centerlines,
#                                     additionalPoint=None,
#                                     lower='bif',
#                                     version=True
#                                 )
