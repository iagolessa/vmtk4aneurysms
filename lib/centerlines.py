"""Operations on centerlines"""

import vtk
import numpy as np
from morphman import extract_single_line

from vmtk import vtkvmtk
from vmtk import vmtkscripts

from . import names
from . import constants as const
from . import polydatageometry as geo
from . import polydatatools as tools

_dimensions = int(const.three)
_radiusArrayName = "MaximumInscribedSphereRadius"

def ComputeOpenCenters(
        surface: names.polyDataType
    )   -> tuple:
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
    # I noticed some weird behavior of the vtkvmtkBoundaryReferenceSystems
    # when using it with a surface that has passed through the
    # vtkPolyDataNormals filter. i couldn't solve the problem, so I am putting
    # a clean-up and copy of the input surface to avoid any problems for safety
    # but keep in mind that this did not solved the problem.

    # Clean up any arrays in surface and make copy of surface
    newSurface = vtk.vtkPolyData()
    newSurface.DeepCopy(surface)

    newSurface = tools.CleanupArrays(newSurface)

    inletCenters  = []
    outletCenters = []

    pointArrays  = ['Point1', 'Point2']
    radiusArray  = 'Radius'
    normalsArray = 'BoundaryNormals'

    referenceSystems = vtkvmtk.vtkvmtkBoundaryReferenceSystems()
    referenceSystems.SetInputData(newSurface)
    referenceSystems.SetBoundaryRadiusArrayName(radiusArray)
    referenceSystems.SetBoundaryNormalsArrayName(normalsArray)
    referenceSystems.SetPoint1ArrayName(pointArrays[0])
    referenceSystems.SetPoint2ArrayName(pointArrays[1])
    referenceSystems.Update()

    openBoundariesRefSystem = referenceSystems.GetOutput()
    nOpenBoundaries = openBoundariesRefSystem.GetPoints().GetNumberOfPoints()

    maxRadius = openBoundariesRefSystem.GetPointData().GetArray(
                    radiusArray
                ).GetRange()[1]

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

def GenerateCenterlines(
        surface: names.polyDataType,
        source_points: list = None,
        target_points: list = None
    )   -> names.polyDataType:
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

    centerlineFilter.SetRadiusArrayName(_radiusArrayName)
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
    radiusArrayBounds = centerline.GetPointData().GetArray(_radiusArrayName).GetValueRange()
    maxSphereRadius   = radiusArrayBounds[1]

    # To enlarge the box: could be a fraction of maxSphereRadius
    # tests show that the whole radius is appropriate
    enlargeBoxBounds  = maxSphereRadius

    modelBounds = np.array(centerlineBounds) + \
                  np.array(_dimensions*[-enlargeBoxBounds, enlargeBoxBounds])

    # Extract image with tube function from model
    modeller = vtkvmtk.vtkvmtkPolyBallModeller()
    modeller.SetInputData(centerline)
    modeller.SetRadiusArrayName(_radiusArrayName)

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
    radiusArrayBounds = voronoi_surface.GetPointData().GetArray(_radiusArrayName).GetValueRange()
    maxSphereRadius   = radiusArrayBounds[1]
    enlargeBoxBounds  = maxSphereRadius

    modelBounds = np.array(VoronoiBounds) + \
                  np.array(_dimensions*[-enlargeBoxBounds, enlargeBoxBounds])

    # Building the envelope image function
    modeller = vtkvmtk.vtkvmtkPolyBallModeller()
    modeller.SetInputData(voronoi_surface)
    modeller.SetRadiusArrayName(_radiusArrayName)

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

def ComputeClPatchEndPointParameters(
        patch_centerlines: names.polyDataType,
        patch_id: int
    )   -> tuple:

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
                      _radiusArrayName
                  ).GetTuple1(cell.GetPointId(lastPointId))

        tan = point1 - point0
        vtk.vtkMath.Normalize(tan)

    else:
        # then get the first point
        point0 = np.array(cell.GetPoints().GetPoint(0))
        point1 = np.array(cell.GetPoints().GetPoint(1))
        radius0 = patch_centerlines.GetPointData().GetArray(
                      _radiusArrayName
                  ).GetTuple1(cell.GetPointId(0))

        tan = point1 - point0
        vtk.vtkMath.Normalize(tan)

    return tan, point0, radius0
