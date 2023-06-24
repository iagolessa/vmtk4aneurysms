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

"""Collection of tools that operate or are relateed to centerlines."""

import vtk
import numpy as np
from morphman.manipulate_curvature import extract_single_line

from vmtk import vtkvmtk
from vmtk import vmtkscripts

from . import names
from . import constants as const
from . import polydatageometry as geo
from . import polydatatools as tools

def ComputeOpenCenters(
        surface: names.polyDataType
    )   -> tuple:
    """Compute barycenters of inlets and outlets.

    Computes the geometric center of each open boundary of the model. Computes
    two lists: one with the inlet coordinates (tuple) and another with the
    outlets coordinates also as tuples of three components:

    Inlet coords:  [(xi, yi, zi)]
    Outlet coords: [(xo1,yo1,zo1), (xo2,yo2,zo2), ...  (xon,yon,zon)]

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

    nProfiles = referenceSystems.GetNumberOfPoints()

    # Get inlet center (larger radius)
    radiusArray = np.array([
        referenceSystems.GetPointData().GetArray(boundaryRadiusArrayName).GetTuple1(i)
        for i in range(nProfiles)
    ])

    # maxRadius = max(radiusArray)
    # inletId = int(np.where(radiusArray == maxRadius)[0])
    inletId = radiusArray.argmax()
    inletCenter = [referenceSystems.GetPoint(inletId)]

    # Get outlets
    outletCenters = []

    # Get centers of outlets
    for profileId in range(nProfiles):

        if profileId != inletId:
            outletCenters.append(referenceSystems.GetPoint(profileId))

    return inletCenter, outletCenters

# Code of this functions was based on the vmtkcenterlines.py script of the
# VMTK library: https://github.com/vmtk/vmtk
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
    """Compute the tangent, point and radius at the ends of a centerline."""

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
