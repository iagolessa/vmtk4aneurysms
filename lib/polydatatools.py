"""Provide functions to work with VTK poly data."""

import os
import sys
import pandas as pd

import vtk
from vmtk import vtkvmtk
from vmtk import vmtkscripts
from vmtk import vmtkrenderer
from vtk.numpy_interface import dataset_adapter as dsa

from . import constants as const
from . import names

def ReadSurface(file_name: str) -> names.polyDataType:
    """Read surface file to VTK object.

    Arguments:
    file_name -- complete path with file name
    """
    # Get extension
    extension = os.path.splitext(file_name)[-1]

    if extension == '.vtp':
        reader = vtk.vtkXMLPolyDataReader()

    elif extension == '.vtk':
        reader = vtk.vtkPolyDataReader()

    elif extension == '.stl':
        reader = vtk.vtkSTLReader()

    else:
        sys.exit('Unrecognized file format.')

    reader.SetFileName(file_name)
    reader.Update()

    return reader.GetOutput()


def ViewSurface(surface, array_name=None):
    """View surface vtkPolyData objects.

    Arguments:
    surface -- the surface to be displayed.
    """
    viewer = vmtkscripts.vmtkSurfaceViewer()
    viewer.Surface = surface

    if array_name != None:
        viewer.ArrayName = array_name
        viewer.DisplayCellData = 1
        viewer.Legend = True

    viewer.Execute()

def WriteSurface(surface: names.polyDataType,
                 file_name: str) -> None:
    """Write surface vtkPolyData.

    Arguments:
    surface -- surface vtkPolyData object
    file_name -- output file name with full path

    Optional arguments:
    mode -- mode to write file (ASCII or binary, default binary)
    """
    extension = os.path.splitext(file_name)[-1]

    if extension == '.vtp':
        writer = vtk.vtkXMLPolyDataWriter()

    elif extension == '.vtk':
        writer = vtk.vtkPolyDataWriter()

    elif extension == '.stl':
        writer = vtk.vtkSTLWriter()

    else:
        sys.exit('Unrecognized file format.')

    writer.SetInputData(surface)
    writer.SetFileName(file_name)
    writer.Update()
    writer.Write()

def WriteSpline(points, tangents, file_name):
    """Write spline from points.

    Given a set of points and its tangents at each point, writes to VTP file
    the spline formed by the points set.
    """

    # Write spline to vtp file
    data = vtk.vtkPoints()
    for point in points:
        data.InsertNextPoint(point)

    spline = vtk.vtkPolyData()
    spline.SetPoints(data)

    pointDataArray = vtk.vtkFloatArray()
    pointDataArray.SetNumberOfComponents(3)

    pointDataArray.SetName('Tangents')
    for pointData in tangents:
        pointDataArray.InsertNextTuple(pointData)

    spline.GetPointData().SetActiveVectors('Tangents')
    spline.GetPointData().SetVectors(pointDataArray)

    WriteSurface(spline, file_name)

def SmoothSurface(surface):
    """Smooth surface based on Taubin's algorithm."""
    smoother = vmtkscripts.vmtkSurfaceSmoothing()
    smoother.Surface = surface
    smoother.Method  = 'taubin'
    smoother.NumberOfIterations = int(const.three*const.ten)
    smoother.PassBand = const.one/const.ten
    smoother.Execute()

    return smoother.Surface

def ScaleSurface(surface, scale_factor):
    # Scale surface
    transform = vtk.vtkTransform()
    transform.Scale(3*(scale_factor,))

    transformFilter = vtk.vtkTransformPolyDataFilter()
    transformFilter.SetInputData(surface)
    transformFilter.SetTransform(transform)
    transformFilter.Update()

    return transformFilter.GetOutput()

def Cleaner(surface):
    """Polydata cleaner."""
    cleaner = vtk.vtkCleanPolyData()
    cleaner.SetInputData(surface)
    cleaner.Update()

    return cleaner.GetOutput()

def GetCellArrays(polydata):
    """Return the names and number of arrays for a vtkPolyData."""

    nCellArrays = polydata.GetCellData().GetNumberOfArrays()

    return [polydata.GetCellData().GetArray(id_).GetName()
            for id_ in range(nCellArrays)]

def GetPointArrays(polydata):
    """Return the names of point arrays for a vtkPolyData."""

    nPointArrays = polydata.GetPointData().GetNumberOfArrays()

    return [polydata.GetPointData().GetArray(id_).GetName()
            for id_ in range(nPointArrays)]

def CleanupArrays(surface):
    """Remove any point and/or cell array in a vtkPolyData."""

    for p_array in GetPointArrays(surface):
        surface.GetPointData().RemoveArray(p_array)

    for c_array in GetCellArrays(surface):
        surface.GetCellData().RemoveArray(c_array)

    return surface


def ExtractPortion(polydata, array_name, isovalue):
    """Extract portion of vtkPolyData based on array."""

    threshold = vtk.vtkThreshold()
    threshold.SetInputData(polydata)
    threshold.SetInputArrayToProcess(0, 0, 0, 1, array_name)
    threshold.ThresholdBetween(isovalue, isovalue)
    threshold.Update()

    # Converts vtkUnstructuredGrid -> vtkPolyData
    gridToSurfaceFilter = vtk.vtkGeometryFilter()
    gridToSurfaceFilter.SetInputData(threshold.GetOutput())
    gridToSurfaceFilter.Update()

    return gridToSurfaceFilter.GetOutput()

def ExtractConnectedRegion(regions, method, closest_point=None):
    """Extract the largest or closest to point patch of a disconnected domain.

    Given a disconnected surface, extract a portion of the surface by choosing
    the largest or closest to point patch.
    """

    triangulator = vtk.vtkTriangleFilter()
    triangulator.SetInputData(Cleaner(regions))
    triangulator.Update()

    connectivity = vtk.vtkPolyDataConnectivityFilter()
    connectivity.SetInputData(triangulator.GetOutput())

    if method == 'largest':
        connectivity.SetExtractionModeToLargestRegion()

    elif method == 'closest':
        connectivity.SetExtractionModeToClosestPointRegion()
        connectivity.SetClosestPoint(closest_point)

    connectivity.Update()

    return connectivity.GetOutput()

def ClipWithScalar(surface: names.polyDataType,
                   array_name: str,
                   value: float,
                   inside_out=True) -> names.polyDataType:
    """ Clip surface with scalar field.

    Provided a surface (vtkPolyData), a point scalar array and a 'value' of
    this array, clip the surface portion that have the condition 'scalar_array
    > value'. If inside out is 'False', than the oposite will be output.
    """
    # Get point data and cell data
    pointArrays = GetPointArrays(surface)
    cellArrays = GetCellArrays(surface)

    # TODO: Cannot use a try-statement here because the vtkClipPolyData filter
    # does not throw any exception if error occurs (investigate why, I think I
    # have to activate like an 'error' handler in the filter).

    if array_name not in pointArrays and array_name in cellArrays:
        # Convert cell to point
        pointdata = vtk.vtkCellDataToPointData()
        pointdata.SetInputData(surface)
        pointdata.Update()

        surface = pointdata.GetOutput()

    elif array_name not in pointArrays and array_name not in cellArrays:
        errorMsg = 'I cannot find ' + array_name + 'on the surface.'
        print(errorMsg, end='\n')

    else:
        pass

    # Change active array
    surface.GetPointData().SetActiveScalars(array_name)

    # Clip the aneurysm surface in the lowWSSValue ang gets portion smaller
    # than it
    clipper = vtk.vtkClipPolyData()
    clipper.SetInputData(surface)
    clipper.SetValue(value)

    if inside_out:
        clipper.SetInsideOut(1)
    else:
        clipper.SetInsideOut(0)

    clipper.Update()

    return clipper.GetOutput()

def ClipWithPlane(surface, plane_center, plane_normal, inside_out=False):
    """ Clip a surface with a plane defined with a point and its normal."""

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

def ComputeSurfacesDistance(isurface,
                            rsurface,
                            array_name='DistanceArray',
                            signed_array=True):
    """Compute point-wise distance between two surfaces.

    Compute distance between a reference surface, rsurface, and an input
    surface, isurface, with the resulting array written in the isurface.
    """

    if signed_array:
        normalsFilter = vtk.vtkPolyDataNormals()
        normalsFilter.SetInputData(rsurface)
        normalsFilter.AutoOrientNormalsOn()
        normalsFilter.SetFlipNormals(False)
        normalsFilter.Update()
        rsurface.GetPointData().SetNormals(
            normalsFilter.GetOutput().GetPointData().GetNormals()
        )

    surfaceDistance = vtkvmtk.vtkvmtkSurfaceDistance()
    surfaceDistance.SetInputData(isurface)
    surfaceDistance.SetReferenceSurface(rsurface)

    if signed_array:
        surfaceDistance.SetSignedDistanceArrayName(array_name)
    else:
        surfaceDistance.SetDistanceArrayName(array_name)

    surfaceDistance.Update()

    return surfaceDistance.GetOutput()

def vtkPolyDataToDataFrame(polydata: names.polyDataType) -> pd.core.frame.DataFrame:
    """Convert a vtkPolyData with cell arrays to Pandas DataFrame.

    Given a vtkPolyData object containing cell arrays of any kind (scalars,
    vectors, etc), convert it to a Pandas DataFrame structure after converting
    the cell centers, where the cell data are defined, to points.

    Returns a Panda's DataFrame objects with the columns the cell centers and
    the fields (separated by component, if necessary). The index column is the
    cell index id.
    """
    # Check if polydata has any point arrays and interpolate them to cells
    pointArrays = GetPointArrays(polydata)

    if pointArrays:
        pointToCellData = vtk.vtkPointDataToCellData()
        pointToCellData.SetInputData(polydata)
        pointToCellData.Update()

        polydata = pointToCellData.GetOutput()

    # Convert cell centers to points
    cellCenters = vtk.vtkCellCenters()
    cellCenters.VertexCellsOff()
    cellCenters.SetInputData(polydata)
    cellCenters.Update()

    npPolyData = dsa.WrapDataObject(cellCenters.GetOutput())

    cellCenterArrayName = "CellCenter"

    # Fields components suffixes
    threeDimvectorCompSuffixes = [names.xAxisSufx, 
                                  names.yAxisSufx, 
                                  names.zAxisSufx]

    symmTensorCompSuffix = [2*names.xAxisSufx, 
                            2*names.yAxisSufx, 
                            2*names.zAxisSufx, 
                            names.xAxisSufx + names.yAxisSufx,
                            names.yAxisSufx + names.zAxisSufx,
                            names.xAxisSufx + names.zAxisSufx]

    twoDimVectorCompSuffixes = [names.xAxisSufx, 
                                names.yAxisSufx]

    pointsToDataFrame = pd.DataFrame(npPolyData.GetPoints(),
                                     columns=["_".join([cellCenterArrayName,sfx])
                                              for sfx in threeDimvectorCompSuffixes])

    pointData = npPolyData.GetPointData()

    arrayNames = [pointData.GetArray(index).GetName()
                  for index in range(pointData.GetNumberOfArrays())]

    arraysOnTheSurface = []

    for arrayName in arrayNames:
        # Get array dimension to define columns names
        fieldArray = pointData.GetArray(arrayName)
        nComponents = fieldArray.shape[-1]
        numpyArrayDimension = fieldArray.ndim

        # Not scalar and 3D vector arrays
        if  numpyArrayDimension == 2 and nComponents == 3:
            colNames = ["_".join([arrayName, sfx])
                        for sfx in threeDimvectorCompSuffixes]

        elif numpyArrayDimension == 2 and nComponents == 2:
            colNames = ["_".join([arrayName, sfx])
                        for sfx in twoDimVectorCompSuffixes]

        elif numpyArrayDimension == 2 and nComponents == 6:
            colNames = ["_".join([arrayName, sfx])
                        for sfx in symmTensorCompSuffix]

        elif numpyArrayDimension == 1:
            colNames = [arrayName]

        else:
            errMessage = "There is something wrong!" \
                         "I got an array with dimension > 2 on a surface!"
            sys.exit(errMessage)

        arraysOnTheSurface.append(
            pd.DataFrame(
                pointData.GetArray(arrayName),
                columns=colNames
            )
        )

    return pd.concat([pointsToDataFrame] + arraysOnTheSurface,
                     axis=1)

# This class was adapted from the 'vmtkcenterlines.py' script
# distributed with VMTK in https://github.com/vmtk/vmtk
class PickPointSeedSelector():

    def __init__(self):
        self._Surface = None
        self._SeedIds = None
        self._SourceSeedIds = vtk.vtkIdList()
        self._TargetSeedIds = vtk.vtkIdList()
        self._InputInfo = None

        self.PickedSeedIds = vtk.vtkIdList()
        self.PickedSeeds = vtk.vtkPolyData()
        self.vmtkRenderer = None
        self.OwnRenderer = 0

    def SetSurface(self,surface):
        self._Surface = surface

    def GetSurface(self):
        return self._Surface

    def InputInfo(self, message):
        self._InputInfo = message

    def UndoCallback(self, obj):
        self.InitializeSeeds()
        self.PickedSeeds.Modified()
        self.vmtkRenderer.RenderWindow.Render()

    def PickCallback(self, obj):
        picker = vtk.vtkCellPicker()
        picker.SetTolerance(1e-4*self._Surface.GetLength())
        eventPosition = self.vmtkRenderer.RenderWindowInteractor.GetEventPosition()

        result = picker.Pick(float(eventPosition[0]),
                             float(eventPosition[1]),
                             0.0,
                             self.vmtkRenderer.Renderer)

        if result == 0:
            return

        pickPosition = picker.GetPickPosition()
        pickedCellPointIds = self._Surface.GetCell(picker.GetCellId()).GetPointIds()
        minDistance = 1e10
        pickedSeedId = -1

        for i in range(pickedCellPointIds.GetNumberOfIds()):
            distance = vtk.vtkMath.Distance2BetweenPoints(
                            pickPosition,
                            self._Surface.GetPoint(pickedCellPointIds.GetId(i))
                        )

            if distance < minDistance:
                minDistance  = distance
                pickedSeedId = pickedCellPointIds.GetId(i)

        if pickedSeedId == -1:
            pickedSeedId = pickedCellPointIds.GetId(0)

        self.PickedSeedIds.InsertNextId(pickedSeedId)
        point = self._Surface.GetPoint(pickedSeedId)
        self.PickedSeeds.GetPoints().InsertNextPoint(point)
        self.PickedSeeds.Modified()
        self.vmtkRenderer.RenderWindow.Render()

    def InitializeSeeds(self):
        self.PickedSeedIds.Initialize()
        self.PickedSeeds.Initialize()
        seedPoints = vtk.vtkPoints()
        self.PickedSeeds.SetPoints(seedPoints)

    def Execute(self):

        if (self._Surface == None):
            self.PrintError('vmtkPickPointSeedSelector Error: Surface not set.')
            return

        self._SourceSeedIds.Initialize()
        self._TargetSeedIds.Initialize()

        if not self.vmtkRenderer:
            self.vmtkRenderer = vmtkrenderer.vmtkRenderer()
            self.vmtkRenderer.Initialize()
            self.OwnRenderer = 1

        glyphs = vtk.vtkGlyph3D()
        glyphSource = vtk.vtkSphereSource()
        glyphs.SetInputData(self.PickedSeeds)
        glyphs.SetSourceConnection(glyphSource.GetOutputPort())
        glyphs.SetScaleModeToDataScalingOff()
        glyphs.SetScaleFactor(self._Surface.GetLength()*0.01)
        glyphMapper = vtk.vtkPolyDataMapper()
        glyphMapper.SetInputConnection(glyphs.GetOutputPort())

        self.SeedActor = vtk.vtkActor()
        self.SeedActor.SetMapper(glyphMapper)
        self.SeedActor.GetProperty().SetColor(1.0,0.0,0.0)
        self.SeedActor.PickableOff()
        self.vmtkRenderer.Renderer.AddActor(self.SeedActor)

        self.vmtkRenderer.AddKeyBinding('u','Undo',
                                        self.UndoCallback)

        self.vmtkRenderer.AddKeyBinding('space','Add points',
                                        self.PickCallback)

        surfaceMapper = vtk.vtkPolyDataMapper()
        surfaceMapper.SetInputData(self._Surface)
        surfaceMapper.ScalarVisibilityOff()

        surfaceActor = vtk.vtkActor()
        surfaceActor.SetMapper(surfaceMapper)
        surfaceActor.GetProperty().SetOpacity(1.0)

        self.vmtkRenderer.Renderer.AddActor(surfaceActor)

        if self._InputInfo is not None:
            self.vmtkRenderer.InputInfo(self._InputInfo)

        any_ = 0
        while any_ == 0:
            self.InitializeSeeds()
            self.vmtkRenderer.Render()
            any_ = self.PickedSeedIds.GetNumberOfIds()

        self._SourceSeedIds.DeepCopy(self.PickedSeedIds)

        if self.OwnRenderer:
            self.vmtkRenderer.Deallocate()

def SelectSurfacePoint(surface):
    """Enable selection of aneurysm point."""

    # Select aneurysm tip point
    pickPoint = PickPointSeedSelector()
    pickPoint.SetSurface(surface)
    pickPoint.InputInfo("Select point on the aneurysm surface\n")
    pickPoint.Execute()

    return pickPoint.PickedSeeds.GetPoint(0)
