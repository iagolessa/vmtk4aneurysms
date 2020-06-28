"""Provide functions to work with VTK poly data."""

import vtk
from vmtk import vtkvmtk
from vmtk import vmtkscripts
from vmtk import vmtkrenderer

from constants import *

def readSurface(file_name):
    """Read surface file to VTK object.
               
    Arguments: 
    file_name -- complete path with file name
    """
    reader = vmtkscripts.vmtkSurfaceReader()
    reader.InputFileName = file_name
    reader.Execute()
    
    return reader.Surface


def viewSurface(surface, array_name=None):
    """View surface vtkPolyData objects.
    
    Arguments:
    surface -- the surface to be displayed.
    """
    viewer = vmtkscripts.vmtkSurfaceViewer()
    viewer.Surface = surface
    
    if array_name != None:
        viewer.ArrayName = array_name
        viewer.Legend = True
    
    viewer.Execute()
    
def writeSurface(surface, file_name, mode='binary'):
    """Write surface vtkPolyData. 
        
    Arguments:
    surface -- surface vtkPolyData object
    file_name -- output file name with full path
    
    Optional arguments:
    mode -- mode to write file (ASCII or binary, default binary)
    """
    writer = vmtkscripts.vmtkSurfaceWriter()
    writer.Surface = surface
    writer.Mode = mode
    writer.OutputFileName = file_name
    writer.Execute()

#     writer = vtk.vtkPolyDataWriter()
#     writer.SetFileName(file_name)
#     writer.SetInputData(surface)
#
#     if mode == 'binary':
#         writer.SetFileTypeToBinary()
#     elif mode == 'ascii':
#         writer.SetFileTypeToASCII()
#
#     writer.Write()

def writePolyData(polydata, file_name):
    """Write surface vtkPolyData."""

    writer = vtk.vtkXMLPolyDataWriter()
    writer.SetFileName(file_name)
    writer.SetInputData(polydata)
    writer.Write()

def writePoints(points, file_name):
    """Write vtkPoints to file."""

    pointSet  = vtk.vtkPolyData()
    cellArray = vtk.vtkCellArray()

    for i in range(points.GetNumberOfPoints()):
      cellArray.InsertNextCell(1)
      cellArray.InsertCellPoint(i)

    pointSet.SetPoints(points)
    pointSet.SetVerts(cellArray)

    writePolyData(pointSet, file_name)   


def smoothSurface(surface):
    """Smooth surface based on Taubin's algorithm."""
    smoother = vmtkscripts.vmtkSurfaceSmoothing()
    smoother.Surface = surface
    smoother.Method  = 'taubin'
    smoother.NumberOfIterations = intThree*intTen
    smoother.PassBand = intOne/intTen
    smoother.Execute()
    
    return smoother.Surface

def cleaner(surface):
    """Polydata cleaner."""
    cleaner = vtk.vtkCleanPolyData()
    cleaner.SetInputData(surface)
    cleaner.Update()
    
    return cleaner.GetOutput()

def getCellArrays(polydata):
    """Return the names and number of arrays for a vtkPolyData."""
    
    nCellArrays = polydata.GetCellData().GetNumberOfArrays()
    names = list()
    
    for id_ in range(nCellArrays):
        names.append(polydata.GetCellData().GetArray(id_).GetName())
        
    return names

def getPointArrays(polydata):
    """Return the names of point arrays for a vtkPolyData."""
    
    nPointArrays = polydata.GetPointData().GetNumberOfArrays()
    names = list()
    
    for id_ in range(nPointArrays):
        names.append(polydata.GetPointData().GetArray(id_).GetName())
        
    return names

def extractPortion(polydata, array_name, isovalue):
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

def extractConnectedRegion(regions, method, closest_point=None):
    """Extract the largest or closest to point patch of a disconnected domain. 

    Given a disconnected surface, extract a portion of the surface
    by choosing the largest or closest to point patch.
    """

    triangulator = vtk.vtkTriangleFilter()
    triangulator.SetInputData(cleaner(regions))
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

def clipWithPlane(surface, plane_center, plane_normal, inside_out=False):
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

def computeSurfacesDistance(isurface,
                            rsurface,
                            array_name='DistanceArray',
                            signed_array=True):
    """Compute point-wise distance between two surfaces.

    Compute distance between a reference
    surface, rsurface, and an input surface, isurface, with 
    the resulting array written in the isurface. 
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

def selectSurfacePoint(surface):
    """Enable selection of aneurysm point."""

    # Select aneurysm tip point
    pickPoint = PickPointSeedSelector()
    pickPoint.SetSurface(surface)
    pickPoint.InputInfo("Select point on the aneurysm surface\n")
    pickPoint.Execute()

    return pickPoint.PickedSeeds.GetPoint(0)


# Other stuff
radiusArrayName = 'Abscissas'

def ExtractSingleLine(centerlines, id_):
    cell = vtk.vtkGenericCell()
    centerlines.GetCell(id_, cell)

    line = vtk.vtkPolyData()
    points = vtk.vtkPoints()
    cellArray = vtk.vtkCellArray()
    cellArray.InsertNextCell(cell.GetNumberOfPoints())

    radiusArray = vtk.vtkDoubleArray()
    radiusArray.SetName(radiusArrayName)
    radiusArray.SetNumberOfComponents(1)
    radiusArray.SetNumberOfTuples(cell.GetNumberOfPoints())
    radiusArray.FillComponent(0,0.0)

    for i in range(cell.GetNumberOfPoints()):
        point = [0.0,0.0,0.0]
        point = cell.GetPoints().GetPoint(i)

        points.InsertNextPoint(point)
        cellArray.InsertCellPoint(i)
        radius = centerlines.GetPointData().GetArray(radiusArrayName).GetTuple1(cell.GetPointId(i))
        radiusArray.SetTuple1(i,radius)

    line.SetPoints(points)
    line.SetLines(cellArray)
    line.GetPointData().AddArray(radiusArray)

    return line

def extract_branch(centerlines, cell_id, start_point=None, end_point=None):
    """Extract one line from multiple centerlines.
    If start_id and end_id is set then only a segment of the centerline is extracted.
    Args:
        centerlines (vtkPolyData): Centerline to extract.
        line_id (int): The line ID to extract.
        start_id (int):
        end_id (int):
    Returns:
        centerline (vtkPolyData): The single line extracted
    """
    cell = ExtractSingleLine(centerlines, cell_id)

    start_id = _id_min_dist_to_point(start_point, cell)
    end_id   = _id_min_dist_to_point(end_point, cell)

    print(start_id, end_id)
#     n = cell.GetNumberOfPoints() if end_id is None else end_id + 1

    line = vtk.vtkPolyData()
    cell_array = vtk.vtkCellArray()
    cell_array.InsertNextCell(abs(end_id - start_id))
    line_points = vtk.vtkPoints()

#     arrays = []
#     n_, names = get_number_of_arrays(centerlines)

#     for i in range(n_):
#         tmp = centerlines.GetPointData().GetArray(names[i])
#         tmp_comp = tmp.GetNumberOfComponents()
#         radius_array = get_vtk_array(names[i], tmp_comp, n - start_id)
#         arrays.append(radius_array)

#     point_array = []
#     for i in range(n_):
#         point_array.append(centerlines.GetPointData().GetArray(names[i]))

    count = 0

    # Select appropriante range of ids
    # Important due to inverse numberring of
    # ids that VMTK generate in centerlines
    if start_id < end_id:
        ids = range(start_id, end_id)
    else:
        ids = range(start_id, end_id, -1)


    for i in ids:
        cell_array.InsertCellPoint(count)
        line_points.InsertNextPoint(cell.GetPoints().GetPoint(i))

#         for j in range(n_):
#             num = point_array[j].GetNumberOfComponents()
#             if num == 1:
#                 tmp = point_array[j].GetTuple1(i)
#                 arrays[j].SetTuple1(count, tmp)
#             elif num == 2:
#                 tmp = point_array[j].GetTuple2(i)
#                 arrays[j].SetTuple2(count, tmp[0], tmp[1])
#             elif num == 3:
#                 tmp = point_array[j].GetTuple3(i)
#                 arrays[j].SetTuple3(count, tmp[0], tmp[1], tmp[2])
#             elif num == 9:
#                 tmp = point_array[j].GetTuple9(i)
#                 arrays[j].SetTuple9(count, tmp[0], tmp[1], tmp[2], tmp[3], tmp[4],
#                                     tmp[5], tmp[6], tmp[7], tmp[8])
        count += 1

    line.SetPoints(line_points)
    line.SetLines(cell_array)
#     for j in range(n_):
#         line.GetPointData().AddArray(arrays[j])

    return line
