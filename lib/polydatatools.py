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

"""Collection of tools to manipulate and operate on VTK objetcs."""

import os
import sys
import pandas as pd
from typing import Union
from copy import copy
from numpy import ndarray

import vtk
from vmtk import vtkvmtk
from vmtk import vmtkscripts
from vmtk import vmtkrenderer
from vtk.numpy_interface import dataset_adapter as dsa

from . import constants as const
from . import names

def UnsGridToPolyData(
        mesh: names.unstructuredGridType
    )   -> names.polyDataType:
    """Convert a vtkUnstructuredGrid to vtkPolyData."""

    meshToSurface = vtk.vtkGeometryFilter()
    meshToSurface.SetInputData(mesh)
    meshToSurface.Update()

    return meshToSurface.GetOutput()

def PolyDataToUnsGrid(
        surface: names.polyDataType
    )   -> names.unstructuredGridType:
    """Convert a vtkPolyData to a vtkUnstructuredGrid."""

    surfaceToMesh = vtkvmtk.vtkvmtkPolyDataToUnstructuredGridFilter()
    surfaceToMesh.SetInputData(surface)
    surfaceToMesh.Update()

    return surfaceToMesh.GetOutput()

def ScaleVtkObject(
        vtk_object: Union[names.polyDataType, names.unstructuredGridType],
        scale_factor: float
    )   -> Union[names.polyDataType, names.unstructuredGridType]:
    """Scale a VTK object by a scale factor."""

    # Scaling transform
    scaling = vtk.vtkTransform()
    scaling.Scale(3*(scale_factor,))
    scaling.Update()

    if type(vtk_object) == names.polyDataType:
        transformFilter = vtk.vtkTransformPolyDataFilter()
        transformFilter.SetInputData(vtk_object)
        transformFilter.SetTransform(scaling)
        transformFilter.Update()

    else:
        transformFilter = vtk.vtkTransformFilter()
        transformFilter.SetInputData(vtk_object)
        transformFilter.SetTransform(scaling)
        transformFilter.Update()

    return transformFilter.GetOutput()

def CopyVtkObject(
        vtk_object: Union[names.polyDataType, names.unstructuredGridType]
    )   -> Union[names.polyDataType, names.unstructuredGridType]:
    """Returns a copy of the passed vtkPolyData or vtkUnstructuredGrid."""

    return ScaleVtkObject(vtk_object, 1.0)

def ReadSurface(file_name: str) -> names.polyDataType:
    """Read surface file name to VTK object.

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

def ReadUnsGrid(
        mesh_file_name: str
    )   -> names.unstructuredGridType:
    """Read mesh file name to VTK object."""

    readMesh = vmtkscripts.vmtkMeshReader()
    readMesh.InputFileName = mesh_file_name
    readMesh.Execute()

    return readMesh.Mesh

def ViewVtkObject(
        vtk_object: Union[names.polyDataType, names.unstructuredGridType],
        field_name: str=None
    ):
    """View VTK objects (poly data or unstructured grid)."""

    # Remove the one from the list
    if field_name is not None and field_name not in GetPointArrays(vtk_object):

        if field_name in GetCellArrays(vtk_object):

            cellToPointFilter = vtk.vtkCellDataToPointData()
            cellToPointFilter.SetInputData(vtk_object)
            cellToPointFilter.PassCellDataOff()
            cellToPointFilter.Update()

            vtk_object = cellToPointFilter.GetOutput()

        else:
            raise ValueError("{} not in input VTK object.".format(field_name))

    if type(vtk_object) == names.polyDataType:

        viewer = vmtkscripts.vmtkSurfaceViewer()
        viewer.Surface = vtk_object

        if field_name != None:
            viewer.ArrayName = field_name
            viewer.DisplayCellData = 0
            viewer.Legend = True

        viewer.Execute()

    elif type(vtk_object) == names.unstructuredGridType:

        viewer = vmtkscripts.vmtkMeshViewer()
        viewer.Mesh = vtk_object

        if field_name != None:
            viewer.ArrayName = field_name
            viewer.Legend = True

        viewer.Execute()

    else:
        raise ValueError("Unknown format.")

def ViewSurface(
        surface: names.polyDataType,
        array_name: str=None
    ):
    """View surface vtkPolyData objects.

    Arguments:
    surface -- the surface to be displayed.
    """

    ViewVtkObject(surface, array_name)

def WriteSurface(surface: names.polyDataType,
                 file_name: str) -> None:
    """Write vtkPolyData to file.

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

def WriteUnsGrid(
        grid: names.unstructuredGridType,
        file_name: str
    )   -> None:
    """Write a vtkUnstructuredGrid to .vtk file."""

    writeGrid = vtk.vtkUnstructuredGridWriter()
    writeGrid.SetInputData(grid)
    writeGrid.SetFileTypeToBinary()
    writeGrid.SetFileName(file_name)
    writeGrid.Write()

def WriteSpline(points, tangents, file_name):
    """Write spline from a set of points and tangents.

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

def BuildPolyDataPoints(
        point_coords: Union[ndarray, list],
        point_fields: dict
    )   -> names.polyDataType:
    """Build VTK Polydata composed of points and fields.

    Pass the points coordinates as a list or numpy array and
    a dicttionary with its keys the field names and values
    the field arrays.
    """

    points = vtk.vtkPoints()

    for point in point_coords:
        points.InsertNextPoint(point)

    pointsData = vtk.vtkPolyData()
    pointsData.SetPoints(points)

    # Add point fields to polydata
    npPointsData = dsa.WrapDataObject(pointsData)

    for pfield_name, pfield in point_fields.items():
        npPointsData.PointData.append(
            pfield,
            pfield_name
        )

    return npPointsData.VTKObject

def SmoothSurface(
        surface: names.polyDataType,
        niterations: int=30,
        passband: float=0.1
    )   -> names.polyDataType:
    """Smooth surface based on Taubin's algorithm."""

    smoothingFilter = vtk.vtkWindowedSincPolyDataFilter()

    smoothingFilter.SetInputData(surface)
    smoothingFilter.SetNumberOfIterations(niterations)
    smoothingFilter.SetPassBand(passband)
    smoothingFilter.BoundarySmoothingOff()
    smoothingFilter.NormalizeCoordinatesOn()
    smoothingFilter.Update()

    return smoothingFilter.GetOutput()

def Cleaner(surface):
    """Polydata cleaner."""
    cleaner = vtk.vtkCleanPolyData()
    cleaner.SetInputData(surface)
    cleaner.Update()

    return cleaner.GetOutput()

def RemeshSurface(
        surface: names.polyDataType,
        target_cell_area: float=0.01
    )   -> names.polyDataType:
    """Remesh surface using VMTK and tageting the cell area size."""

    remesher = vmtkscripts.vmtkSurfaceRemeshing()
    remesher.Surface = surface
    remesher.ElementSizeMode = 'area'
    remesher.TargetArea = target_cell_area
    remesher.PreserveBoundaryEdges = 1
    remesher.Execute()

    return remesher.Surface

def GetCellArrays(
        vtk_object: Union[names.polyDataType, names.unstructuredGridType]
    )   -> list:
    """Return the names of CELL arrays in a VTK object."""

    nCellArrays = vtk_object.GetCellData().GetNumberOfArrays()

    return [vtk_object.GetCellData().GetArray(id_).GetName()
            for id_ in range(nCellArrays)]

def GetPointArrays(
        vtk_object: Union[names.polyDataType, names.unstructuredGridType]
    )   -> list:
    """Return the names of POINT arrays in a VTK object."""

    nPointArrays = vtk_object.GetPointData().GetNumberOfArrays()

    return [vtk_object.GetPointData().GetArray(id_).GetName()
            for id_ in range(nPointArrays)]

def _project_point_arrays_mesh_to_mesh(
        mesh: names.unstructuredGridType,
        ref_mesh: names.unstructuredGridType
    )   -> names.unstructuredGridType:

    projection = vtkvmtk.vtkvmtkMeshProjection()
    projection.SetInputData(mesh)
    projection.SetReferenceMesh(ref_mesh)
    projection.Update()

    return projection.GetOutput()

def _project_point_arrays_surface_to_surface(
        surface: names.polyDataType,
        ref_surface: names.polyDataType
    )   -> names.polyDataType:

    surfaceProjection = vtkvmtk.vtkvmtkSurfaceProjection()
    surfaceProjection.SetInputData(surface)
    surfaceProjection.SetReferenceSurface(ref_surface)
    surfaceProjection.Update()

    return surfaceProjection.GetOutput()

def ProjectPointArray(
        vtk_object: Union[names.polyDataType, names.unstructuredGridType],
        ref_vtk_object: Union[names.polyDataType, names.unstructuredGridType],
        field_name: str
    )   -> Union[names.polyDataType, names.unstructuredGridType]:
    """Project a point field from a reference VTK object into another one.

    Given a vtkPolyData or a vtkUnstructuredGrid, project a point field named
    'field_name' from a reference VTK object (vtkPolyData or
    vtkUnstructuredGrid) to original object. Returns a copy of the original
    input vtk_object (with the same type).
    """

    # Operates on copy
    vtk_object = CopyVtkObject(vtk_object)
    ref_vtk_object = CopyVtkObject(ref_vtk_object)

    if type(vtk_object) == names.polyDataType:
        # Clean before smoothing array
        cleaner = vtk.vtkCleanPolyData()
        cleaner.SetInputData(vtk_object)
        cleaner.Update()

        vtk_object = cleaner.GetOutput()

    if type(ref_vtk_object) == names.polyDataType:
        # Clean ref. surface too
        cleaner = vtk.vtkCleanPolyData()
        cleaner.SetInputData(ref_vtk_object)
        cleaner.Update()

        ref_vtk_object = cleaner.GetOutput()

    # Remove spourious array from final surface
    cellData  = ref_vtk_object.GetCellData()
    pointData = ref_vtk_object.GetPointData()

    cellArrays  = GetCellArrays(ref_vtk_object)
    pointArrays = GetPointArrays(ref_vtk_object)

    # Remove the one from the list
    if field_name in pointArrays:
        pointArrays.remove(field_name)

    else:
        raise ValueError("{} not in input VTK object.".format(field_name))

    for point_array in pointArrays:
        pointData.RemoveArray(point_array)

    for cell_array in cellArrays:
        cellData.RemoveArray(cell_array)

    # Then project the left one to new vtk_object
    # Convert ref. surface to mesh
    if type(ref_vtk_object) == names.polyDataType:

        if type(vtk_object) == names.polyDataType:

            return _project_point_arrays_surface_to_surface(
                       vtk_object,
                       ref_vtk_object
                   )

        elif type(vtk_object) == names.unstructuredGridType:

            refSurfaceToMesh = vtkvmtk.vtkvmtkPolyDataToUnstructuredGridFilter()
            refSurfaceToMesh.SetInputData(ref_vtk_object)
            refSurfaceToMesh.Update()

            ref_vtk_object = refSurfaceToMesh.GetOutput()

            return _project_point_arrays_mesh_to_mesh(
                       vtk_object,
                       ref_vtk_object
                   )

        else:
            raise TypeError(
                    "Input object neither {} or {}".format(
                        names.polyDataType,
                        names.unstructuredGridType
                    )
                )

    elif type(ref_vtk_object) == names.unstructuredGridType:

        if type(vtk_object) == names.unstructuredGridType:

            return _project_point_arrays_mesh_to_mesh(
                       vtk_object,
                       ref_vtk_object
                   )

        elif type(vtk_object) == names.polyDataType:

            surfaceToMesh = vtkvmtk.vtkvmtkPolyDataToUnstructuredGridFilter()
            surfaceToMesh.SetInputData(vtk_object)
            surfaceToMesh.Update()

            vtk_object = surfaceToMesh.GetOutput()

            vtk_object =  _project_point_arrays_mesh_to_mesh(
                               vtk_object,
                               ref_vtk_object
                           )

            # Convert back to vtkPolyData
            return UnsGridToPolyData(vtk_object)

        else:
            raise TypeError(
                    "Input object neither {} or {}".format(
                        names.polyDataType,
                        names.unstructuredGridType
                    )
                )

    else:
        raise TypeError(
                "Input object neither {} or {}".format(
                    names.polyDataType,
                    names.unstructuredGridType
                )
            )

def ProjectCellArray(
        surface: names.polyDataType,
        ref_surface: names.polyDataType,
        field_name: str
    )   -> names.polyDataType:
    """Project a cell field from a reference VTK polydata into another one.

    Given a vtkPolyData, project a cell field named 'field_name' from a
    reference VTK surface to the original object. Returns a copy of the
    original input surface.
    """

    surface = CopyVtkObject(surface)
    surface = Cleaner(surface)

    if field_name not in GetCellArrays(ref_surface):
        raise ValueError(
                  "No field {} on the reference surface.".format(self.FieldName)
              )

    else:
        # Then project the left one to new surface
        projector = vtkvmtk.vtkvmtkSurfaceProjectCellArray()
        projector.SetInputData(surface)
        projector.SetReferenceSurface(ref_surface)
        projector.SetProjectedArrayName(field_name)
        projector.SetDefaultValue(0.0)
        projector.Update()

        return projector.GetOutput()

def ResampleFieldsToSurface(
        source_mesh: names.unstructuredGridType,
        target_surface: names.polyDataType,
        field_names: Union[str, list]="all"
    )   -> names.polyDataType:
    """Resample fields of a vtkUnstructuredGrid into a surface contained in it.

    Given a volumetric mesh (vtkUnstructuredGrid) object with cell or point
    fields defined on it, resample the passed fields to a surface (vtkPolyData)
    that is contained by the volumetric mesh. The resampling filter resamples
    all fields to point fields. If no list or string with the field names to
    resample, resamples all the fields.

    .. warning ::
        This procedure may generate erroneous results for cell fields,
        particularly. Therefore, it first converts all cell fields to point
        fields, resamples, and then convert all fields to cell fields again in
        the resulting mesh.
    """

    # It is better for this procedure to convert all cell fields to point
    # fields first
    cellToPointFilter = vtk.vtkCellDataToPointData()
    cellToPointFilter.SetInputData(source_mesh)
    cellToPointFilter.PassCellDataOff()
    cellToPointFilter.Update()

    source_mesh = cellToPointFilter.GetOutput()

    if field_names != "all":

        # Check whether only one string or a list was passed
        field_names = [field_names] \
                      if type(field_names) is str \
                      else field_names

        # Leave only field_names in the source_mesh
        pointArrays = GetPointArrays(source_mesh)

        # Check whether the fields are on the source_mesh
        fieldsInPointArrays = all(field_name in pointArrays
                                  for field_name in field_names)

        # If all the fields are on the mesh, remove them from the list
        if fieldsInPointArrays:

            for field_name in field_names:
                pointArrays.remove(field_name)

        else:
            raise ValueError(
                      "Some fields (or all) are not in input VTK object."
                  )

        # Now delete the arrays left in pointArrays
        pointData = source_mesh.GetPointData()

        for point_array in pointArrays:
            pointData.RemoveArray(point_array)

    # Finally, resample (wth only the field_names)
    resampleToMidSurface = vtk.vtkResampleWithDataSet()
    # resampleToMidSurface.SetSourceConnection(source_mesh.GetOutputPort())
    resampleToMidSurface.SetSourceData(source_mesh)
    resampleToMidSurface.SetInputData(target_surface)
    resampleToMidSurface.SetPassPointArrays(True)
    resampleToMidSurface.SetPassCellArrays(True)
    resampleToMidSurface.Update()

    # Convert all fields back to cell data in new mesh
    pointToCell = vtk.vtkPointDataToCellData()
    pointToCell.SetInputConnection(resampleToMidSurface.GetOutputPort())
    pointToCell.PassPointDataOff()
    pointToCell.Update()

    return pointToCell.GetOutput()

def CleanupArrays(
        vtk_object: Union[names.polyDataType, names.unstructuredGridType]
    )   -> Union[names.polyDataType, names.unstructuredGridType]:
    """Remove all point and/or cell arrays in a VTK object."""

    for p_array in GetPointArrays(vtk_object):
        vtk_object.GetPointData().RemoveArray(p_array)

    for c_array in GetCellArrays(vtk_object):
        vtk_object.GetCellData().RemoveArray(c_array)

    return vtk_object


def ExtractPortion(
        vtk_object: Union[names.polyDataType, names.unstructuredGridType],
        array_name: str,
        isovalue: float
    )   -> names.polyDataType:
    """Extract portion of vtkPolyData based on array value.

    Given a VTK poly data or unstructured grid, extract the portion of it
    marked by the value 'iso_value' of the field 'array_name'. Note that it
    extracts exactly the regions with this value. To extract a region marked by
    values above or below an isovalue, use ClipWithScalar.

    .. warning::
        array_name must be a cell-wise field defined on the surface.

    .. warning::
        Retun 'None' if the result is empty.
    """

    if array_name not in GetCellArrays(vtk_object) and \
       array_name not in GetPointArrays(vtk_object):
        raise ValueError("{} not in the object".format(array_name))

    if array_name in GetPointArrays(vtk_object):
        fieldAssociation = "pointField"

    elif array_name in GetCellArrays(vtk_object):
        fieldAssociation = "cellField"

    else:
        sys.exit(
            "Can't find array {} in surface".format(array_name)
        )

    threshold = vtk.vtkThreshold()
    threshold.SetInputData(vtk_object)

    # 1 indicates cell values
    threshold.SetInputArrayToProcess(
        0, 0, 0,
        0 if fieldAssociation == "pointField" else 1,
        array_name
    )

    threshold.ThresholdBetween(isovalue, isovalue)
    threshold.Update()

    # Converts vtkUnstructuredGrid -> vtkPolyData, if needed
    # (vtkThreshold return an unstructured grid, by defautl)
    if type(vtk_object) == names.polyDataType:
         portion = UnsGridToPolyData(threshold.GetOutput())

    else:
         portion = threshold.GetOutput()

    # Check if there is at least one cell
    if portion.GetNumberOfCells() == 0:
        return None

    else:
        return portion

def ExtractConnectedRegion(
        regions: names.polyDataType,
        method: str,
        closest_point: tuple=None
    )   -> names.polyDataType:
    """Extract the largest or closest to point patch of a disconnected domain.

    Given a disconnected surface, extract a portion of the surface by choosing
    the largest patch or the one closest to a point.
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

def ClipWithScalar(
        vtk_object: Union[names.polyDataType, names.unstructuredGridType],
        array_name: str,
        value: float,
        inside_out=True
    )   -> Union[names.polyDataType, names.unstructuredGridType]:
    """Clip a vtkPolyData or vtkUnstructuredGrid with scalar field.

    Provided a vtk object of type vtkPolyData or vtkUnstructuredGrid, a point
    scalar array and a 'value' of this array, clip the object portion that
    have the condition 'scalar_array < value'. If inside out is 'False', than
    the oposite will be output.
    """
    # Get point data and cell data
    pointArrays = GetPointArrays(vtk_object)
    cellArrays  = GetCellArrays(vtk_object)

    # TODO: Cannot use a try-statement here because the vtkClipPolyData filter
    # does not throw any exception if error occurs (investigate why, I think I
    # have to activate like an 'error' handler in the filter).

    if array_name not in pointArrays and array_name in cellArrays:

        # Convert cell to point
        vtk_object = CellFieldToPointField(vtk_object, array_name)

    elif array_name not in pointArrays and array_name not in cellArrays:
        raise ValueError("Cannot find " + array_name + "on the object.")

    else:
        pass

    # Change active array
    vtk_object.GetPointData().SetActiveScalars(array_name)

    # Clip the object ang gets portion smaller
    # than it (ClipDataSet returns a unstructured grid)
    clipper = vtk.vtkClipDataSet()
    clipper.SetInputData(vtk_object)
    clipper.SetValue(value)

    if inside_out:
        # From VTK, this will get the portion with array < value
        clipper.SetInsideOut(1)
    else:
        # From VTK, this will get the portion with array > value
        clipper.SetInsideOut(0)

    clipper.Update()

    # Convert output to vtkPolyData if input was vtkPolyData
    if type(vtk_object) == names.polyDataType:
        return UnsGridToPolyData(clipper.GetOutput())

    else:
        return clipper.GetOutput()

def ClipWithPlane(
        surface: names.polyDataType,
        plane_center: tuple,
        plane_normal: tuple,
        inside_out: bool = False
    )   -> names.polyDataType:
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

def ContourCutWithPlane(
        surface: names.polyDataType,
        plane_center: tuple,
        plane_normal: tuple
    )   -> names.polyDataType:
    """Cuts a surface with a plane, returning the cut contour."""

    plane = vtk.vtkPlane()
    plane.SetOrigin(plane_center)
    plane.SetNormal(plane_normal)

    # Cut initial aneurysm surface with create plane
    cutWithPlane = vtk.vtkCutter()
    cutWithPlane.SetInputData(surface)
    cutWithPlane.SetCutFunction(plane)
    cutWithPlane.Update()

    return cutWithPlane.GetOutput()

def ComputeSurfacesDistance(
        isurface: names.polyDataType,
        rsurface: names.polyDataType,
        array_name: str='DistanceArray',
        signed_array: bool=True
    )   -> names.polyDataType:
    """Compute point-wise distance between two surfaces.

    Compute distance between a reference surface, rsurface, and an input
    surface, isurface. Return a copy of isurface with the distance array
    defined on it.
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

def vtkPolyDataToDataFrame(
        polydata: names.polyDataType
    )   -> pd.core.frame.DataFrame:
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

def CellFieldToPointField(
        vtk_object: Union[names.polyDataType, names.unstructuredGridType],
        cell_field_name: str=None
    )   -> Union[names.polyDataType, names.unstructuredGridType]:
    """Convert a single cell field to a point field defined on a VTK object.

    Given a VTK poly data or an unstructured grid, convert the
    passed cell field name to a point field. If the field_name is
    None, convert all cell fields to point fields.

    .. note::
        By default, it leaves the cell field on the object.
    """
    cellFields = GetCellArrays(vtk_object)

    if cell_field_name is not None:
        if cell_field_name not in cellFields:
            raise ValueError(
                      cell_field_name + " not found in object. Aborting."
                  )

        # To allow the behavior of a single field being interpolated,
        # the fields that will be kept as cells fields only must be
        # deleted after the operation, so keep a list with them
        cellFieldsToKeep = copy(cellFields)
        cellFieldsToKeep.remove(cell_field_name)

    cellToPoint = vtk.vtkCellDataToPointData()
    cellToPoint.SetInputData(vtk_object)
    cellToPoint.SetPassCellData(True)
    cellToPoint.Update()

    vtk_object = cellToPoint.GetOutput()

    if cell_field_name is not None:
        # Now delete all the point fields in cellFieldsToKeep
        for pfield in cellFieldsToKeep:
            vtk_object.GetPointData().RemoveArray(pfield)

    return vtk_object

def PointFieldToCellField(
        vtk_object: Union[names.polyDataType, names.unstructuredGridType],
        point_field_name: str=None
    )   -> Union[names.polyDataType, names.unstructuredGridType]:
    """Convert a single point field to a cell field defined on a VTK object.

    Given a VTK poly data or an unstructured grid, convert the
    passed point field name to a cell field. If the field_name is
    None, convert all point fields to cell fields.

    .. note::
        By default, it leaves the point field on the object.
    """
    pointFields = GetPointArrays(vtk_object)

    if point_field_name is not None:
        if point_field_name not in pointFields:
            raise ValueError(
                      point_field_name + " not found in object. Aborting."
                  )

        # To allow the behavior of a single field being interpolated,
        # the fields that will be kept as cells fields only must be
        # deleted after the operation, so keep a list with them
        pointFieldsToKeep = copy(pointFields)
        pointFieldsToKeep.remove(point_field_name)

    pointToCell = vtk.vtkPointDataToCellData()
    pointToCell.SetInputData(vtk_object)
    pointToCell.SetPassPointData(True)
    pointToCell.Update()

    vtk_object = pointToCell.GetOutput()

    if point_field_name is not None:
        # Now delete all the cell fields in pointFieldsToKeep
        for cfield in pointFieldsToKeep:
            vtk_object.GetCellData().RemoveArray(cfield)

    return vtk_object

# This class was adapted from the 'vmtkcenterlines.py' script
# distributed with VMTK in https://github.com/vmtk/vmtk
class PickPointSeedSelector():
    """Select point on a surface."""

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

def SelectSurfacePoint(
        surface: names.polyDataType
    )   -> Union[tuple,list]:
    """Enable selection of point on a surface."""

    # Select aneurysm tip point
    pickPoint = PickPointSeedSelector()
    pickPoint.SetSurface(surface)
    pickPoint.InputInfo("Select point on the aneurysm surface\n")
    pickPoint.Execute()

    return pickPoint.PickedSeeds.GetPoint(0)
