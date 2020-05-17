"""Provide functions to compute geometric properties of VTK poly data."""

import vtk
from vmtk import vmtkscripts
from vmtk import vmtkrenderer

from constants import *


def distance(point1, point2):
    sqrDistance = vtk.vtkMath.Distance2BetweenPoints(
                    point1,
                    point2
                )
    
    return np.sqrt(sqrDistance)

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
