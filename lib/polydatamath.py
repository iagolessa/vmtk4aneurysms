import sys
import numpy as np
from typing import Union
from scipy.integrate import simps

import vtk

from vtk.numpy_interface import dataset_adapter as dsa
from . import names
from . import polydatatools as tools
from .polydatageometry import Surface

def NormL2(array, axis):
    """Compute L2-norm of an array along an axis."""

    return np.linalg.norm(array, ord=2, axis=axis)

def TimeAverage(y_array, time_array):
    """Compute temporal average of a time-dependent variable."""

    period = time_array.max() - time_array.min()

    return simps(y_array, x=time_array, axis=0)/period

# TODO: improve this computattion. I thought about using vtkIntegrateAttributes
# but is not available in the version shipped with vmtk!
def SurfaceAverage(
        vtk_object: Union[names.polyDataType, names.unstructuredGridType],
        array_name: str
    )   -> float:
    """Compute area-averaged array over surface with first-order accuracy."""

    # Operate on copy of the vtk_object to be able to destroy all other
    # fields on the surface (improves performance when triangulating)
    vtk_object = tools.CopyVtkObject(vtk_object)
    cellArrays = tools.GetCellArrays(vtk_object)

    # Check if array is in surface
    if array_name not in cellArrays:
        raise ValueError(array_name + " not found in the VTK object.")

    else:
        # Delete all fields in vtk_object, except the one passed
        cellArrays.remove(array_name)

        for field_name in cellArrays:
            vtk_object.GetCellData().RemoveArray(field_name)

    # Needs to triangulate the object to get the cell areas and volumes
    # This generic filter can oerate on both vtkPolyData or vtkUnstructuredGrid
    triangleFilter = vtk.vtkDataSetTriangleFilter()
    triangleFilter.SetInputData(vtk_object)
    triangleFilter.Update()

    vtk_object = triangleFilter.GetOutput()

    # Use Numpy interface to compute field norm
    npVtkObject   = dsa.WrapDataObject(vtk_object)
    arrayOnObject = npVtkObject.GetCellData().GetArray(array_name)

    # Check type of field: vector or scalar
    nComponents = arrayOnObject.shape[-1]

    if nComponents == 3 or nComponents == 6:
        arrayOnObject = NormL2(arrayOnObject, 1)

        npVtkObject.CellData.append(arrayOnObject, array_name)

    # back to VTK interface
    vtk_object = npVtkObject.VTKObject

    # Check if surface or volume
    if type(vtk_object) == names.polyDataType:

        cellAreas = np.array(
                        [vtk_object.GetCell(id_).ComputeArea()
                         for id_ in range(vtk_object.GetNumberOfCells())]
                    )

        return np.sum(arrayOnObject*cellAreas)/Surface.Area(vtk_object)

    elif type(vtk_object) == names.unstructuredGridType:

        # Compute Cell volumes
        meshQuality = vtk.vtkMeshQuality()
        meshQuality.SetInputData(vtk_object)
        meshQuality.SetTetQualityMeasureToVolume()
        meshQuality.Update()

        npVtkObject = dsa.WrapDataObject(meshQuality.GetOutput())
        cellVolumes = npVtkObject.CellData.GetArray("Quality")

        return np.sum(arrayOnObject*cellVolumes)/np.sum(cellVolumes)

    else:
        raise TypeError(
                "VTK object must be either vtkPolyData or vtkUnstructuredGrid."
              )


def SurfaceFieldStatistics(
        surface: names.polyDataType,
        field_name: str,
        n_percentile: float=99
    )   -> dict:
    """Computes a field statistics defined on a surface.

    Given a surface (or a surface patch), as a vtkPolyData type, with a
    field_name defined on it, compute the field's descriptive statistics
    (average, maximum, minimum, and the 95 percentil by default) and the
    surface average. Returns a dict with each statistic.

    The function accepts either point or a cell array, but if a point array is
    passed, the function first converts it to a cell array before the
    computation of the stats.
    """

    if type(surface) != names.polyDataType:
        raise TypeError("Need vtkPolyData surface, not {}.". format(
                    type(surface)
                 )
              )
    if surface.GetNumberOfCells() == 0:
        raise ValueError("The surface passed has no cells!")

    pointArrays = tools.GetPointArrays(surface)
    cellArrays  = tools.GetCellArrays(surface)

    fieldInSurface = field_name in pointArrays or field_name in cellArrays

    # Check if arrays are on surface
    if not fieldInSurface:
        raise ValueError(
                  "{} field not found in the surface.".format(field_name)
              )

    if field_name in pointArrays:
        pointToCell = vtk.vtkPointDataToCellData()
        pointToCell.SetInputData(surface)
        pointToCell.PassPointDataOff()
        pointToCell.Update()

        surface = pointToCell.GetOutput()

    # Get Array
    npSurface = dsa.WrapDataObject(surface)

    fieldOnSurface = npSurface.GetCellData().GetArray(field_name)

    # Check type of field: vector or scalar
    nComponents = fieldOnSurface.shape[-1]

    if nComponents == 3 or nComponents == 6:
        fieldOnSurface = NormL2(fieldOnSurface, 1)

    # Compute statistics
    return {"surf_avg": SurfaceAverage(surface, field_name),
            "average": np.average(fieldOnSurface),
            "maximum": np.max(fieldOnSurface),
            "minimum": np.min(fieldOnSurface),
            "percentil"+str(n_percentile): np.percentile(
                                               fieldOnSurface,
                                               n_percentile
                                           )}

def HadamardDot(np_array1, np_array2):
    """Computes dot product in a Hadamard product way.

    Given two Numpy arrays representing arrays of vectors on a surface, compute
    the vector-wise dot product between each element.
    """
    # Seems that multiply is faster than a*b
    return np.multiply(np_array1, np_array2).sum(axis=1)
