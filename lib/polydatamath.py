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

"""Collection of mathematical functions that operates on VTK objects."""

import sys
import numpy as np
from typing import Union
from scipy.integrate import simps

import vtk

from vtk.numpy_interface import dataset_adapter as dsa
from . import names
from . import polydatatools as tools
from . import polydatageometry as geo


def GetFieldType(
        vtk_np_array: names.vtkArrayType
    )   -> str:
    """Return the field type (scalar, vector or tensor).

    Given a field defined via the VTK's numpy adaptor, ie a NumPy-like array,
    return the type of the field.

    Raises NonImplementedError for fields of tensors of higher order or
    temporal arrays. In this case, only scalars, vector, and tensor of
    second-order are contemplated.

    Returns
    -------
        str ('scalarField', 'vectorField',
             'tensor2SymmField', 'tensor2Field')
    """

    if type(vtk_np_array) == dsa.VTKNoneArray:
        raise ValueError("Detected None array. The array must not be empy.")

    # Get whether scalar, vector or tensor
    # being field a numpy array
    npDimensions = vtk_np_array.ndim

    if npDimensions == 1:
        # Due to the nature of a numpy array
        # in this case it will be a scalar field
        fieldType = names.scalarFieldLabel

    elif npDimensions == 2:
        # Either vector of second order symmetric tensor

        # get the axis 1 size of the numpy array
        # since it is better to get the rank of
        # the entity based on it
        fieldDim = vtk_np_array.shape[-1]

        if fieldDim == 3:
            fieldType = names.vectorFieldLabel

        elif fieldDim == 6:
            # Second-order tensor
            fieldType = names.tensor2SymmFieldLabel

        else:
            raise ValueError(
                     "Unknown field type. Field dimensionality " + str(fieldDim) + "."
                  )

    elif npDimensions == 3:
        fieldType = names.tensor2FieldLabel

    else:
        raise NotImplementedError("Field array has a higher dimensionality.")

    return fieldType

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
    """Compute area-averaged array over surface with first-order accuracy.

    .. warning::
        The computation requires scalar field only to be integrated. If a
        vector or second-order symmstric tensors are passed, it first computes
        the L2-norm of them. Higher-order tensor are still not supported.
    """

    vtkObjectOriginalType = type(vtk_object)

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
    # But it returns unstructuredGrid!
    triangleFilter = vtk.vtkDataSetTriangleFilter()
    triangleFilter.SetInputData(vtk_object)
    triangleFilter.Update()

    # Dataset Triangle filter outputs a unstrcturedGrid, so convert it if
    # vtk_object was a vtk_poly data originally
    if vtkObjectOriginalType == names.polyDataType:
        vtk_object = tools.UnsGridToPolyData(triangleFilter.GetOutput())

    # Use Numpy interface to compute field norm
    npVtkObject   = dsa.WrapDataObject(vtk_object)
    arrayOnObject = npVtkObject.GetCellData().GetArray(array_name)

    # Check type of field: vector or scalar
    arrayType = GetFieldType(arrayOnObject)

    if arrayType == names.vectorFieldLabel or \
       arrayType == names.tensor2SymmFieldLabel:
        # Compute a simple L2 norm, even for second-order symm tensors
        arrayOnObject = NormL2(arrayOnObject, 1)

        npVtkObject.CellData.append(arrayOnObject, array_name)

    elif arrayType == names.scalarFieldLabel:
        pass

    else:
        raise NotImplementedError(
                  "Array of type " + arrayType + ". Norm not implemented, yet."
              )

    # back to VTK interface
    vtk_object = npVtkObject.VTKObject

    # Check if surface or volume
    if type(vtk_object) == names.polyDataType:

        cellAreas = np.array(
                        [vtk_object.GetCell(id_).ComputeArea()
                         for id_ in range(vtk_object.GetNumberOfCells())]
                    )

        return np.sum(arrayOnObject*cellAreas)/geo.Surface.Area(vtk_object)

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
        vtk_object: Union[names.polyDataType, names.unstructuredGridType],
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

    # if type(vtk_object) != names.polyDataType:
    #     raise TypeError("Need vtkPolyData surface, not {}.". format(
    #                 type(vtk_object)
    #              )
    #           )

    if vtk_object.GetNumberOfCells() == 0:
        raise ValueError("The object passed has no cells!")

    pointArrays = tools.GetPointArrays(vtk_object)
    cellArrays  = tools.GetCellArrays(vtk_object)

    isPointField = field_name in pointArrays
    isCellField  = field_name in cellArrays

    fieldInSurface = isPointField or isCellField

    # Check if arrays are on vtk_object
    if not fieldInSurface:
        raise ValueError(
                  "{} field not found in the VTK object.".format(field_name)
              )

    if isPointField and not isCellField:
        # Convert to cell field
        vtk_object = tools.PointFieldToCellField(
                      vtk_object,
                      field_name
                  )

    # Use Numpy interface to compute field norm
    npVtkObject    = dsa.WrapDataObject(vtk_object)
    fieldOnSurface = npVtkObject.GetCellData().GetArray(field_name)

    # Check type of field: vector or scalar
    arrayType = GetFieldType(fieldOnSurface)

    if arrayType == names.vectorFieldLabel or \
       arrayType == names.tensor2SymmFieldLabel:

        # Compute a simple L2 norm, even for second-order symm tensors
        fieldOnSurface = NormL2(fieldOnSurface, 1)

        npVtkObject.CellData.append(fieldOnSurface, field_name)

    elif arrayType == names.scalarFieldLabel:
        pass

    else:
        raise NotImplementedError(
                  "Array of type " + arrayType + ". Norm not implemented, yet."
              )

    # Compute statistics
    return {"surf_avg": SurfaceAverage(vtk_object, field_name),
            "average": np.average(fieldOnSurface), # mean with wieghts = 1
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
