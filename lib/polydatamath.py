import sys
import numpy as np
from scipy.integrate import simps

import vtk

from vtk.numpy_interface import dataset_adapter as dsa
from . import polydatatools as tools
from .polydatageometry import Surface

def NormL2(array, axis):
    """Compute L2-norm of an array along an axis."""

    return np.linalg.norm(array, ord=2, axis=axis)

def TimeAverage(array, step, period):
    """Compute temporal average of a time-dependent variable."""

    return simps(array, dx=step, axis=0)/period

# TODO: improve this computattion. I thought about using vtkIntegrateAttributes
# but is not available in the version shipped with vmtk!
def SurfaceAverage(surface, array_name):
    """Compute area-averaged array over surface with first-order accuracy."""

    triangulate = vtk.vtkTriangleFilter()
    triangulate.SetInputData(surface)
    triangulate.Update()

    surface = triangulate.GetOutput()

    # Check if array is in surface
    if array_name not in tools.GetCellArrays(surface):
        sys.exit(array_name + " not found in the surface.")
    else:
        pass

    # Use Numpy interface to compute field norm
    npSurface = dsa.WrapDataObject(surface)
    arrayOnSurface = npSurface.GetCellData().GetArray(array_name)

    # Check type of field: vector or scalar
    nComponents = arrayOnSurface.shape[-1]

    if nComponents == 3 or nComponents == 6:
        arrayOnSurface = NormL2(arrayOnSurface, 1)

        npSurface.CellData.append(arrayOnSurface, array_name)

        # back to VTK interface
        surface = npSurface.VTKObject
    else:
        pass

    # Helper functions
    cellData = surface.GetCellData()
    getArea  = lambda id_: surface.GetCell(id_).ComputeArea()
    getValue = lambda id_, name: cellData.GetArray(name).GetValue(id_)

    def getCellValue(id_):
        cellArea   = getArea(id_)
        arrayValue = getValue(id_, array_name)

        return cellArea, arrayValue

    integral = 0.0
    cellIds = range(surface.GetNumberOfCells())

    # Map function to cell ids
    integral = sum(area*value for area, value in map(getCellValue, cellIds))

    surfaceArea = Surface.Area(surface)

    # Compute L2-norm
    return integral/surfaceArea

def HadamardDot(np_array1, np_array2):
    """Computes dot product in a Hadamard product way.

    Given two Numpy arrays representing arrays of vectors on a surface, compute
    the vector-wise dot product between each element.
    """
    # Seems that multiply is faster than a*b
    return np.multiply(np_array1, np_array2).sum(axis=1)
