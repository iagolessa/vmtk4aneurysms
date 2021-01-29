import numpy as np
from scipy.integrate import simps

import vtk
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
