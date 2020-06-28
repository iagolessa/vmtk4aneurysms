"""Provide functions to compute geometric properties of VTK poly data."""

import vtk
import math
from vmtk import vtkvmtk
from vmtk import vmtkscripts
from vmtk import vmtkrenderer

from constants import *


def distance(point1, point2):
    sqrDistance = vtk.vtkMath.Distance2BetweenPoints(
                    point1,
                    point2
                )
    
    return math.sqrt(sqrDistance)

def surfaceArea(surface):
    """Compute the surface area of an input surface."""

    surface_area = vtk.vtkMassProperties()
    surface_area.SetInputData(surface)
    surface_area.Update()

    return surface_area.GetSurfaceArea()

def surfaceVolume(surface):
    """Compute voluem of closed surface.

    Computes the volume of an assumed orientable 
    surface. Works internally with VTK, so it 
    assumes that the surface is closed. 
    """

    volume = vtk.vtkMassProperties()
    volume.SetInputData(surface)
    volume.Update()

    return volume.GetVolume()

def contourPerimeter(contour):
    """Compute the perimeter of a contour defined in 3D space."""

    nContourVertices = contour.GetNumberOfPoints()

    # Compute neck perimeter
    perimeter = intZero
    previous = contour.GetPoint(intZero)

    for index in range(nContourVertices):
        if index > intZero:
            previous = contour.GetPoint(index - intOne)

        vertex = contour.GetPoint(index)

        perimeter += distance(previous, vertex)

    return perimeter

def contourBarycenter(contour):
    """Return contour barycenter."""

    # For the barycenter, the contour can be open
    contourPoints = contour.GetPoints()

    barycenter = [0.0, 0.0, 0.0]
    vtkvmtk.vtkvmtkBoundaryReferenceSystems.ComputeBoundaryBarycenter(
        contourPoints,
        barycenter
    )

    return tuple(barycenter)

def contourPlaneArea(contour):
    """Compute plane surface area enclosed by a 3D contour path."""

    # Fill contour
    # TODO: This algorithms seems to fail to triangulate the contour if
    # it is not closed. Need to find alternatives
    fillContour = vtk.vtkContourTriangulator()
    fillContour.SetInputData(contour)
    fillContour.Update()

    # Convert vtkUnstructuredData to vtkPolyData
    meshToSurfaceFilter = vtk.vtkGeometryFilter()
    meshToSurfaceFilter.SetInputData(fillContour.GetOutput())
    meshToSurfaceFilter.Update()

    computeArea = vtk.vtkMassProperties()
    computeArea.SetInputData(meshToSurfaceFilter.GetOutput())
    computeArea.Update()

    return computeArea.GetSurfaceArea()

    # if computeArea.GetSurfaceArea() == 0.0:
        # raise ValueError
    # else:
        # return computeArea.GetSurfaceArea()
#
# except ValueError:
    # # Alternative procedure based on closing the aneurysm
    # pass

## TODO: overload this functions: receive a surface closed too
## investigate functionoverloading for modules in Python
def contourHydraulicDiameter(contour):
    """Compute hydraulic diameter of a plane contour."""

    perimeter = contourPerimeter(contour)
    area = contourPlaneArea(contour)

    # Return hydraulic diameter of neck
    return intFour*area/perimeter

def contourIsClosed(contour):
    """Check if contour (vtkPolyData) is closed."""
    nVertices = contour.GetNumberOfPoints()
    nEdges = contour.GetNumberOfCells()

    return nVertices == nEdges



