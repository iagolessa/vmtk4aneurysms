"""Provide functions to compute geometric properties of VTK poly data."""

import vtk
import math
from vmtk import vtkvmtk

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

# TODO: overload this functions: receive a surface closed too
# investigate functionoverloading for modules in Python


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


def surfaceCurvature(surface):
    """Compute curvature of surface.

    Uses VTK to compute the mean and Gauss curvature 
    of a surface represented as a vtkPolydata. Also
    computes a integer array that identify the local
    shape of a surface, as presented by Ma et al. (2004)
    for intracranial aneurysms, if Kg and Km are the 
    Gauss and mean curvature, we have:

        Kg   Km     Local Shape         Int Label
        > 0  > 0    Elliptical Convex   0
        > 0  < 0    Elliptical Concave  1
        > 0  = 0    Not possible        2
        < 0  > 0    Hyperbolic Convex   3
        < 0  < 0    Hyperbolic Concave  4
        < 0  = 0    Hyperbolic          5
        = 0  > 0    Cylidrical Convex   6
        = 0  < 0    Cylidrical Concave  7
        = 0  = 0    Planar              8

    The name of the generated arrays are: "Mean_Curvature", 
    "Gauss_Curvature", and "Local_Shape_Type".
    """
    # Compute mean curvature
    meanCurvature = vtk.vtkCurvatures()
    meanCurvature.SetInputData(surface)
    meanCurvature.SetCurvatureTypeToMean()
    meanCurvature.Update()

    # Compute Gaussian curvature
    gaussianCurvature = vtk.vtkCurvatures()
    gaussianCurvature.SetInputData(meanCurvature.GetOutput())
    gaussianCurvature.SetCurvatureTypeToGaussian()
    gaussianCurvature.Update()

    cellCurvatures = vtk.vtkPointDataToCellData()
    cellCurvatures.SetInputData(gaussianCurvature.GetOutput())
    cellCurvatures.PassPointDataOff()
    cellCurvatures.Update()

    curvatureSurface = cellCurvatures.GetOutput()

    # Set types of surface patches based on curvatures
    gaussCurvatureArray = curvatureSurface.GetCellData().GetArray('Gauss_Curvature')
    meanCurvatureArray = curvatureSurface.GetCellData().GetArray('Mean_Curvature')

    # Add an int array tp hold the surface local type
    localShapeScalars = vtk.vtkIntArray()
    localShapeScalars.SetNumberOfComponents(1)
    localShapeScalars.SetNumberOfTuples(curvatureSurface.GetNumberOfCells())
    localShapeScalars.SetName('Local_Shape_Type')
    localShapeScalars.FillComponent(0, 0)

    curvatureSurface.GetCellData().AddArray(localShapeScalars)
    # curvatureSurface.GetPointData().SetActiveScalars('Local_Shape_Type')

    # Update local type based on curvatures
    for cell in range(cellCurvatures.GetOutput().GetNumberOfCells()):
        meanCurvature = meanCurvatureArray.GetValue(cell)
        GaussCurvature = gaussCurvatureArray.GetValue(cell)

        # Elliptical convex
        if GaussCurvature > 0.0 and meanCurvature > 0.0:
            localShapeScalars.SetValue(cell, 0)

        # Elliptical concave
        elif GaussCurvature > 0.0 and meanCurvature < 0.0:
            localShapeScalars.SetValue(cell, 1)

        # Apparently, not possible
        elif GaussCurvature > 0.0 and meanCurvature == 0.0:
            localShapeScalars.SetValue(cell, 2)

        # Hyperbolic "more convex"
        elif GaussCurvature < 0.0 and meanCurvature > 0.0:
            localShapeScalars.SetValue(cell, 3)

        # Hyperbolic "more concave"
        elif GaussCurvature < 0.0 and meanCurvature < 0.0:
            localShapeScalars.SetValue(cell, 4)

        # Hyperbolic
        elif GaussCurvature < 0.0 and meanCurvature == 0.0:
            localShapeScalars.SetValue(cell, 5)

        # Cylindric convex
        elif GaussCurvature == 0.0 and meanCurvature > 0.0:
            localShapeScalars.SetValue(cell, 6)

        # Cylindric concave
        elif GaussCurvature == 0.0 and meanCurvature < 0.0:
            localShapeScalars.SetValue(cell, 7)

        # Planar
        elif GaussCurvature == 0.0 and meanCurvature == 0.0:
            localShapeScalars.SetValue(cell, 8)

    return curvatureSurface


if __name__ == '__main__':
    import sys
    import polydatatools as tools

    filename = sys.argv[1]

    surface = tools.readSurface(filename)

    curvatures = surfaceCurvature(surface)

    tools.writeSurface(curvatures, '/home/iagolessa/tmp.vtp')