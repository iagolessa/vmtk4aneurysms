"""Provide functions to compute geometric properties of VTK poly data."""

import vtk
import math

from vmtk import vtkvmtk
from numpy import multiply
from vtk.numpy_interface import dataset_adapter as dsa

from . import constants as const

# Attribute array names
_polyDataType = vtk.vtkCommonDataModelPython.vtkPolyData
_multiBlockType = vtk.vtkCommonDataModelPython.vtkMultiBlockDataSet

_grad = '_gradient'
_sgrad = '_sgradient'
_normals = 'Normals'

def Distance(point1, point2):
    """Compute distance between two points."""
    sqrDistance = vtk.vtkMath.Distance2BetweenPoints(
        point1,
        point2
    )

    return math.sqrt(sqrDistance)

def SurfaceArea(surface: _polyDataType) -> float:
    """Compute the surface area of an input surface."""

    triangulate = vtk.vtkTriangleFilter()
    triangulate.SetInputData(surface)
    triangulate.Update()

    surface_area = vtk.vtkMassProperties()
    surface_area.SetInputData(triangulate.GetOutput())
    surface_area.Update()

    return surface_area.GetSurfaceArea()


def SurfaceVolume(surface: _polyDataType) -> float:
    """Compute voluem of closed surface.

    Computes the volume of an assumed orientable surface. Works internally with
    VTK, so it assumes that the surface is closed. 
    """

    triangulate = vtk.vtkTriangleFilter()
    triangulate.SetInputData(surface)
    triangulate.Update()

    volume = vtk.vtkMassProperties()
    volume.SetInputData(triangulate.GetOutput())
    volume.Update()

    return volume.GetVolume()

def SurfaceNormals(surface: _polyDataType) -> _polyDataType:
    """Compute outward surface normals."""
    
    normals = vtk.vtkPolyDataNormals()
    
    normals.ComputeCellNormalsOn()
    normals.ComputePointNormalsOff()
    # normals.AutoOrientNormalsOff()
    # normals.FlipNormalsOn()
    normals.SetInputData(surface)
    normals.Update()
    
    return normals.GetOutput()

def SpatialGradient(surface: _polyDataType, 
                    field_name: str) -> _polyDataType:
    """Compute gradient of field on a surface."""
    
    gradient = vtk.vtkGradientFilter()
    gradient.SetInputData(surface)

    # TODO: Make check of type of field
    # scalar or vector
    # 1 is the field type: means vector
    gradient.SetInputScalars(1, field_name)
    gradient.SetResultArrayName(field_name+_grad)
    gradient.ComputeDivergenceOff()
    gradient.ComputeGradientOn()
    gradient.ComputeQCriterionOff()
    gradient.ComputeVorticityOff()
    gradient.Update()

    return gradient.GetOutput()

def SurfaceGradient(surface: _polyDataType, 
                    field_name: str) -> _polyDataType:
    """Compute surface gradient of field on a surface.
    
    Given the surface (vtkPolyData) and the scalar field name, compute the
    tangential or surface gradient of it on the surface.
    """
    
    cellData = surface.GetCellData()
    nArrays  = cellData.GetNumberOfArrays()
    
    # Compute normals, if necessary
    arrays = [cellData.GetArray(id_).GetName()
              for id_ in range(nArrays)]
    
    if _normals not in arrays:
        surface = SurfaceNormals(surface)
        
    # Compute spatial gradient (adds field)
    surfaceWithGradient = SpatialGradient(surface, field_name)
    
    # GetArrays
    npSurface = dsa.WrapDataObject(surfaceWithGradient)
    getArray = npSurface.GetCellData().GetArray
    
    normalsArray  = getArray(_normals)
    gradientArray = getArray(field_name + _grad)
    
    # Compute the normal gradient = vec(n) dot grad(field)
    normalGradient = multiply(normalsArray, gradientArray).sum(axis=1)

    # Compute the surface gradient
    surfaceGrad = gradientArray - normalGradient*normalsArray
    
    npSurface.CellData.append(surfaceGrad, 
                              field_name + _sgrad)
    
    # Clean up
    npSurface.GetCellData().RemoveArray(field_name + _grad)
    
    return npSurface.VTKObject

def ContourPerimeter(contour):
    """Compute the perimeter of a contour defined in 3D space."""

    nContourVertices = contour.GetNumberOfPoints()

    # Compute neck perimeter
    perimeter = const.zero
    previous = contour.GetPoint(int(const.zero))

    for index in range(nContourVertices):
        if index > int(const.zero):
            previous = contour.GetPoint(index - 1)

        vertex = contour.GetPoint(index)

        perimeter += Distance(previous, vertex)

    return perimeter


def ContourBarycenter(contour):
    """Return contour barycenter."""

    # For the barycenter, the contour can be open
    contourPoints = contour.GetPoints()

    barycenter = [0.0, 0.0, 0.0]
    vtkvmtk.vtkvmtkBoundaryReferenceSystems.ComputeBoundaryBarycenter(
        contourPoints,
        barycenter
    )

    return tuple(barycenter)


def ContourPlaneArea(contour):
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


def ContourHydraulicDiameter(contour):
    """Compute hydraulic diameter of a plane contour."""

    perimeter = ContourPerimeter(contour)
    area = ContourPlaneArea(contour)

    # Return hydraulic diameter of neck
    return const.four*area/perimeter


# TODO: review this function to check closed contour
def ContourIsClosed(contour):
    """Check if contour (vtkPolyData) is closed."""
    nVertices = contour.GetNumberOfPoints()
    nEdges = contour.GetNumberOfCells()

    return nVertices == nEdges


def SurfaceCurvature(surface):
    """Compute curvature of surface.

    Uses VTK to compute the mean and Gauss curvature of a surface represented
    as a vtkPolydata. Also computes an integer array that identify the local
    shape of the surface, as presented by Ma et al. (2004) for intracranial
    aneurysms, if Kg and Km are the Gauss and mean curvature, we have:

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

    The name of the generated arrays are: "Mean_Curvature", "Gauss_Curvature",
    and "Local_Shape_Type".
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
