"""Collection of tools to compute geometric properties of VTK objects."""

import vtk
import math
import numpy as np
from typing import Union

from vmtk import vtkvmtk
from numpy import multiply, zeros, where
from vtk.numpy_interface import dataset_adapter as dsa

from . import names
from . import constants as const
from . import polydatatools as tools

def Distance(
        point1: tuple,
        point2: tuple
    )   -> float:
    """Compute distance between two points."""
    sqrDistance = vtk.vtkMath.Distance2BetweenPoints(
        point1,
        point2
    )

    return math.sqrt(sqrDistance)

def SpatialGradient(
        surface: names.polyDataType,
        field_name: str
    )   -> names.polyDataType:
    """Compute gradient of field on a surface."""

    gradient = vtk.vtkGradientFilter()
    gradient.SetInputData(surface)

    # TODO: Make check of type of field
    # scalar or vector
    # 1 is the field type: means vector
    gradient.SetInputScalars(1, field_name)
    gradient.SetResultArrayName(field_name+names.grad)
    gradient.ComputeDivergenceOff()
    gradient.ComputeGradientOn()
    gradient.ComputeQCriterionOff()
    gradient.ComputeVorticityOff()
    gradient.Update()

    return gradient.GetOutput()

def SurfaceGradient(
        surface: names.polyDataType,
        field_name: str
    )   -> names.polyDataType:
    """Compute surface gradient of field on a surface.

    Given the surface (vtkPolyData) and the scalar field name, compute the
    tangential or surface gradient of it on the surface.
    """

    cellData = surface.GetCellData()
    nArrays  = cellData.GetNumberOfArrays()

    # Compute normals, if necessary
    arrays = (cellData.GetArray(id_).GetName()
              for id_ in range(nArrays))

    if names.normals not in arrays:
        surface = Surface.Normals(surface)

    # Compute spatial gradient (adds field)
    surfaceWithGradient = SpatialGradient(surface, field_name)

    # GetArrays
    npSurface = dsa.WrapDataObject(surfaceWithGradient)
    getArray = npSurface.GetCellData().GetArray

    normalsArray  = getArray(names.normals)
    gradientArray = getArray(field_name + names.grad)

    # Compute the normal gradient = vec(n) dot grad(field)
    normalGradient = multiply(normalsArray, gradientArray).sum(axis=1)

    # Compute the surface gradient
    surfaceGrad = gradientArray - normalGradient*normalsArray

    npSurface.CellData.append(surfaceGrad,
                              field_name + names.sgrad)

    # Clean up
    npSurface.GetCellData().RemoveArray(field_name + names.grad)

    return npSurface.VTKObject

def ContourPerimeter(
        contour: names.polyDataType
    )   -> float:
    """Compute the perimeter of a contour defined in 3D space."""

    perimeter = 0.0

    # The approach employed here was chosen to keep a consistency
    # for the periemeter calculation of contours
    # obtained with both vtkvmtkBoundaryExtractor (single cell)
    # and vtkCutter.
    # TODO: The ideal alternative for this function is to use the
    # vtkIntegrateAttributes filter with VTK > 8.2
    #
    # Indicates that it was obtained with vtkvmtkBoundaryExtractor
    if contour.GetNumberOfCells() == 1:
        # Then use vtk-numpy interface
        npContour = dsa.WrapDataObject(contour)

        points = npContour.GetPoints()

        rotatedPoints = np.concatenate(
                            (np.array(points[1:]),
                             np.array([points[0]]))
                        )

        perimeter = sum([Distance(point1, point2)
                         for point1, point2 in zip(points, rotatedPoints)])

    else:
        # This one for some reason, does not work with the numpy interface
        perimeter = sum([math.sqrt(contour.GetCell(cell_id).GetLength2())
                         for cell_id in range(contour.GetNumberOfCells())])

    return perimeter

def ContourBarycenter(
        contour: names.polyDataType
    )   -> tuple:
    """Return contour barycenter."""

    # For the barycenter, the contour can be open
    contourPoints = contour.GetPoints()

    barycenter = [0.0, 0.0, 0.0]
    vtkvmtk.vtkvmtkBoundaryReferenceSystems.ComputeBoundaryBarycenter(
        contourPoints,
        barycenter
    )

    return tuple(barycenter)

def ContourPlaneArea(
        contour: names.polyDataType
    )   -> float:
    """Compute plane surface area enclosed by a 3D contour path."""

    # Fill contour
    # TODO: This algorithms seems to fail to triangulate the contour if
    # it is not closed. Need to find alternatives
    fillContour = vtk.vtkContourTriangulator()
    fillContour.SetInputData(contour)
    fillContour.Update()

    # Convert vtkUnstructuredData to vtkPolyData and compute area
    computeArea = vtk.vtkMassProperties()
    computeArea.SetInputData(
        tools.UnsGridToPolyData(fillContour.GetOutput())
    )
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

def ContourHydraulicDiameter(
        contour: names.polyDataType
    )   -> float:
    """Compute hydraulic diameter of a plane contour."""

    perimeter = ContourPerimeter(contour)
    area = ContourPlaneArea(contour)

    # Return hydraulic diameter of neck
    return const.four*area/perimeter


# TODO: review this function to check closed contour
def ContourIsClosed(
        contour: names.polyDataType
    )   -> bool:
    """Check if contour (vtkPolyData) is closed."""
    nVertices = contour.GetNumberOfPoints()
    nEdges = contour.GetNumberOfCells()

    return nVertices == nEdges

def WarpVtkObject(
        vtk_object: Union[names.polyDataType, names.unstructuredGridType],
        field_name: str
    )   -> Union[names.polyDataType, names.unstructuredGridType]:
    """Warp (or deform) a VTK object with a vector field (cell or point)."""

    # First check whether the field is on the object
    # and convert it to point field if necessary
    if field_name not in tools.GetPointArrays(vtk_object):

        if field_name in tools.GetCellArrays(vtk_object):

            # Convert field to point data
            print(
                "Found " + field_name + " in cell data. Converting to point field."
            )

            vtk_object = tools.CellFieldToPointField(
                             vtk_object,
                             field_name
                         )

        else:
            raise ValueError(field_name + "not found in object.")

    else:
        pass

    # Set active field
    displData = vtk_object.GetPointData().GetArray(field_name)
    vtk_object.GetPointData().SetActiveVectors(displData.GetName())

    # Warp the surfaces at the two instants
    warpObject = vtk.vtkWarpVector()
    warpObject.SetInputData(vtk_object)
    warpObject.SetScaleFactor(1.0)
    warpObject.Update()

    return warpObject.GetOutput()

def WarpPolydata(
        polydata: names.polyDataType,
        field_name: str
    )   -> names.polyDataType:
    """Given a vtkPolyData with a field, warp it by the field."""

    # For backward compatibility
    return WarpVtkObject(polydata, field_name)

class Surface():
    """Computational model of a three-dimensional surface."""

    def __init__(self, vtk_poly_data):
        """Build surface model from vtkPolyData.

        Given a vtkPolyData characterizing a surface in the 3D Euclidean space,
        automatically computes its outwards unit normal fiels, stored as
        'Normals' and its curvature type field based on the Gaussian and mean
        curvatures.
        """

        self._surface_object = Surface.Normals(vtk_poly_data)
        self._surface_object = Surface.Curvatures(self._surface_object)

    @classmethod
    def from_file(cls, file_name):
        """Build surface model from file."""

        return cls(tools.ReadSurface(file_name))

    @staticmethod
    def Area(surface_object: names.polyDataType) -> float:
        """Return the surface area in the units of the original data."""

        triangulate = vtk.vtkTriangleFilter()
        triangulate.SetInputData(surface_object)
        triangulate.Update()

        surface_area = vtk.vtkMassProperties()
        surface_area.SetInputData(triangulate.GetOutput())
        surface_area.Update()

        return surface_area.GetSurfaceArea()

    @staticmethod
    def Volume(surface_object: names.polyDataType) -> float:
        """Compute volume of closed surface.

        Computes the volume of an assumed orientable surface. Works internally
        with VTK, so it assumes that the surface is closed.
        """

        triangulate = vtk.vtkTriangleFilter()
        triangulate.SetInputData(surface_object)
        triangulate.Update()

        # Cap surface (mass properties requires it closed)
        surfaceCapper = vtkvmtk.vtkvmtkCapPolyData()
        surfaceCapper.SetInputConnection(triangulate.GetOutputPort())
        surfaceCapper.SetDisplacement(const.zero)
        surfaceCapper.SetInPlaneDisplacement(const.zero)
        surfaceCapper.Update()

        volume = vtk.vtkMassProperties()
        volume.SetInputData(surfaceCapper.GetOutput())
        volume.Update()

        return volume.GetVolume()

    @staticmethod
    def Normals(surface_object: names.polyDataType) -> names.polyDataType:
        """Compute outward surface normals."""

        normals = vtk.vtkPolyDataNormals()

        normals.ComputeCellNormalsOn()
        normals.ComputePointNormalsOff()
        # normals.AutoOrientNormalsOff()
        # normals.FlipNormalsOn()
        normals.SetInputData(surface_object)
        normals.Update()

        return normals.GetOutput()

    @staticmethod
    def Curvatures(surface_object: names.polyDataType) -> names.polyDataType:
        """Compute curvature of surface.

        Uses VTK to compute the mean and Gauss curvature of a surface
        represented as a vtkPolydata. Also computes an integer array that
        identify the local shape of the surface, as presented by Ma et al.
        (2004) for intracranial aneurysms, if Kg and Km are the Gauss and mean
        curvature, we have:

        .. table:: Local shape characterization
           :widths: auto

           === ==== ================== =========
           Kg   Km  Local Shape        Int Label
           === ==== ================== =========
           > 0  > 0 Elliptical Convex  0
           > 0  < 0 Elliptical Concave 1
           > 0  = 0 Not possible       2
           < 0  > 0 Hyperbolic Convex  3
           < 0  < 0 Hyperbolic Concave 4
           < 0  = 0 Hyperbolic         5
           = 0  > 0 Cylidrical Convex  6
           = 0  < 0 Cylidrical Concave 7
           = 0  = 0 Planar             8
           === ==== ================== =========

        The name of the generated arrays are: "Mean_Curvature",
        "Gauss_Curvature", and "Local_Shape_Type".
        """
        # Compute mean curvature
        meanCurvature = vtk.vtkCurvatures()
        meanCurvature.SetInputData(surface_object)
        meanCurvature.SetCurvatureTypeToMean()
        meanCurvature.Update()

        # Compute Gaussian curvature
        gaussianCurvature = vtk.vtkCurvatures()
        gaussianCurvature.SetInputData(meanCurvature.GetOutput())
        gaussianCurvature.SetCurvatureTypeToGaussian()
        gaussianCurvature.Update()

        # Compute Min and Max curvature
        minCurvature = vtk.vtkCurvatures()
        minCurvature.SetInputData(gaussianCurvature.GetOutput())
        minCurvature.SetCurvatureTypeToMinimum()
        minCurvature.Update()

        maxCurvature = vtk.vtkCurvatures()
        maxCurvature.SetInputData(minCurvature.GetOutput())
        maxCurvature.SetCurvatureTypeToMaximum()
        maxCurvature.Update()

        cellCurvatures = vtk.vtkPointDataToCellData()
        cellCurvatures.SetInputData(maxCurvature.GetOutput())
        cellCurvatures.PassPointDataOn()
        cellCurvatures.Update()

        npCurvatures   = dsa.WrapDataObject(cellCurvatures.GetOutput())
        GaussCurvature = npCurvatures.GetCellData().GetArray('Gauss_Curvature')
        meanCurvature  = npCurvatures.GetCellData().GetArray('Mean_Curvature')

        surfaceLocalShapes = {
            'ellipticalConvex' :
                {'condition': (GaussCurvature >  0.0) & (meanCurvature >  0.0),
                 'id': 0},
            'ellipticalConcave':
                {'condition': (GaussCurvature >  0.0) & (meanCurvature <  0.0),
                 'id': 1},
            'elliptical'       : # apparently, not possible
                {'condition': (GaussCurvature >  0.0) & (meanCurvature == 0.0),
                 'id': 2},
            'hyperbolicConvex' :
                {'condition': (GaussCurvature <  0.0) & (meanCurvature >  0.0),
                 'id': 3},
            'hyperboliConcave' :
                {'condition': (GaussCurvature <  0.0) & (meanCurvature <  0.0),
                 'id': 4},
            'hyperbolic'       :
                {'condition': (GaussCurvature <  0.0) & (meanCurvature == 0.0),
                 'id': 5},
            'cylindricConvex'  :
                {'condition': (GaussCurvature == 0.0) & (meanCurvature >  0.0),
                 'id': 6},
            'cylindricConcave' :
                {'condition': (GaussCurvature == 0.0) & (meanCurvature <  0.0),
                 'id': 7},
            'planar'           :
                {'condition': (GaussCurvature == 0.0) & (meanCurvature == 0.0),
                 'id': 8}
        }

        LocalShapeArray = zeros(shape=len(meanCurvature), dtype=int)

        for shape in surfaceLocalShapes.values():
            LocalShapeArray += where(shape.get('condition'), shape.get('id'), 0)

        npCurvatures.CellData.append(LocalShapeArray, 'Local_Shape_Type')

        return npCurvatures.VTKObject

    def GetSurfaceObject(self):
        """Return the surface vtkPolyData object."""
        return self._surface_object

    def GetSurfaceArea(self):
        """Return the surface total area."""
        return Surface.Area(self._surface_object)

    def GetSurfaceVolume(self):
        """Return the surface total enclosed volume."""
        return Surface.Volume(self._surface_object)

    def GetCellArrays(self):
        """Return the names of arrays for a vtkPolyData."""
        return tools.GetCellArrays(self._surface_object)

    def GetPointArrays(self):
        """Return the names of point arrays for a vtkPolyData."""
        return tools.GetPointArrays(self._surface_object)
