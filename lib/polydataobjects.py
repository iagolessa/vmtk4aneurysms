"""Provides geometric objects of the 3D space."""

import vtk
from vtk.numpy_interface import dataset_adapter as dsa
from numpy import zeros, where

from . import polydatatools as tools

class Surface():
    """Computational model of a three-dimensional surface."""
    
    def __init__(self, vtk_poly_data):
        """Build surface model from vtkPolyData.
        
        Given a vtkPolyData characterizing a surface in the 3D Euclidean space,
        automatically computes its outwards unit normal fiels, stored as
        'Normals' and its curvature type field based on the Gaussian and mean
        curvatures.
        """
        
        self._surface_object = vtk_poly_data
        self._compute_normals()
        self._compute_curvature()

    @classmethod
    def from_file(cls, file_name):
        """Build surface model from file."""

        return cls(tools.ReadSurface(file_name))
        
    def _compute_normals(self):
        """Compute outward surface normals."""

        normals = vtk.vtkPolyDataNormals()

        normals.ComputeCellNormalsOn()
        normals.ComputePointNormalsOff()
        # normals.AutoOrientNormalsOff()
        # normals.FlipNormalsOn()
        normals.SetInputData(self._surface_object)
        normals.Update()

        self._surface_object = normals.GetOutput()
 
    def _compute_curvature(self):
        """Compute curvature of surface.

        Uses VTK to compute the mean and Gauss curvature of a surface
        represented as a vtkPolydata. Also computes an integer array that
        identify the local shape of the surface, as presented by Ma et al.
        (2004) for intracranial aneurysms, if Kg and Km are the Gauss and mean
        curvature, we have:

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
        meanCurvature.SetInputData(self._surface_object)
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

        npCurvatures   = dsa.WrapDataObject(cellCurvatures.GetOutput())
        GaussCurvature = npCurvatures.GetCellData().GetArray('Gauss_Curvature')
        meanCurvature  = npCurvatures.GetCellData().GetArray('Mean_Curvature')

        surfaceLocalShapes = {
            'ellipticalConvex' : 
                {'condition': (GaussCurvature >  0.0) & (meanCurvature >  0.0), 
                 'id': 1},
            'ellipticalConcave': 
                {'condition': (GaussCurvature >  0.0) & (meanCurvature <  0.0), 
                 'id': 2},
            'elliptical'       : # apparently, not possible
                {'condition': (GaussCurvature >  0.0) & (meanCurvature == 0.0), 
                 'id': 3}, 
            'hyperbolicConvex' : 
                {'condition': (GaussCurvature <  0.0) & (meanCurvature >  0.0), 
                 'id': 4},
            'hyperboliConcave' : 
                {'condition': (GaussCurvature <  0.0) & (meanCurvature <  0.0), 
                 'id': 5},
            'hyperbolic'       : 
                {'condition': (GaussCurvature <  0.0) & (meanCurvature == 0.0), 
                 'id': 6},
            'cylindricConvex'  : 
                {'condition': (GaussCurvature == 0.0) & (meanCurvature >  0.0), 
                 'id': 7},
            'cylindricConcave' : 
                {'condition': (GaussCurvature == 0.0) & (meanCurvature <  0.0), 
                 'id': 8},
            'planar'           : 
                {'condition': (GaussCurvature == 0.0) & (meanCurvature == 0.0), 
                 'id': 9}
        }

        LocalShapeArray = zeros(shape=len(meanCurvature), dtype=int)

        for shape in surfaceLocalShapes.values(): 
            LocalShapeArray += where(shape.get('condition'), shape.get('id'), 0)

        npCurvatures.CellData.append(LocalShapeArray, 'Local_Shape_Type')

        self._surface_object = npCurvatures.VTKObject

    def GetSurfaceObject(self):
        """Return the surface vtkPolyData object."""
        return self._surface_object
    
    def GetSurfaceArea(self):
        """Return the surface area in the units of the original data."""

        triangulate = vtk.vtkTriangleFilter()
        triangulate.SetInputData(self._surface_object)
        triangulate.Update()

        surface_area = vtk.vtkMassProperties()
        surface_area.SetInputData(triangulate.GetOutput())
        surface_area.Update()

        return surface_area.GetSurfaceArea()

    def GetVolume(self):
        """Compute volume of closed surface.

        Computes the volume of an assumed orientable surface. Works internally
        with VTK, so it assumes that the surface is closed. 
        """
        
        triangulate = vtk.vtkTriangleFilter()
        triangulate.SetInputData(self._surface_object)
        triangulate.Update()

        volume = vtk.vtkMassProperties()
        volume.SetInputData(triangulate.GetOutput())
        volume.Update()

        return volume.GetVolume()
    
    def GetCellArrays(self):
        """Return the names and number of arrays for a vtkPolyData."""
    
        nCellArrays = self._surface_object.GetCellData().GetNumberOfArrays()

        return [self._surface_object.GetCellData().GetArray(id_).GetName()
                for id_ in range(nCellArrays)]

    def GetPointArrays(self):
        """Return the names of point arrays for a vtkPolyData."""

        nPointArrays = self._surface_object.GetPointData().GetNumberOfArrays()

        return [self._surface_object.GetPointData().GetArray(id_).GetName()
                for id_ in range(nPointArrays)]
