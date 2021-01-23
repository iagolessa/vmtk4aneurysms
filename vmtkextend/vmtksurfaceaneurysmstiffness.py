#! /usr/bin/env python

# Creating Python Class to handle aneurysms operations
# My idea is to expand this class in the future:
# extract the aneyurysm and calculate other geometric parameters
import sys
import vtk
import math
import vtk.numpy_interface.dataset_adapter as dsa

from vmtk import vtkvmtk
from vmtk import vmtkscripts
from vmtk import pypes

vmtksurfaceaneurysmstiffness = 'vmtkSurfaceAneurysmStiffness'

class vmtkSurfaceAneurysmStiffness(pypes.pypeScript):

    def __init__(self):
        pypes.pypeScript.__init__(self)

        self.Surface = None
        self.UniformStiffness = False
        self.StiffnessArrayName = "E"
        self.DistanceArrayName = "DistanceToNeckArray"

        self.ArteriesStiffness = 5e6
        self.AneurysmStiffness = 1e6

        self.SetScriptName('vmtksurfaceaneurysmstiffness')
        self.SetScriptDoc('')

        self.SetInputMembers([
            ['Surface', 'i', 'vtkPolyData', 1, '',
                'the input surface', 'vmtksurfacereader'],

            ['UniformStiffness', 'uniformstiffness', 'bool', 1, '',
                'indicates uniform aneurysm stiffness'],

            ['StiffnessArrayName', 'stiffnessarray', 'str', 1, '',
                'name of the resulting stiffness array'],

            ['ArteriesStiffness', 'arteriesstiffness', 'float', 1, '',
                'stiffness of the arteries (and aneurysm neck)'],

            ['AneurysmStiffness', 'aneurysmstiffness', 'float', 1, '',
                'aneurysm stiffness (also aneurysm fundus stiffness)'],
        ])

        self.SetOutputMembers([
            ['Surface', 'o', 'vtkPolyData', 1, '',
                'the input surface with thickness array', 'vmtksurfacewriter'],
        ])

    def _smooth_array(self, surface, array, niterations=5, relax_factor=1.0):
        """Surface array smoother."""

        # arraySmoother = vmtkscripts.vmtkSurfaceArraySmoothing()
        # arraySmoother.Surface = surface
        # arraySmoother.SurfaceArrayName = array
        # arraySmoother.Connexity = 1
        # arraySmoother.Relaxation = 1.0
        # arraySmoother.Iterations = niterations
        # arraySmoother.Execute()

        _SMALL = 1e-12
        array = surface.GetPointData().GetArray(array)

        extractEdges = vtk.vtkExtractEdges()
        extractEdges.SetInputData(surface)
        extractEdges.Update()

        # Get surface edges
        surfEdges = extractEdges.GetOutput()

        for n in range(niterations):

            # Iterate over all edges cells
            for i in range(surfEdges.GetNumberOfPoints()):
                # Get edge cells
                cells = vtk.vtkIdList()
                surfEdges.GetPointCells(i, cells)

                sum_ = 0.0
                normFactor = 0.0

                # For each edge cells
                for j in range(cells.GetNumberOfIds()):

                    # Get points
                    points = vtk.vtkIdList()
                    surfEdges.GetCellPoints(cells.GetId(j), points)

                    # Over points in edge cells
                    for k in range(points.GetNumberOfIds()):

                        # Compute distance of the current point
                        # to all surface points
                        if points.GetId(k) != i:

                            # Compute distance between a point and surrounding
                            distance = math.sqrt(
                                vtk.vtkMath.Distance2BetweenPoints(
                                    surface.GetPoint(i),
                                    surface.GetPoint(points.GetId(k))
                                )
                            )

                            # Get inverse to act as weight?
                            weight = 1.0/(distance + _SMALL)

                            # Get value
                            value = array.GetTuple1(points.GetId(k))

                            normFactor += weight
                            sum_ += value*weight

                currVal = array.GetTuple1(i)

                # Average value weighted by the surrounding values
                weightedValue = sum_/normFactor

                newValue = relax_factor*weightedValue + \
                           (1.0 - relax_factor)*currVal

                array.SetTuple1(i, newValue)

        return surface

    def Execute(self):

        if self.Surface == None:
            self.PrintError('Error: no Surface.')

        # Fundus and neck stiffness
        fundusStiffness = self.AneurysmStiffness
        neckStiffness = self.ArteriesStiffness

        # I had a bug with the 'select thinner regions' with 
        # polygonal meshes. So, operate on a triangulated surface
        # and map final result to orignal surface
        cleaner = vtk.vtkCleanPolyData()
        cleaner.SetInputData(self.Surface)
        cleaner.Update()
        
        # Reference to original surface 
        polygonalSurface = cleaner.GetOutput()

        # But will operate on this one
        self.Surface = cleaner.GetOutput()

        # Will operate on the triangulated one
        triangulate = vtk.vtkTriangleFilter()
        triangulate.SetInputData(self.Surface)
        triangulate.Update()

        self.Surface = triangulate.GetOutput()

        selectAneurysm = vmtkscripts.vmtkSurfaceRegionDrawing()
        selectAneurysm.Surface = self.Surface

        if self.UniformStiffness:
            selectAneurysm.OutsideValue = self.ArteriesStiffness
            selectAneurysm.InsideValue = self.AneurysmStiffness
            selectAneurysm.Binary = True
            selectAneurysm.ContourScalarsArrayName = self.StiffnessArrayName
            selectAneurysm.Execute()

            self.Surface = self._smooth_array(selectAneurysm.Surface, 
                                              self.StiffnessArrayName)

        else:
            selectAneurysm.OutsideValue = 0.0
            selectAneurysm.Binary = False
            selectAneurysm.ContourScalarsArrayName = self.DistanceArrayName
            selectAneurysm.Execute()

            self.Surface = self._smooth_array(selectAneurysm.Surface, 
                                              self.DistanceArrayName)

            npDistanceSurface = dsa.WrapDataObject(self.Surface)

            # Note that the distances computed above are negative
            distanceArray = -npDistanceSurface.GetPointData().GetArray(self.DistanceArrayName)

            angCoeff = (neckStiffness  - fundusStiffness)/max(distanceArray)

            # Compute stiffness array by linear relationship with distance
            stiffnessArray = neckStiffness - angCoeff*distanceArray

            npDistanceSurface.PointData.append(
                stiffnessArray, 
                self.StiffnessArrayName
            )
            
            npDistanceSurface.PointData.append(
                distanceArray, 
                self.DistanceArrayName
            )

            self.Surface = npDistanceSurface.VTKObject

        # Map final thickness field to original surface
        surfaceProjection = vtkvmtk.vtkvmtkSurfaceProjection()
        surfaceProjection.SetInputData(polygonalSurface)
        surfaceProjection.SetReferenceSurface(self.Surface)
        surfaceProjection.Update()

        self.Surface = surfaceProjection.GetOutput()


if __name__ == '__main__':
    main = pypes.pypeMain()
    main.Arguments = sys.argv
    main.Execute()
