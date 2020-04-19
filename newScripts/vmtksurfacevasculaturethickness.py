#! /usr/bin/env python

# Creating Python Class to handle aneurysms operations
# My idea is to expand this class in the future:
# extract the aneyurysm and calculate other geometric parameters
import sys
import vtk
import math

from vmtk import vtkvmtk
from vmtk import vmtkscripts
from vmtk import vmtkrenderer
from vmtk import pypes


vmtksurfacevasculaturethickness = 'vmtkSurfaceVasculatureThickness'


class vmtkSurfaceVasculatureThickness(pypes.pypeScript):

    _SMALL = 1e-12

    # Constructor
    def __init__(self):
        pypes.pypeScript.__init__(self)

        # "Private" members
        self._wlrMedium = 0.07
        self._wlrLarge = 0.088

        self._centerlines = None
        self._radiusArrayName = "RadiusArray"

        # Public member
        self.Surface  = None

        self.ManualMode = True
        self.ScaleFactor = 0.75
        self.WallLumenRatio = self._wlrMedium
        self.ThicknessArrayName = 'Thickness'
        self.SmoothingIterations = 10

        self.GenerateWallMesh = False
        self.WallMesh = None
        self.WallMeshLayers = 3

        self.vmtkRenderer = None
        self.OwnRenderer = 0
        self.ContourWidget = None

        self.SetScriptName('vmtksurfacevasculaturethickness')
        self.SetScriptDoc('')

        self.SetInputMembers([
            ['Surface', 'i', 'vtkPolyData', 1, '',
                'the input surface', 'vmtksurfacereader'],

            ['ScaleFactor', 'scalefactor', 'float', 1, '',
                'scale fator to control size of aneurysm thickness'],

            ['ManualMode', 'manual', 'bool', 1, '',
                'enable manual mode to select centerline endpoints'],

            ['WallLumenRatio', 'wlr', 'float', 1, '',
                'wall to lumen ration'],

            ['ThicknessArrayName', 'thicknessarray', 'str', 1, '',
                'name of the resulting thickness array'],

            ['SmoothingIterations', 'iterations', 'int', 1, '',
                'number of iterations for array smoothing'],

            ['GenerateWallMesh', 'wallmesh', 'bool', 1, '',
                'automatically extrude wall mesh with thickness array'],

            ['WallMeshLayers', 'layers', 'int', 1, '',
                'the number of layers to extrude the wall mesh'],
        ])

        self.SetOutputMembers([
            ['Surface', 'o', 'vtkPolyData', 1, '',
                'the input surface with thickness array', 'vmtksurfacewriter'],

            ['WallMesh', 'owallmesh', 'vtkUnstructuredGrid', 1, '',
                'the output wall mesh', 'vmtkmeshwriter'],
        ])

    def _delete_contour(self, obj):
        self.ContourWidget.Initialize()

    def _interact(self, obj):
        if self.ContourWidget.GetEnabled() == 1:
            self.ContourWidget.SetEnabled(0)
        else:
            self.ContourWidget.SetEnabled(1)

    def _display(self):
        self.vmtkRenderer.Render()

    def _smooth_array(self, surface, array, niterations=5, relax_factor=1.0):
        """Surface array smoother."""

#         arraySmoother = vmtkscripts.vmtkSurfaceArraySmoothing()
#         arraySmoother.Surface = surface
#         arraySmoother.SurfaceArrayName = array
#         arraySmoother.Connexity = 1
#         arraySmoother.Relaxation = 1.0
#         arraySmoother.Iterations = niterations
#         arraySmoother.Execute()
#
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

                newValue = relax_factor*weightedValue
                + (1.0 - relax_factor)*currVal

                array.SetTuple1(i, newValue)

        return surface

    def _generate_eneterlines(self):
        centerlines = vmtkscripts.vmtkCenterlines()
        centerlines.Surface = self.Surface
        centerlines.SeedSelectorName = 'openprofiles'
        centerlines.AppendEndPoints = 1
        centerlines.RadiusArrayName = self._radiusArrayName
        centerlines.Execute()

        self._centerlines = centerlines.Centerlines

    def ComputeVasculatureThickness(self):
        """Compute thickness array based on diameter and WLR.

        Given input surface with the radius array, 
        computes the thickness by multiplying by 
        the wall-to-lumen ration. The aneurysm portion
        is also multiplyed.
        """

        # Compute centerlines
        self._generate_eneterlines()

        # Compute distance to centerlines
        distanceToCenterlines = vmtkscripts.vmtkDistanceToCenterlines()
        distanceToCenterlines.Surface = self.Surface
        distanceToCenterlines.Centerlines = self._centerlines
        distanceToCenterlines.UseRadiusInformation = 1
#         distanceToCenterlines.UseCombinedDistance = 1
        distanceToCenterlines.RadiusArrayName = self._radiusArrayName
        distanceToCenterlines.Execute()

        surface = distanceToCenterlines.Surface
        distanceArrayName = distanceToCenterlines.DistanceToCenterlinesArrayName

        # Smooth the distance to centerline array
        # to avoid sudden changes of thickness in
        # certain regions
        surface = self._smooth_array(surface, distanceArrayName)

        # Multiply by WLR to have a prelimimar thickness array
        # I assume that the WLR is the same for medium sized arteries
        # but I can change this in a point-wise manner based on
        # the local radius array by using the algorithm contained
        # in the vmtksurfacearrayoperation script
        wlrFactor = vmtkscripts.vmtkSurfaceArrayOperation()
        wlrFactor.Surface = surface
        wlrFactor.Operation = 'multiplybyc'
        wlrFactor.InputArrayName = distanceArrayName
        wlrFactor.Constant = 2.0*self.WallLumenRatio
        wlrFactor.ResultArrayName = self.ThicknessArrayName
        wlrFactor.Execute()

        wlrFactor.Surface.GetPointData().RemoveArray(distanceArrayName)

        self.Surface = wlrFactor.Surface

    def SetAneurysmThickness(self):
        """Calculate and set aneurysm thickness.

        Based on the vasculature thickness distribution,
        defined as the outside portion of the complete
        geometry from a neck drawn by the user, estimates 
        an aneurysm thickness by averaging the vasculature
        thickness using as weight function the distance to
        the neck line. The estimated aneurysm thickness is,
        then, set on the aneurysm surface in the thickness
        array."""

        # Save reference to original polySurface
        polySurface = self.Surface

        # Triangulate surface
        # (contour selection only accepts triangulated surface)
        triangleFilter = vtk.vtkTriangleFilter()
        triangleFilter.SetInputData(self.Surface)
        triangleFilter.Update()
        self.Surface = triangleFilter.GetOutput()

        # Create mapper for the surface
        mapper = vtk.vtkPolyDataMapper()
        mapper.SetInputData(self.Surface)
        mapper.ScalarVisibilityOn()

        # Add surface as an actor to the scene
        Actor = vtk.vtkActor()
        Actor.SetMapper(mapper)
        Actor.GetMapper().SetScalarRange(-1.0, 0.0)

        # Add the surface actor to the renderer
        self.vmtkRenderer.Renderer.AddActor(Actor)

        # Create contour widget
        self.ContourWidget = vtk.vtkContourWidget()
        self.ContourWidget.SetInteractor(
            self.vmtkRenderer.RenderWindowInteractor)

        rep = vtk.vtkOrientedGlyphContourRepresentation.SafeDownCast(
            self.ContourWidget.GetRepresentation()
        )

        rep.GetLinesProperty().SetColor(1, 0.2, 0)
        rep.GetLinesProperty().SetLineWidth(3.0)

        pointPlacer = vtk.vtkPolygonalSurfacePointPlacer()
        pointPlacer.AddProp(Actor)
        pointPlacer.GetPolys().AddItem(self.Surface)

        rep.SetPointPlacer(pointPlacer)

        Interpolator = vtk.vtkPolygonalSurfaceContourLineInterpolator()
        Interpolator.GetPolys().AddItem(self.Surface)
        rep.SetLineInterpolator(Interpolator)

        self.vmtkRenderer.AddKeyBinding('i', 'Start interaction',
                                        self._interact)

        self.vmtkRenderer.AddKeyBinding('d', 'Delete contour',
                                        self._delete_contour)

        self._display()
        self.vmtkRenderer.Deallocate()

        # Get loop points from representation to vtkPoints
        pointIds = vtk.vtkIdList()
        Interpolator.GetContourPointIds(rep, pointIds)

        points = vtk.vtkPoints()
        points.SetNumberOfPoints(pointIds.GetNumberOfIds())

        # Get points in surface
        for i in range(pointIds.GetNumberOfIds()):
            pointId = pointIds.GetId(i)
            point = self.Surface.GetPoint(pointId)
            points.SetPoint(i, point)

        # Get array of surface selection based on loop points
        selectionFilter = vtk.vtkSelectPolyData()
        selectionFilter.SetInputData(self.Surface)
        selectionFilter.SetLoop(points)
        selectionFilter.GenerateSelectionScalarsOn()
        selectionFilter.SetSelectionModeToSmallestRegion()
        selectionFilter.Update()

        # Get selection scalars
        selectionScalars = selectionFilter.GetOutput().GetPointData().GetScalars()

        # Update both fields with selection
        thicknessArray = self.Surface.GetPointData().GetArray(self.ThicknessArrayName)

        # Compute aneurysm thickness
        aneurysmThickness = 0.0
        normFactor = 0.0

        _SMALL = 1e-12
        # First compute aneurysm thickness based on vasculature thickness
        # the vasculature is selection value > 0
        for i in range(thicknessArray.GetNumberOfTuples()):
            selectionValue = selectionScalars.GetTuple1(i)
            thicknessValue = thicknessArray.GetTuple1(i)

            if selectionValue > 0.0:
                weight = 1.0/(selectionValue + _SMALL)

                aneurysmThickness += weight*thicknessValue
                normFactor += weight

        aneurysmThickness = self.ScaleFactor*aneurysmThickness/normFactor

        # Then, substitute thickness array by aneurysmThickness
        for i in range(thicknessArray.GetNumberOfTuples()):
            selectionValue = selectionScalars.GetTuple1(i)

            if selectionValue < 0.0:
                # Set new aneurysm thickness
                thicknessArray.SetTuple1(i, aneurysmThickness)

        # Project fields back to original surface
        surfaceProjection = vtkvmtk.vtkvmtkSurfaceProjection()
        surfaceProjection.SetInputData(polySurface)
        surfaceProjection.SetReferenceSurface(self.Surface)
        surfaceProjection.Update()

        self.Surface = self._smooth_array(surfaceProjection.GetOutput(),
                                          self.ThicknessArrayName,
                                          niterations=self.SmoothingIterations)


    def ExtrudeWallMesh(self):
        """Extrude wall along normals with thickness array."""

        normals = vmtkscripts.vmtkSurfaceNormals()
        normals.Surface = self.Surface 
        normals.FlipNormals = 0
        normals.Execute()

        surfaceToMesh = vmtkscripts.vmtkSurfaceToMesh()
        surfaceToMesh.Surface = normals.Surface
        surfaceToMesh.Execute()

        wallMesh = vmtkscripts.vmtkBoundaryLayer()
        wallMesh.Mesh = surfaceToMesh.Mesh
        wallMesh.WarpVectorsArrayName = normals.NormalsArrayName 
        wallMesh.ThicknessArrayName = self.ThicknessArrayName
        wallMesh.ThicknessRatio = 1
        wallMesh.NumberOfSubLayers = self.WallMeshLayers

        # wallMesh.UseWarpVectorMagnitudeAsThickness = 1
        # wallMesh.Thickness = 0.2
        # wallMesh.ConstantThickness = 1

        # Setup
        wallMesh.NumberOfSubsteps = 7000
        wallMesh.Relaxation = 0.01
        wallMesh.LocalCorrectionFactor = 0.5

        # Entity ids for new mesh
        wallMesh.VolumeCellEntityId = 0
        wallMesh.InnerSurfaceCellEntityId = 1
        wallMesh.OuterSurfaceCellEntityId = 2
        wallMesh.SidewallCellEntityId = 3

        wallMesh.IncludeSidewallCells = 0
        wallMesh.IncludeSurfaceCells = 0
        wallMesh.NegateWarpVectors = 0
        wallMesh.Execute()

        self.WallMesh = wallMesh.Mesh


    def Execute(self):

        if self.Surface == None:
            self.PrintError('Error: no Surface.')

        # Initialize renderer
        if not self.vmtkRenderer:
            self.vmtkRenderer = vmtkrenderer.vmtkRenderer()
            self.vmtkRenderer.Initialize()
            self.OwnRenderer = 1

        self.vmtkRenderer.RegisterScript(self)

        self.ComputeVasculatureThickness()
        self.SetAneurysmThickness()

        if self.GenerateWallMesh:
            self.ExtrudeWallMesh()


if __name__ == '__main__':
    main = pypes.pypeMain()
    main.Arguments = sys.argv
    main.Execute()
