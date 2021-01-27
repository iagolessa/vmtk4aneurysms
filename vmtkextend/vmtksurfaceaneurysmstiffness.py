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
from vmtk import vmtkrenderer
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

        self.SelectAneurysmRegions = False
        self.LocalScaleFactor = 0.75
        self.OnlyUpdateStiffness = False

        self.vmtkRenderer = None
        self.OwnRenderer = 0
        self.ContourWidget = None
        self.Actor = None
        self.Interpolator = None

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

            ['SelectAneurysmRegions', 'aneurysmregions', 'bool', 1, '',
                'enable selection of aneurysm thinner or thicker regions'],

            ['LocalScaleFactor', 'localfactor', 'float', 1, '',
                'scale fator to control local aneurysm thickness'],

            ['OnlyUpdateStiffness', 'updatestiffness', 'bool', 1, '',
                'if the stiffness array already exists, this options enables '\
                'only to update it'],
        ])

        self.SetOutputMembers([
            ['Surface', 'o', 'vtkPolyData', 1, '',
                'the input surface with thickness array', 'vmtksurfacewriter'],
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

    def _set_patch_stiffness(self, obj):
        """Set stiffness on selected region."""

        rep = vtk.vtkOrientedGlyphContourRepresentation.SafeDownCast(
            self.ContourWidget.GetRepresentation()
        )

        # Get loop points from representation to vtkPoints
        pointIds = vtk.vtkIdList()
        self.Interpolator.GetContourPointIds(rep, pointIds)

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
        stiffnessArray = self.Surface.GetPointData().GetArray(
                            self.StiffnessArrayName
                        )

        # TODO: how to select local scale factor on the fly?
        # queryString = 'Enter scale factor: '
        # self.LocalScaleFactor = int(self.InputText(queryString))
        # print(self.LocalScaleFactor)

        # multiply thickness by scale factor in inside regions
        # where selection value is < 0.0, for this case
        for i in range(stiffnessArray.GetNumberOfTuples()):
            selectionValue = selectionScalars.GetTuple1(i)
            stiffnessValue = stiffnessArray.GetTuple1(i)

            if selectionValue < 0.0:
                newStiffness = self.LocalScaleFactor*stiffnessValue
                stiffnessArray.SetTuple1(i, newStiffness)

        self.Actor.GetMapper().SetScalarRange(stiffnessArray.GetRange(0))
        self.Surface.Modified()
        self.ContourWidget.Initialize()

    def SelectPatchStiffness(self):
        """Interactvely select patches of different stiffness."""

        # Initialize renderer
        if not self.OwnRenderer:
            self.vmtkRenderer = vmtkrenderer.vmtkRenderer()
            self.vmtkRenderer.Initialize()
            self.OwnRenderer = 1

        self.Surface.GetPointData().SetActiveScalars(self.StiffnessArrayName)

        # Create mapper and actor to scene
        mapper = vtk.vtkPolyDataMapper()
        mapper.SetInputData(self.Surface)
        mapper.ScalarVisibilityOn()

        self.Actor = vtk.vtkActor()
        self.Actor.SetMapper(mapper)
        self.Actor.GetMapper().SetScalarRange(-1.0, 0.0)
        self.vmtkRenderer.Renderer.AddActor(self.Actor)

        # Create representation to draw contour
        self.ContourWidget = vtk.vtkContourWidget()
        self.ContourWidget.SetInteractor(
            self.vmtkRenderer.RenderWindowInteractor)

        rep = vtk.vtkOrientedGlyphContourRepresentation.SafeDownCast(
                self.ContourWidget.GetRepresentation()
            )

        rep.GetLinesProperty().SetColor(1, 0.2, 0)
        rep.GetLinesProperty().SetLineWidth(3.0)

        pointPlacer = vtk.vtkPolygonalSurfacePointPlacer()
        pointPlacer.AddProp(self.Actor)
        pointPlacer.GetPolys().AddItem(self.Surface)
        rep.SetPointPlacer(pointPlacer)

        self.Interpolator = vtk.vtkPolygonalSurfaceContourLineInterpolator()
        self.Interpolator.GetPolys().AddItem(self.Surface)
        rep.SetLineInterpolator(self.Interpolator)

        self.vmtkRenderer.AddKeyBinding(
            'i',
            'Start interaction: select region',
            self._interact
        )

        self.vmtkRenderer.AddKeyBinding(
            'space',
            'Update stiffness',
            self._set_patch_stiffness
        )

        self.vmtkRenderer.AddKeyBinding(
            'd',
            'Delete contour',
            self._delete_contour
        )

        self.vmtkRenderer.InputInfo(
            'Select regions to update stiffness\n'  \
            'Current local scale factor: '+         \
            str(self.LocalScaleFactor)+'\n'
        )

        # Update range for lengend
        stiffnessArray = self.Surface.GetPointData().GetArray(
                            self.StiffnessArrayName
                        )

        self.Actor.GetMapper().SetScalarRange(stiffnessArray.GetRange(0))
        self.Surface.Modified()

        self.Legend = 1
        if self.Legend and self.Actor:
            self.ScalarBarActor = vtk.vtkScalarBarActor()
            self.ScalarBarActor.SetLookupTable(
                self.Actor.GetMapper().GetLookupTable()
            )
            self.ScalarBarActor.GetLabelTextProperty().ItalicOff()
            self.ScalarBarActor.GetLabelTextProperty().BoldOff()
            self.ScalarBarActor.GetLabelTextProperty().ShadowOff()
            # self.ScalarBarActor.GetLabelTextProperty().SetColor(0.0,0.0,0.0)
            self.ScalarBarActor.SetLabelFormat('%.2f')
            self.ScalarBarActor.SetTitle(self.StiffnessArrayName)
            self.vmtkRenderer.Renderer.AddActor(self.ScalarBarActor)

        self._display()

        if self.OwnRenderer:
            self.vmtkRenderer.Deallocate()
            self.OwnRenderer = 0


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

        if not self.OnlyUpdateStiffness:

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
                distanceArray = -npDistanceSurface.GetPointData().GetArray(
                                    self.DistanceArrayName
                                )

                angCoeff = (neckStiffness - fundusStiffness)/max(distanceArray)

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

        else:
            self.SelectAneurysmRegions = True

        # Update with stiffer regions
        if self.SelectAneurysmRegions:
            self.SelectPatchStiffness()

            self.Surface = self._smooth_array(self.Surface,
                                              self.StiffnessArrayName)

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
