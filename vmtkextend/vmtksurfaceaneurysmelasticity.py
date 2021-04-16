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

vmtksurfaceaneurysmelasticity = 'vmtkSurfaceAneurysmelasticity'

class vmtkSurfaceAneurysmElasticity(pypes.pypeScript):

    def __init__(self):
        pypes.pypeScript.__init__(self)

        self.Surface = None
        self.UniformElasticity = False
        self.ElasticityArrayName = "E"
        self.DistanceToNeckArrayName = 'DistanceToNeck'

        self.ArteriesElasticity = 5e6
        self.AneurysmElasticity = 2e6

        self.SelectAneurysmRegions = False
        self.LocalScaleFactor = 0.75
        self.OnlyUpdateElasticity = False

        self.AbnormalHemodynamicsRegions = False
        self.WallTypeArrayName = "WallType"
        self.AtheroscleroticFactor = 1.20
        self.RedRegionsFactor = 0.95

        self.vmtkRenderer = None
        self.OwnRenderer = 0
        self.ContourWidget = None
        self.Actor = None
        self.Interpolator = None

        self.SetScriptName('vmtksurfaceaneurysmelasticity')
        self.SetScriptDoc('')

        self.SetInputMembers([
            ['Surface', 'i', 'vtkPolyData', 1, '',
                'the input surface', 'vmtksurfacereader'],

            ['UniformElasticity', 'uniformelasticity', 'bool', 1, '',
                'indicates uniform aneurysm elasticity'],

            ['ElasticityArrayName', 'elasticityarray', 'str', 1, '',
                'name of the resulting elasticity array'],

            ['ArteriesElasticity', 'arterieselasticity', 'float', 1, '',
                'elasticity of the arteries (and aneurysm neck)'],

            ['AneurysmElasticity', 'aneurysmelasticity', 'float', 1, '',
                'aneurysm elasticity (also aneurysm fundus elasticity)'],

            ['SelectAneurysmRegions', 'aneurysmregions', 'bool', 1, '',
                'enable selection of aneurysm less stiff or stiffer regions'],

            ['LocalScaleFactor', 'localfactor', 'float', 1, '',
                'scale fator to control local aneurysm elasticity'],

            ['OnlyUpdateElasticity', 'updateelasticity', 'bool', 1, '',
                'if the elasticity array already exists, this options enables '\
                'only to update it'],

            ['AbnormalHemodynamicsRegions', 'abnormalregions', 'bool', 1, '',
                'enable update on elasticity based on WallType array created '\
                'based on hemodynamics variables (must be used with '\
                'OnlyUpdateElasticity on)'],

            ['AtheroscleroticFactor', 'atheroscleroticfactor', 'float', 1, '',
                'scale fator to update elasticity of atherosclerotic regions '\
                'if AbnormalHemodynamicsRegions is true'],

            ['RedRegionsFactor', 'redregionsfactor', 'float', 1, '',
                'scale fator to update elasticity of red regions '\
                'if AbnormalHemodynamicsRegions is true'],

            ['WallTypeArrayName', 'walltypearray', 'str', 1, '',
                'name of wall type characterization array']

        ])

        self.SetOutputMembers([
            ['Surface', 'o', 'vtkPolyData', 1, '',
                'the input surface with elasticity array', 'vmtksurfacewriter'],
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

    def _smooth_array(self, surface, array, niterations=10, relax_factor=1.0):
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

    def _set_patch_elasticity(self, obj):
        """Set elasticity on selected region."""

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
        elasticityArray = self.Surface.GetPointData().GetArray(
                            self.ElasticityArrayName
                        )

        # TODO: how to select local scale factor on the fly?
        # queryString = 'Enter scale factor: '
        # self.LocalScaleFactor = int(self.InputText(queryString))
        # print(self.LocalScaleFactor)

        # multiply elasticity by scale factor in inside regions
        # where selection value is < 0.0, for this case
        for i in range(elasticityArray.GetNumberOfTuples()):
            selectionValue = selectionScalars.GetTuple1(i)
            elasticityValue = elasticityArray.GetTuple1(i)

            if selectionValue < 0.0:
                newElasticity = self.LocalScaleFactor*elasticityValue
                elasticityArray.SetTuple1(i, newElasticity)

        self.Actor.GetMapper().SetScalarRange(elasticityArray.GetRange(0))
        self.Surface.Modified()
        self.ContourWidget.Initialize()

    def SelectPatchElasticity(self):
        """Interactvely select patches of different elasticity."""

        # Initialize renderer
        if not self.OwnRenderer:
            self.vmtkRenderer = vmtkrenderer.vmtkRenderer()
            self.vmtkRenderer.Initialize()
            self.OwnRenderer = 1

        self.Surface.GetPointData().SetActiveScalars(self.ElasticityArrayName)

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
            'Update elasticity',
            self._set_patch_elasticity
        )

        self.vmtkRenderer.AddKeyBinding(
            'd',
            'Delete contour',
            self._delete_contour
        )

        self.vmtkRenderer.InputInfo(
            'Select regions to update elasticity\n'  \
            'Current local scale factor: '+         \
            str(self.LocalScaleFactor)+'\n'
        )

        # Update range for lengend
        elasticityArray = self.Surface.GetPointData().GetArray(
                            self.ElasticityArrayName
                        )

        self.Actor.GetMapper().SetScalarRange(elasticityArray.GetRange(0))
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
            self.ScalarBarActor.SetTitle(self.ElasticityArrayName)
            self.vmtkRenderer.Renderer.AddActor(self.ScalarBarActor)

        self._display()

        if self.OwnRenderer:
            self.vmtkRenderer.Deallocate()
            self.OwnRenderer = 0

    def UpdateAbnormalHemodynamicsRegions(self):
        """Based on wall type array, increase or deacrease elasticity.

        With a global elasticity array already defined on the surface, update
        the elasticity based on the wall type array created based on the
        hemodynamics variables, by multiplying it by a factor defined below. As
        explained in the function WallTypeCharacterization of wallmotion.py,
        the three types of wall and the operation performed here for each are:

            Label   Wall Type       Operation
            -----   ---------       ---------
                0   Normal wall     Nothing (default = 1)
                1   Atherosclerotic Increase elasticity (default factor = 1.15)
                2   "Red" wall      Decrease elasticity (default factor = 0.95)

        The multiplying factors for the atherosclerotic and red wall must be
        provided at object instantiation, with default values given above.
        The function will look for the array named "WallType" for defining
        its operation.

        Note also that, although our indication with this description does
        indicate that atherosclerotic regions are stiffer, this may not be true
        hence the user is free to input an atherosclerotic scale factor
        smaller than 1 to this case. The same comment is valid for the red
        wall cases.
        """
        # Labels for wall classification
        normalWall  = 0
        stifferWall = 1
        lessStiffWall = 2

        # Update both fields with selection
        elasticityArray = self.Surface.GetPointData().GetArray(
                             self.ElasticityArrayName
                         )

        # factor array: name WallType
        wallTypeArray = self.Surface.GetCellData().GetArray(
                            self.WallTypeArrayName
                        )

        # Update WallType array with scale factor
        # this is important to have a smooth field to multiply with the
        # Elasticity array (scale factor can be viewed as a continous
        # distribution in contrast to the WallType array that is discrete)
        for i in range(wallTypeArray.GetNumberOfTuples()):
            wallTypeValue  = wallTypeArray.GetTuple1(i)

            if wallTypeValue == stifferWall:
                newValue = self.AtheroscleroticFactor

            elif wallTypeValue == lessStiffWall:
                newValue = self.RedRegionsFactor

            else:
                newValue = 1.0

            wallTypeArray.SetTuple1(i, newValue)

        # Interpolate WallType cell data to point data
        cellDataToPointData = vtk.vtkCellDataToPointData()
        cellDataToPointData.SetInputData(self.Surface)
        cellDataToPointData.PassCellDataOff()
        cellDataToPointData.Update()

        self.Surface = cellDataToPointData.GetOutput()

        wallTypeArray = self.Surface.GetPointData().GetArray(
                            self.WallTypeArrayName
                        )

        # multiply elasticity by scale factor
        for i in range(elasticityArray.GetNumberOfTuples()):
            wallTypeValue  = wallTypeArray.GetTuple1(i)
            elasticityValue = elasticityArray.GetTuple1(i)

            elasticityArray.SetTuple1(i, wallTypeValue*elasticityValue)

        # Update name of abnormal regions factor final array
        wallTypeArray.SetName("AbnormalFactorArray")


    def Execute(self):

        if self.Surface == None:
            self.PrintError('Error: no Surface.')

        # Fundus and neck elasticity
        fundusElasticity = self.AneurysmElasticity
        neckElasticity = self.ArteriesElasticity

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

        # Two first branches assume the E field to exist already
        if self.OnlyUpdateElasticity and not self.AbnormalHemodynamicsRegions:
            self.SelectPatchElasticity()

        elif self.OnlyUpdateElasticity and self.AbnormalHemodynamicsRegions:
            self.UpdateAbnormalHemodynamicsRegions()

        # The E field does not exist
        else:
            # Check whether the DistanceToNeck array already exists
            nPointArrays = self.Surface.GetPointData().GetNumberOfArrays()

            pointArrays = [self.Surface.GetPointData().GetArray(id_).GetName()
                           for id_ in range(nPointArrays)]


            if self.DistanceToNeckArrayName not in pointArrays:
                selectAneurysm = vmtkscripts.vmtkSurfaceRegionDrawing()
                selectAneurysm.Surface = self.Surface
                selectAneurysm.OutsideValue = 1.0 # positive to identify artery
                selectAneurysm.Binary = False
                selectAneurysm.ContourScalarsArrayName = self.DistanceToNeckArrayName
                selectAneurysm.Execute()

                self.Surface = self._smooth_array(selectAneurysm.Surface,
                                                  self.DistanceToNeckArrayName)

            # Build elasticity on aneurysm based on DistanceToNeck array
            npDistanceSurface = dsa.WrapDataObject(self.Surface)

            # Note that the distances computed 'inside' the aneurysm
            # are negative originally, so we invert the sign to get
            # positive values inside the aneurysm
            distanceArray = -npDistanceSurface.GetPointData().GetArray(
                                self.DistanceToNeckArrayName
                            )

            elasticities = []

            if self.UniformElasticity:
                elasticities = [self.AneurysmElasticity
                                if distance >= 0.0
                                else self.ArteriesElasticity
                                for distance in distanceArray]

            else: # linear varying aneurysm elasticity

                # Angular coeff. for linear elasticity on the aneurysm sac
                angCoeff = (neckElasticity - fundusElasticity)/max(distanceArray)

                elasticities = [neckElasticity - angCoeff*distance
                                if distance >= 0.0
                                else neckElasticity
                                for distance in distanceArray]

            npDistanceSurface.PointData.append(
                dsa.VTKArray(elasticities),
                self.ElasticityArrayName
            )

            self.Surface = npDistanceSurface.VTKObject

        self.Surface = self._smooth_array(self.Surface,
                                          self.ElasticityArrayName)

        # Map final elasticity field to original surface
        surfaceProjection = vtkvmtk.vtkvmtkSurfaceProjection()
        surfaceProjection.SetInputData(polygonalSurface)
        surfaceProjection.SetReferenceSurface(self.Surface)
        surfaceProjection.Update()

        self.Surface = surfaceProjection.GetOutput()


if __name__ == '__main__':
    main = pypes.pypeMain()
    main.Arguments = sys.argv
    main.Execute()
