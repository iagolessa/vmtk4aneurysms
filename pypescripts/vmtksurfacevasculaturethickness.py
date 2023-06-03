#! /usr/bin/env python3

# Copyright (C) 2022, Iago L. de Oliveira

# vmtk4aneurysms is free software: you can redistribute it and/or modify
# it under the terms of the GNU General Public License as published by
# the Free Software Foundation, either version 3 of the License, or
# (at your option) any later version.

# This program is distributed in the hope that it will be useful,
# but WITHOUT ANY WARRANTY; without even the implied warranty of
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
# GNU General Public License for more details.

# You should have received a copy of the GNU General Public License
# along with this program.  If not, see <http://www.gnu.org/licenses/>.


import re
import sys
import vtk
import numpy as np
import vtk.numpy_interface.dataset_adapter as dsa

from vmtk import vtkvmtk
from vmtk import vmtkscripts
from vmtk import vmtkrenderer
from vmtk import pypes

from vmtk4aneurysms import vascular_operations as vscop
from vmtk4aneurysms.lib import polydatatools as tools
from vmtk4aneurysms.lib import centerlines as cl
from vmtk4aneurysms.lib import names
from vmtk4aneurysms.lib import constants as const

vmtksurfacevasculaturethickness = 'vmtkSurfaceVasculatureThickness'

class vmtkSurfaceVasculatureThickness(pypes.pypeScript):

    _SMALL = 1e-12

    # Constructor
    def __init__(self):
        pypes.pypeScript.__init__(self)

        # Public member
        self.Surface = None
        self.Centerlines = None
        self.Aneurysm = True
        # self.NumberOfAneurysms = 1
        self.AneurysmType = None # in case only 1 aneurysm
        self.ParentVesselSurface = None
        self.DomePoint = []

        # Fields that are created or edited in this script
        # - User-modifiable
        self.DistanceToNeckArrayName = names.DistanceToNeckArrayName
        self.ThicknessArrayName = names.ThicknessArrayName

        # Non-modifiable by user
        self.RadiusArrayName = names.VascularRadiusArrayName
        self.AbnormalFactorArrayName = "AbnormalFactorArray"

        # Vasculature thickness parameters
        self.UniformWallToLumenRatio = False
        self.WallLumenRatio = const.WlrMedium
        self.SmoothingIterations = 10

        # Aneurysm thickness parameters
        self.NeckComputationMode = "interactive"
        self.AneurysmInfluencedRegionDistance = 0.5
        self.GlobalScaleFactor = 0.75

        self.SelectAneurysmRegions = False
        self.LocalScaleFactor = 0.75
        self.OnlyUpdateThickness = False

        self.AbnormalHemodynamicsRegions = False
        self.WallTypeArrayName = names.WallTypeArrayName
        self.AtheroscleroticFactor = 1.20
        self.RedRegionsFactor = 0.95

        self.GenerateWallMesh = False
        self.WallMesh = None
        self.WallMeshLayers = 3

        self.vmtkRenderer = None
        self.OwnRenderer = 0
        self.ContourWidget = None
        self.Actor = None
        self.Interpolator = None

        self.SetScriptName('vmtksurfacevasculaturethickness')
        self.SetScriptDoc('')

        self.SetInputMembers([
            ['Surface', 'i', 'vtkPolyData', 1, '',
                'the input surface', 'vmtksurfacereader'],

            ['Centerlines', 'icenterline', 'vtkPolyData', 1, '',
                'the centerlines of the input surface (optional; if not '\
                'passed, it is calculated automatically', 'vmtksurfacereader'],

            ['RadiusArrayName', 'radiusarray', 'str', 1, '',
                'centerline radius array name, if loaded externally'],

            ['Aneurysm', 'aneurysm', 'bool', 1, '',
                'to indicate presence of an aneurysm'],

            # ['NumberOfAneurysms', 'naneurysms', 'int', 1, '',
                # 'integer with number of aneurysms on vasculature'],

            ['AneurysmType','aneurysmtype', 'str' , 1,
                '["lateral","bifurcation"]',
                'if only one aneurysm, pass also its type'],

            ['ParentVesselSurface', 'iparentvessel', 'vtkPolyData', 1, '',
                'the parent vessel surface (if not passed, computed externally)',
                'vmtksurfacereader'],

            ['DomePoint','domepoint', 'float', -1, '',
                'coordinates of aneurysm dome point'],

            ['NeckComputationMode','neckcomputationmode', 'str' , 1,
                '["interactive","automatic"]',
                'if the neck array is not in the surface, compute it using '\
                'one of these methods'],

            ['AneurysmInfluencedRegionDistance', 'influencedistance',
                'float', 1, '',
                'distance (in mm) that controls how far the '\
                'aneurysm-influenced region goes from the aneurysm neck line'],

            ['GlobalScaleFactor', 'globalfactor', 'float', 1, '',
                'scale fator to control global aneurysm thickness'],

            ['UniformWallToLumenRatio', 'uniformwlr', 'bool', 1, '',
                'forces a uniform WLR informed by the user'],

            ['WallLumenRatio', 'wlr', 'float', 1, '',
                'wall to lumen ratio if uniformwlr is active'],

            ['ThicknessArrayName', 'thicknessarray', 'str', 1, '',
                'name of the resulting thickness array'],

            ['SmoothingIterations', 'iterations', 'int', 1, '',
                'number of iterations for array smoothing'],

            ['GenerateWallMesh', 'wallmesh', 'bool', 1, '',
                'automatically extrude wall mesh with thickness array'],

            ['WallMeshLayers', 'layers', 'int', 1, '',
                'the number of layers to extrude the wall mesh'],

            ['SelectAneurysmRegions', 'aneurysmregions', 'bool', 1, '',
                'enable selection of aneurysm thinner or thicker regions'],

            ['LocalScaleFactor', 'localfactor', 'float', 1, '',
                'scale fator to control local aneurysm thickness'],

            ['OnlyUpdateThickness', 'updatethickness', 'bool', 1, '',
                'if the thickness array already exists, this options enables '\
                'only to update it through manual selection or automatically'],

            ['AbnormalHemodynamicsRegions', 'abnormalregions', 'bool', 1, '',
                'enable update on thickness based on WallType array created '\
                'based on hemodynamics variables (must be used with '\
                'OnlyUpdateThickness on)'],

            ['AtheroscleroticFactor', 'atheroscleroticfactor', 'float', 1, '',
                'scale fator to update thickness of atherosclerotic regions '\
                'if AbnormalHemodynamicsRegions is true'],

            ['RedRegionsFactor', 'redregionsfactor', 'float', 1, '',
                'scale fator to update thickness of red regions '\
                'if AbnormalHemodynamicsRegions is true'],

            ['WallTypeArrayName', 'walltypearray', 'str', 1, '',
                'name of wall type characterization array']
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

    def _set_thinner_thickness(self, obj):
        """Set thinner thickness on selected region."""

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
        thicknessArray = self.Surface.GetPointData().GetArray(
            self.ThicknessArrayName
        )

        # TODO: how to select local scale factor on the fly?
        # queryString = 'Enter scale factor: '
        # self.LocalScaleFactor = int(self.InputText(queryString))
        # print(self.LocalScaleFactor)

        # multiply thickness by scale factor in inside regions
        # where selection value is < 0.0, for this case
        for i in range(thicknessArray.GetNumberOfTuples()):
            selectionValue = selectionScalars.GetTuple1(i)
            thicknessValue = thicknessArray.GetTuple1(i)

            if selectionValue < 0.0:
                thinnerThickness = self.LocalScaleFactor*thicknessValue
                thicknessArray.SetTuple1(i, thinnerThickness)

        self.Actor.GetMapper().SetScalarRange(thicknessArray.GetRange(0))
        self.Surface.Modified()
        self.ContourWidget.Initialize()

    def AneurysmTypeValidator(self, iaType):
        if iaType == 'lateral' or iaType == 'bifurcation':
            return 1

        else:
            return 0

        # if text == 'i':
        #     self.vmtkRenderer.Render()
        #     return 0

        # try:
        #     float(text)
        # except ValueError:
        #     return 0
        # return 1

    def SelectThinnerRegions(self):
        """Interactvely select thinner regions of the aneurysm."""

        # Initialize renderer
        if not self.OwnRenderer:
            self.vmtkRenderer = vmtkrenderer.vmtkRenderer()
            self.vmtkRenderer.Initialize()
            self.OwnRenderer = 1

        self.Surface.GetPointData().SetActiveScalars(self.ThicknessArrayName)

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
           'Update thickness',
           self._set_thinner_thickness
        )

        self.vmtkRenderer.AddKeyBinding(
            'd',
            'Delete contour',
            self._delete_contour
        )

        self.vmtkRenderer.InputInfo(
            'Select regions to update thickness\n'  \
            'Current local scale factor: '+         \
            str(self.LocalScaleFactor)+'\n'
        )

        # Update range for lengend
        thicknessArray = self.Surface.GetPointData().GetArray(
                            self.ThicknessArrayName
                        )

        self.Actor.GetMapper().SetScalarRange(thicknessArray.GetRange(0))
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
            self.ScalarBarActor.SetLabelFormat('%.2f')
            self.ScalarBarActor.SetTitle(self.ThicknessArrayName)
            self.vmtkRenderer.Renderer.AddActor(self.ScalarBarActor)

        self._display()

        if self.OwnRenderer:
            self.vmtkRenderer.Deallocate()
            self.OwnRenderer = 0

    def UpdateAbnormalHemodynamicsRegions(self):
        """Based on wall type array, increase or deacrease thickness.

        With a global thickness array already defined on the surface, update
        the thickness based on the wall type array created based on the
        hemodynamics variables, by multiplying it by a factor defined below. As
        explained in the function WallTypeCharacterization of wallmotion.py,
        the three types of wall and the operation performed here for each are:

        .. table:: Local wall type characterization
            :widths: auto

            =====   =============== =========
            Label   Wall Type       Operation
            =====   =============== =========
                0   Normal wall     Nothing (default = 1)
                1   Atherosclerotic Increase elasticity (default factor = 1.20)
                2   "Red" wall      Decrease elasticity (default factor = 0.95)
            =====   =============== =========

        The multiplying factors for the atherosclerotic and red wall must be
        provided at object instantiation, with default values given above.
        The function will look for the array named "WallType" for defining
        its operation.
        """
        # Labels for wall classification
        normalWall  = 0
        thickerWall = 1
        thinnerWall = 2

        # Update both fields with selection
        thicknessArray = self.Surface.GetPointData().GetArray(
                             self.ThicknessArrayName
                         )

        # factor array: name WallType
        wallTypeArray = self.Surface.GetCellData().GetArray(
                            self.WallTypeArrayName
                        )

        # Update WallType array with scale factor
        # this is important to have a smooth field to multiply with the
        # thickness array (scale factor can be viewed as a continous
        # distribution in contrast to the WallType array that is discrete)
        for i in range(wallTypeArray.GetNumberOfTuples()):
            wallTypeValue  = wallTypeArray.GetTuple1(i)

            if wallTypeValue == thickerWall:
                newValue = self.AtheroscleroticFactor

            elif wallTypeValue == thinnerWall:
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

        # self.Surface.GetCellData().RemoveArray(self.WallTypeArrayName)

        wallTypeArray = self.Surface.GetPointData().GetArray(
                            self.WallTypeArrayName
                        )

        # multiply thickness by scale factor
        for i in range(thicknessArray.GetNumberOfTuples()):
            wallTypeValue  = wallTypeArray.GetTuple1(i)
            thicknessValue = thicknessArray.GetTuple1(i)

            thicknessArray.SetTuple1(i, wallTypeValue*thicknessValue)

        # Update name of abnormal regions factor final array
        wallTypeArray.SetName("AbnormalFactorArray")


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

        # Clean fields and cell arrays
        solidWallMesh = wallMesh.Mesh

        nFields     = solidWallMesh.GetFieldData().GetNumberOfArrays()
        nCellArrays = solidWallMesh.GetCellData().GetNumberOfArrays()

        for field in range(nFields):
            solidWallMesh.GetFieldData().RemoveArray(field)

        for array in range(nCellArrays):
            solidWallMesh.GetCellData().RemoveArray(array)

        self.WallMesh = solidWallMesh

    def Execute(self):

        if self.Surface == None:
            self.PrintError('Error: no Surface.')

        # # The aneurysm type mjst be informed here because ot may
        # # change with the case of multiple aneurysms
        # if (naneurysms != 1) or \
        #    (naneurysms == 1 and \
        #     aneurysm_type is None):

        #     aneurysmType = self.InputText(
        #                        "Type aneurysm type ['lateral','bifurcation']:",
        #                        aneurysm_typeValidator
        #                    )

        # else:
        #     aneurysmType = aneurysm_type

        # Store the point and cell array that were already on the surface

        origCellArrays  = tools.GetCellArrays(self.Surface)
        origPointArrays = tools.GetPointArrays(self.Surface)

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

        # Initialize renderer
        if not self.vmtkRenderer:
            self.vmtkRenderer = vmtkrenderer.vmtkRenderer()
            self.vmtkRenderer.Initialize()
            self.OwnRenderer = 1

        self.vmtkRenderer.RegisterScript(self)

        if self.OnlyUpdateThickness and not self.AbnormalHemodynamicsRegions:
            self.SelectThinnerRegions()

        elif self.OnlyUpdateThickness and self.AbnormalHemodynamicsRegions:
            self.UpdateAbnormalHemodynamicsRegions()

        else:
            if self.Aneurysm:
                self.Surface = vscop.ComputeVasculatureThicknessWithAneurysm(
                                   self.Surface,
                                   self.Centerlines,
                                   thickness_field_name=self.ThicknessArrayName,
                                   set_uniform_wlr=self.UniformWallToLumenRatio,
                                   uniform_wlr_value=self.WallLumenRatio,
                                   neck_comp_mode=self.NeckComputationMode,
                                   gdistance_to_neck_array_name=self.DistanceToNeckArrayName,
                                   aneurysm_type=self.AneurysmType,
                                   aneurysm_influence_dist=self.AneurysmInfluencedRegionDistance,
                                   scale_factor=self.GlobalScaleFactor,
                                   parent_vessel_surface=self.ParentVesselSurface,
                                   dome_point=self.DomePoint
                               )

                if self.OwnRenderer:
                    self.vmtkRenderer.Deallocate()
                    self.OwnRenderer = 0

                if self.SelectAneurysmRegions:
                    self.SelectThinnerRegions()

            else:
                self.Surface = vscop.ComputeVasculatureThickness(
                                  self.Surface,
                                  self.Centerlines,
                                  thickness_field_name=self.ThicknessArrayName,
                                  set_uniform_wlr=self.UniformWallToLumenRatio,
                                  uniform_wlr_value=self.WallLumenRatio
                               )

        # After array create, smooth it hard
        self.Surface = self._smooth_array(self.Surface,
                                          self.ThicknessArrayName,
                                          niterations=self.SmoothingIterations)

        # Map final thickness field to original surface
        surfaceProjection = vtkvmtk.vtkvmtkSurfaceProjection()
        surfaceProjection.SetInputData(polygonalSurface)
        surfaceProjection.SetReferenceSurface(self.Surface)
        surfaceProjection.Update()

        self.Surface = surfaceProjection.GetOutput()

        if self.GenerateWallMesh:
            self.ExtrudeWallMesh()

        if self.OwnRenderer:
            self.vmtkRenderer.Deallocate()


if __name__ == '__main__':
    main = pypes.pypeMain()
    main.Arguments = sys.argv
    main.Execute()
