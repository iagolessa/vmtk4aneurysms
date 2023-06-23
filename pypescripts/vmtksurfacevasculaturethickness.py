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

import sys
import vtk

from vmtk import pypes
from vmtk import vmtkscripts
from vmtk import vmtkrenderer

from vmtk4aneurysms import vascular_operations as vscop
from vmtk4aneurysms.lib import polydatatools as tools
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

        # Vasculature thickness parameters
        self.UniformWallToLumenRatio = False
        self.WallLumenRatio = const.WlrMedium
        self.SmoothingIterations = 5

        # Aneurysm thickness parameters
        self.NeckComputationMode = "interactive"
        self.AneurysmInfluencedRegionDistance = 0.5
        self.GlobalScaleFactor = 0.75

        self.AbnormalHemodynamicsRegions = False
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

            ['AbnormalHemodynamicsRegions', 'abnormalregions', 'bool', 1, '',
                'enable update on thickness based on WallType array created '\
                'based on hemodynamics variables'],

            ['AtheroscleroticFactor', 'atheroscleroticfactor', 'float', 1, '',
                'scale fator to update thickness of atherosclerotic regions '\
                'if AbnormalHemodynamicsRegions is true'],

            ['RedRegionsFactor', 'redregionsfactor', 'float', 1, '',
                'scale fator to update thickness of red regions '\
                'if AbnormalHemodynamicsRegions is true'],
        ])

        self.SetOutputMembers([
            ['Surface', 'o', 'vtkPolyData', 1, '',
                'the input surface with thickness array', 'vmtksurfacewriter'],

            ['WallMesh', 'owallmesh', 'vtkUnstructuredGrid', 1, '',
                'the output wall mesh', 'vmtkmeshwriter'],
        ])

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

        # I had a bug with the 'select thinner regions' with polygonal meshes.
        # So, operate on a triangulated surface and map final result to orignal
        # surface
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

        # Compute the thickness field
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
                               parent_vascular_surface=self.ParentVesselSurface,
                               dome_point=self.DomePoint,
                               abnormal_thickness=self.AbnormalHemodynamicsRegions,
                               atherosclerotic_factor=self.AtheroscleroticFactor,
                               red_regions_factor=self.RedRegionsFactor,
                               nsmooth_iterations=self.SmoothingIterations
                           )

            if self.OwnRenderer:
                self.vmtkRenderer.Deallocate()
                self.OwnRenderer = 0

        else:
            self.Surface = vscop.ComputeVasculatureThickness(
                              self.Surface,
                              self.Centerlines,
                              thickness_field_name=self.ThicknessArrayName,
                              set_uniform_wlr=self.UniformWallToLumenRatio,
                              uniform_wlr_value=self.WallLumenRatio
                           )

        # Get all arrays
        newCellArrays  = [arr for arr in tools.GetCellArrays(self.Surface)
                          if arr not in origCellArrays]

        newPointArrays = [arr for arr in tools.GetPointArrays(self.Surface)
                          if arr not in origPointArrays]

        # Project new arrays to original surface
        for arr in newCellArrays:
            polygonalSurface = tools.ProjectCellArray(
                                   polygonalSurface,
                                   self.Surface,
                                   arr
                               )

        for arr in newPointArrays:
            polygonalSurface = tools.ProjectPointArray(
                                   polygonalSurface,
                                   self.Surface,
                                   arr
                               )

        self.Surface = polygonalSurface

        if self.GenerateWallMesh:
            self.ExtrudeWallMesh()

        if self.OwnRenderer:
            self.vmtkRenderer.Deallocate()

if __name__ == '__main__':
    main = pypes.pypeMain()
    main.Arguments = sys.argv
    main.Execute()
