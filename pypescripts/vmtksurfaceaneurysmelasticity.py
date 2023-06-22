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
import math
import numpy as np
import vtk.numpy_interface.dataset_adapter as dsa

from vmtk import vtkvmtk
from vmtk import vmtkscripts
from vmtk import pypes

from vmtk4aneurysms.lib import polydatatools as tools
from vmtk4aneurysms.lib import names

from vmtk4aneurysms import vascular_operations as vscop

vmtksurfaceaneurysmelasticity = 'vmtkSurfaceAneurysmelasticity'

class vmtkSurfaceAneurysmElasticity(pypes.pypeScript):

    def __init__(self):
        pypes.pypeScript.__init__(self)

        self.Surface = None
        self.AneurysmElasticityMode = "uniform"
        self.ElasticityArrayName = names.ElasticityArrayName
        self.ArteriesElasticity = 5e6
        self.AneurysmElasticity = 2e6
        self.SmoothingIterations = 5

        self.NeckComputationMode = "interactive"

        # Required if neck computation mode is 'automatic'
        self.ParentVesselSurface = None
        self.AneurysmType = None # in case only 1 aneurysm
        self.DomePoint = []
        self.DistanceToNeckArrayName = names.DistanceToNeckArrayName

        self.AbnormalHemodynamicsRegions = False
        self.AtheroscleroticFactor = 1.20
        self.RedRegionsFactor = 0.95

        self.SetScriptName('vmtksurfaceaneurysmelasticity')
        self.SetScriptDoc(
            """Adds an array of elasticities on a vascular surface with an
            aneurysm. """
        )

        self.SetInputMembers([
            ['Surface', 'i', 'vtkPolyData', 1, '',
                'the input surface', 'vmtksurfacereader'],

            ['AneurysmElasticityMode', 'aneurysmelasticitymode', 'str', 1,
                '["uniform", "linear"]',
                'indicates uniform aneurysm elasticity'],

            ['ElasticityArrayName', 'elasticityarray', 'str', 1, '',
                'name of the resulting elasticity array'],

            ['ArteriesElasticity', 'arterieselasticity', 'float', 1, '',
                'elasticity of the arteries (and aneurysm neck)'],

            ['AneurysmElasticity', 'aneurysmelasticity', 'float', 1, '',
                'aneurysm elasticity, if uniform, or it is the aneurysm fundus'\
                'elasticity, if the linear varying option is enabled.'],

            ['SmoothingIterations', 'iterations', 'int', 1, '',
                'number of iterations for array smoothing'],

            ['NeckComputationMode','neckcomputationmode', 'str' , 1,
                '["interactive","automatic"]',
                'if the neck array is not in the surface, compute it using '\
                'one of these methods'],

            ['ParentVesselSurface', 'iparentvessel', 'vtkPolyData', 1, '',
                'the parent vessel surface (if not passed, computed externally)',
                'vmtksurfacereader'],

            ['DomePoint','domepoint', 'float', -1, '',
                'coordinates of aneurysm dome point'],

            ['AneurysmType','aneurysmtype', 'str' , 1,
                '["lateral","bifurcation"]',
                'if only one aneurysm, pass also its type'],

            ['AbnormalHemodynamicsRegions', 'abnormalregions', 'bool', 1, '',
                'enable update on elasticity based on WallType array created '\
                'based on hemodynamics variables'],

            ['AtheroscleroticFactor', 'atheroscleroticfactor', 'float', 1, '',
                'scale fator to update elasticity of atherosclerotic regions '\
                'if AbnormalHemodynamicsRegions is true'],

            ['RedRegionsFactor', 'redregionsfactor', 'float', 1, '',
                'scale fator to update elasticity of red regions '            \
                'if AbnormalHemodynamicsRegions is true']
        ])

        self.SetOutputMembers([
            ['Surface', 'o', 'vtkPolyData', 1, '',
                'the input surface with elasticity array', 'vmtksurfacewriter'],
        ])

    def Execute(self):

        if not self.Surface:
            self.PrintError('Error: no Surface.')

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

        self.Surface = vscop.ComputeVasculatureElasticityWithAneurysm(
                           self.Surface,
                           elasticity_field_name=self.ElasticityArrayName,
                           aneurysm_elasticity_mode=self.AneurysmElasticityMode,
                           arteries_elasticity=self.ArteriesElasticity,
                           aneurysm_elasticity=self.AneurysmElasticity,
                           neck_comp_mode=self.NeckComputationMode,
                           gdistance_to_neck_array_name=self.DistanceToNeckArrayName,
                           aneurysm_type=self.AneurysmType,
                           parent_vessel_surface=self.ParentVesselSurface,
                           dome_point=self.DomePoint,
                           abnormal_elasticity=self.AbnormalHemodynamicsRegions,
                           atherosclerotic_factor=self.AtheroscleroticFactor,
                           red_regions_factor=self.RedRegionsFactor,
                           nsmooth_iterations=self.SmoothingIterations
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

if __name__ == '__main__':
    main = pypes.pypeMain()
    main.Arguments = sys.argv
    main.Execute()
