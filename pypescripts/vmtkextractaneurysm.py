#! /usr/bin/env python

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

from vmtk import vtkvmtk
from vmtk import vmtkscripts
from vmtk import pypes

from vmtk4aneurysms.vascular_operations import ExtractAneurysmSacSurface
from vmtk4aneurysms.aneurysms import GenerateOstiumSurface

vmtkextractaneurysm = 'vmtkExtractAneurysm'

class vmtkExtractAneurysm(pypes.pypeScript):

    # Constructor
    def __init__(self):
        pypes.pypeScript.__init__(self)

        self.Surface = None
        self.AneurysmSurface = None
        self.VesselSurface   = None
        self.OstiumSurface   = None
        self.AneurysmType    = None

        self.ComputationMode = "interactive"
        self.ParentVesselSurface = None
        self.ComputeOstium   = False

        self.SetScriptName('vmtkextractaneurysm')
        self.SetScriptDoc('extract aneurysm from vascular surface')

        self.SetInputMembers([
            ['Surface','i', 'vtkPolyData', 1, '',
                'the input surface', 'vmtksurfacereader'],

            ['AneurysmType','type', 'str', 1, '["lateral", "bifurcation"]',
                'aneurysm type'],

            ['ComputationMode','mode', 'str', 1,
                '["interactive", "automatic", "plane"]',
                'mode of neck ostium computation'],

            ['ParentVesselSurface', 'iparentvessel', 'vtkPolyData', 1, '',
                'the parent vessel surface (if not passed, computed externally)',
                'vmtksurfacereader'],

            ['ComputeOstium','computeostium', 'bool', 1,'',
                'do not generate ostium surface']
        ])

        self.SetOutputMembers([
            ['AneurysmSurface','oaneurysm','vtkPolyData',1,'',
             'the aneurysm sac surface', 'vmtksurfacewriter'],

            ['OstiumSurface','oostium','vtkPolyData',1,'',
             'the ostium surface generated from the contour scalar neck',
             'vmtksurfacewriter'],
        ])

    def Execute(self):
        if not self.Surface:
            self.PrintError('Error: no Surface.')

        # Filter input surface
        triangleFilter = vtk.vtkTriangleFilter()
        triangleFilter.SetInputData(self.Surface)
        triangleFilter.Update()

        self.Surface = triangleFilter.GetOutput()

        self.AneurysmSurface = ExtractAneurysmSacSurface(
                                   self.Surface,
                                   mode=self.ComputationMode,
                                   parent_vascular_surface=self.ParentVesselSurface,
                                   aneurysm_type=self.AneurysmType
                               )

        # Generate ostium surface
        if self.ComputeOstium:
            self.OstiumSurface = GenerateOstiumSurface(
                                    self.AneurysmSurface,
                                    compute_normals=True
                                )

if __name__ == '__main__':
    main = pypes.pypeMain()
    main.Arguments = sys.argv
    main.Execute()
