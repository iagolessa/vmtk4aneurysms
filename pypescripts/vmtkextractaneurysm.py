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
from pprint import PrettyPrinter

from vmtk4aneurysms.lib import names
from vmtk4aneurysms.lib import constants as const

from vmtk4aneurysms.lib.polydatatools import RemeshSurface, ClipWithScalar

from vmtk4aneurysms.aneurysms import SaccularAneurysm
from vmtk4aneurysms.neck_extractor import ClipAneurysmSacSurface

vmtkextractaneurysm = 'vmtkExtractAneurysm'

class vmtkExtractAneurysm(pypes.pypeScript):

    # Constructor
    def __init__(self):
        pypes.pypeScript.__init__(self)

        self.Surface = None
        self.AneurysmSurface = None
        self.VesselSurface   = None
        self.OstiumSurface   = None
        self.HullSurface     = None
        self.AneurysmType    = None
        self.AneurysmStatus  = None
        self.DomePoint       = None

        self.ComputationMode = "interactive"
        self.ParentVesselSurface = None

        self.ShowAneurysm = False

        self.SetScriptName('vmtkextractaneurysm')
        self.SetScriptDoc('extract aneurysm from vascular surface')

        self.SetInputMembers([
            ['Surface','i', 'vtkPolyData', 1, '',
             'the input surface', 'vmtksurfacereader'],

            ['AneurysmType','type', 'str', 1, '["lateral", "bifurcation"]',
             'aneurysm type'],

            ['AneurysmStatus','status', 'str', 1, '["ruptured", "unruptured"]',
             'rupture status'],

            ['DomePoint', 'domepoint', 'float', -1, '',
             'coordinates of aneurysm dome point'],

            ['ComputationMode','mode', 'str', 1,
             '["interactive", "automatic", "plane"]',
             'mode of neck ostium computation'],

            ['ParentVesselSurface', 'iparentvessel', 'vtkPolyData', 1, '',
             'the parent vessel surface (if not passed, computed externally)',
             'vmtksurfacereader'],

            ['ShowAneurysm','showaneurysm','bool', 1, '',
             'toggle visualization of the aneurysm, hull, and ostium surfaces']
        ])

        self.SetOutputMembers([
            ['Surface','o','vtkPolyData',1,'',
             'the vascular surface',
             'vmtksurfacewriter'],

            ['AneurysmSurface','oaneurysm','vtkPolyData',1,'',
             'the aneurysm sac surface', 'vmtksurfacewriter'],

            ['OstiumSurface','oostium','vtkPolyData',1,'',
             'the ostium surface generated from the contour scalar neck',
             'vmtksurfacewriter'],

            ['HullSurface','ohull','vtkPolyData',1,'',
             'the ostium surface generated from the contour scalar neck',
             'vmtksurfacewriter'],
        ])

    def Execute(self):

        raise DeprecationWarning(
            self.__class__.__name__ + ' is deprecated and will be deleted '\
            'soon. Use "vmtksurfacevasculatureinfo" instead.'
        )

if __name__ == '__main__':
    main = pypes.pypeMain()
    main.Arguments = sys.argv
    main.Execute()
