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
from vmtk import vmtkrenderer
from vmtk import pypes

from vmtk4aneurysms import vascular_operations as vscop
from vmtk4aneurysms.lib import polydatatools as tools

vmtksurfacehealthyvasculature = 'vmtkSurfaceHealthyVasculature'

class vmtkSurfaceHealthyVasculature(pypes.pypeScript):

    # Constructor
    def __init__(self):
        pypes.pypeScript.__init__(self)

        self.Surface = None
        self.AneurysmType = None # in case only 1 aneurysm
        self.DomePoint = None

        self.SetScriptName('vmtksurfacehealthyvasculature')
        self.SetScriptDoc(
            """compute the healthy version of the parent vessel of a
            vasculature with a saccular aneurysm."""
        )

        self.SetInputMembers([
            ['Surface','i', 'vtkPolyData', 1, '',
                'the input surface', 'vmtksurfacereader'],

            ['AneurysmType', 'aneurysmtype', 'str' , 1,
                '["lateral","bifurcation"]',
                'if only one aneurysm, pass also its type'],

            ['DomePoint', 'domepoint', 'float', -1, '',
                'coordinates of aneurysm dome point']
        ])

        self.SetOutputMembers([
            ['Surface','o','vtkPolyData',1,'',
             'the healthy vessel surface', 'vmtksurfacewriter'],
        ])


    def Execute(self):
        if not self.Surface:
            self.PrintError("Error: no Surface.")

        if not self.AneurysmType:
            self.PrintError("Error: provide aneurysm type.")

        # Operate on a triangulated surface and map final result to orignal
        # surface
        self.Surface = tools.Cleaner(self.Surface)

        # Will operate on the triangulated one
        triangulate = vtk.vtkTriangleFilter()
        triangulate.SetInputData(self.Surface)
        triangulate.Update()

        self.Surface = triangulate.GetOutput()

        self.Surface = vscop.HealthyVesselReconstruction(
                            self.Surface,
                            self.AneurysmType,
                            self.DomePoint
                        )

if __name__ == '__main__':
    main = pypes.pypeMain()
    main.Arguments = sys.argv
    main.Execute()
