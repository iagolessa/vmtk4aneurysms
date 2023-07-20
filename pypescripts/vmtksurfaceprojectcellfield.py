#!/usr/bin/env python

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


from __future__ import absolute_import #NEEDS TO STAY AS TOP LEVEL MODULE FOR Py2-3 COMPATIBILITY

import sys
import vtk

from vmtk import vmtkscripts
from vmtk import vtkvmtk
from vmtk import pypes

vmtksurfaceprojectcellfield = 'vmtkSurfaceProjectCellField'

class vmtkSurfaceProjectCellField(pypes.pypeScript):

    def __init__(self):

        pypes.pypeScript.__init__(self)

        self.Surface = None
        self.ReferenceSurface = None
        self.FieldName = ""

        self.SetScriptName('vmtksurfaceprojectcellfield')
        self.SetScriptDoc(
            """Project a cell field from a reference surface to another."""
        )

        self.SetInputMembers([
            ['Surface',	'i', 'vtkPolyData', 1, '',
                'the input surface', 'vmtksurfacereader'],

            ['FieldName', 'fieldname', 'str', 1, '',
                'the name of the field to project'],

            ['ReferenceSurface', 'r', 'vtkPolyData', 1, '',
                'the reference surface', 'vmtksurfacereader'],
        ])

        self.SetOutputMembers([
            ['Surface', 'o', 'vtkPolyData', 1, '',
                'the output surface', 'vmtksurfacewriter']
        ])


    def Execute(self):
        if self.Surface == None:
            self.PrintError('Error: no Surface.')

        if self.ReferenceSurface == None:
            self.PrintError('Error: no Reference Surface.')

        # Clean before smoothing array
        cleaner = vtk.vtkCleanPolyData()
        cleaner.SetInputData(self.Surface)
        cleaner.Update()

        self.Surface = cleaner.GetOutput()

        self.Surface = tools.ProjectCellArray(
                           self.Surface,
                           self.ReferenceSurface,
                           self.FieldName
                       )


if __name__=='__main__':
    main = pypes.pypeMain()
    main.Arguments = sys.argv
    main.Execute()
