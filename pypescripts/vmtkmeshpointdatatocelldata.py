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
import vtk
from vmtk import vtkvmtk
import sys

from vmtk import pypes

vmtkmeshpointdatatocelldata = 'vmtkMeshPointDataToCellData'

class vmtkMeshPointDataToCellData(pypes.pypeScript):

    def __init__(self):

        pypes.pypeScript.__init__(self)
        
        self.Mesh = None

        self.SetScriptName('vmtkmeshpointdatatocelldata')
        self.SetScriptDoc('convert point data arrays to cell data surface arrays')
        self.SetInputMembers([
            ['Mesh','i','vtkUnstructuredGrid',1,'','the input mesh','vmtkmeshreader']
            ])
        self.SetOutputMembers([
            ['Mesh','o','vtkUnstructuredGrid',1,'','the output mesh','vmtkmeshwriter']
            ])

    def Execute(self):

        if self.Mesh == None:
            self.PrintError('Error: No Mesh.')

        pointDataToCellDataFilter = vtk.vtkPointDataToCellData()
        pointDataToCellDataFilter.SetInputData(self.Mesh)
        pointDataToCellDataFilter.PassPointDataOn()
        pointDataToCellDataFilter.Update()

        self.Mesh = pointDataToCellDataFilter.GetUnstructuredGridOutput()


if __name__=='__main__':
    main = pypes.pypeMain()
    main.Arguments = sys.argv
    main.Execute()
