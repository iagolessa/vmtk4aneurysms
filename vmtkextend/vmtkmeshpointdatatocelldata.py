#!/usr/bin/env python

from __future__ import absolute_import #NEEDS TO STAY AS TOP LEVEL MODULE FOR Py2-3 COMPATIBILITY
import vtk
from vmtk import vtkvmtk
import sys

from vmtk import pypes

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
