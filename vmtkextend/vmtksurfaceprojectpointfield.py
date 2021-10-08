#!/usr/bin/env python

from __future__ import absolute_import #NEEDS TO STAY AS TOP LEVEL MODULE FOR Py2-3 COMPATIBILITY

import sys
import vtk

from vmtk import vmtkscripts
from vmtk import vtkvmtk
from vmtk import pypes

vmtksurfaceprojectpointfield = 'vmtkSurfaceProjectPointField'

class vmtkSurfaceProjectPointField(pypes.pypeScript):

    def __init__(self):

        pypes.pypeScript.__init__(self)

        self.Surface = None
        self.ReferenceSurface = None
        self.FieldName = ""

        self.SetScriptName('vmtksurfaceprojectpointfield')
        self.SetScriptDoc(
            """Project a Point Field from a reference surface to another."""
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

        # Remove spourious array from final surface
        cellData  = self.ReferenceSurface.GetCellData()
        pointData = self.ReferenceSurface.GetPointData()

        cellArrays  = [cellData.GetArray(id_).GetName()
                       for id_ in range(cellData.GetNumberOfArrays())]

        pointArrays = [pointData.GetArray(id_).GetName()
                       for id_ in range(pointData.GetNumberOfArrays())]

        # Remove the one from the list
        pointArrays.remove(self.FieldName)

        for point_array in pointArrays:
            pointData.RemoveArray(point_array)
            
        for cell_array in cellArrays:
            cellData.RemoveArray(cell_array)

        # Then project the left one to new surface
        surfaceProjection = vtkvmtk.vtkvmtkSurfaceProjection()
        surfaceProjection.SetInputData(self.Surface)
        surfaceProjection.SetReferenceSurface(self.ReferenceSurface)
        surfaceProjection.Update()

        self.Surface = surfaceProjection.GetOutput()

if __name__=='__main__':
    main = pypes.pypeMain()
    main.Arguments = sys.argv
    main.Execute()
