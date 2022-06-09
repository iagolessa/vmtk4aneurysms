#!/usr/bin/env python

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
            """Project a Cell Field from a reference surface to another."""
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
        cellArrays  = [cellData.GetArray(id_).GetName()
                       for id_ in range(cellData.GetNumberOfArrays())]

        if self.FieldName not in cellArrays:
            self.PrintError(
                "No field {} on the reference surface.".format(self.FieldName)
            )

        else:
            # Then project the left one to new surface
            projector = vtkvmtk.vtkvmtkSurfaceProjectCellArray()
            projector.SetInputData(self.Surface)
            projector.SetReferenceSurface(self.ReferenceSurface)
            projector.SetProjectedArrayName(self.FieldName)
            projector.SetDefaultValue(0.0)
            projector.Update()

            self.Surface = projector.GetOutput()


if __name__=='__main__':
    main = pypes.pypeMain()
    main.Arguments = sys.argv
    main.Execute()
