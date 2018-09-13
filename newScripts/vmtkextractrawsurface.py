#! /usr/bin/env python

import sys

from vmtk import pypes
from vmtk import vmtkscripts

# Defining the relation between the class name 'customScrpt'
# and this file name
vmtkextractrawsurface = 'vmtkExtractRawSurface'


class vmtkExtractRawSurface(pypes.pypeScript):
    def __init__(self):
        pypes.pypeScript.__init__(self)

        self.Image   = None
        self.Surface = None
        self.Levels  = []
        
        self.SetScriptName('vmtkextractrawsurface')
        self.SetScriptDoc('Extract raw surface from image.')
        self.SetInputMembers([
            ['Image','i','vtkImageData',1,'','the input image','vmtkimagereader'],
            ['Levels','levels','float',-1,'','graylevels to generate the isosurface at']])

        self.SetOutputMembers([['Surface','o','vtkPolyData',1,'','the output surface','vmtksurfacewriter']])

    def Execute(self):
        if self.Image == None:
            self.PrintError('Error: No Image.')
        if self.Levels == []:
            self.PrintError('Error: No Levels')

        self.marchingCubes = vmtkscripts.vmtkMarchingCubes()
        self.marchingCubes.Image = self.Image
        self.marchingCubes.Connectivity = 1
        self.vmtkRenderer = vmtkscripts.vmtkRenderer()
        self.vmtkRenderer.Initialize()
        self.SurfaceViewer = vmtkscripts.vmtkSurfaceViewer()
        self.SurfaceViewer.vmtkRenderer = self.vmtkRenderer

        for level in self.Levels:
            self.marchingCubes.Level = level
            self.marchingCubes.Execute()
            self.Surface = self.marchingCubes.Surface
            self.SurfaceViewer.Surface = self.Surface
            self.SurfaceViewer.BuildView()


if __name__ == '__main__':
    main = pypes.pypeMain()
    main.Arguments = sys.argv
    main.Execute()
