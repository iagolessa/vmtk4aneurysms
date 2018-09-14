#! /usr/bin/env python

import sys
import vtk

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
        
        self.initializationImage = vmtkscripts.vmtkImageInitialization()
        self.initializationImage.Image = self.Image
        self.initializationImage.Method = 'isosurface'
        self.initializationImage.IsoSurfaceValue = self.Levels[0]
        self.initializationImage.Interactive = 0
        self.initializationImage.Execute()

        # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~ #
	# If image was generated from dicom above
        self.imageLevelSets = vmtkscripts.vmtkLevelSetSegmentation()
        
        # Input image
        self.imageLevelSets.Image = self.initializationImage.Image
        self.imageLevelSets.InitialLevelSets = self.initializationImage.InitialLevelSets
        self.imageLevelSets.NumberOfIterations  = 300
        self.imageLevelSets.PropagationScaling  = 0.5
        self.imageLevelSets.CurvatureScaling    = 0.1
        self.imageLevelSets.AdvectionScaling    = 1.0
        self.imageLevelSets.SmoothingIterations = 20
        self.imageLevelSets.Execute()
        
        self.marchingCubes = vmtkscripts.vmtkMarchingCubes()
        self.marchingCubes.Image = self.imageLevelSets.LevelSets
#         self.marchingCubes.Level = 0.1
        self.marchingCubes.Connectivity = 1
        self.marchingCubes.Execute()
       
        cleaner = vtk.vtkCleanPolyData()
        cleaner.SetInputData(self.marchingCubes.Surface)
        cleaner.Update()

        triangleFilter = vtk.vtkTriangleFilter()
        triangleFilter.SetInputConnection(cleaner.GetOutputPort())
        triangleFilter.Update()

        # Extract largest connected surface
#         self.surfaceConnected = vmtkscripts.vmtkSurfaceConnectivity()
#         self.surfaceConnected.Surface = surfaceTriangle.Surface
#         self.surfaceConnected.Execute()

        self.vmtkRenderer = vmtkscripts.vmtkRenderer()
        self.vmtkRenderer.Initialize()
        self.surfaceViewer = vmtkscripts.vmtkSurfaceViewer()
        self.surfaceViewer.vmtkRenderer = self.vmtkRenderer
        
        self.Surface = triangleFilter.GetOutput()
        self.surfaceViewer.Surface = self.Surface
        self.surfaceViewer.BuildView()
        
#         for level in self.Levels:
#             self.marchingCubes.Level = level
#             self.marchingCubes.Execute()
#             self.Surface = self.marchingCubes.Surface
#             self.SurfaceViewer.Surface = self.Surface
#             self.SurfaceViewer.BuildView()


if __name__ == '__main__':
    main = pypes.pypeMain()
    main.Arguments = sys.argv
    main.Execute()
