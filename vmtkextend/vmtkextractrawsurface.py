#! /usr/bin/env python

import sys
import vtk

from vmtk import pypes
from vmtk import vmtkscripts
from vmtk import vmtkrenderer

# Defining the relation between the class name 'customScrpt'
# and this file name
vmtkextractrawsurface = 'vmtkExtractRawSurface'


class vmtkExtractRawSurface(pypes.pypeScript):
    def __init__(self):     
        pypes.pypeScript.__init__(self)

        self.Image   = None
        self.Surface = None
        self.Inflation = 0.0
        self.Level = 0.0
#         self.vmtkRenderer = None
        self.OwnRenderer = 0
        self.ShowOutput = False
        
        self.SetScriptName('vmtkextractrawsurface')
        self.SetScriptDoc('Extract raw surface from image.')
        self.SetInputMembers([
            ['Image','i','vtkImageData',1,'','the input image','vmtkimagereader'],
            ['Level','level','float',1,'','graylevels to generate the isosurface at'],
            ['Inflation','inflation','float',1,'','inflation parameters of the Marching Cubes algorithm'],
            ['ShowOutput','showoutput','bool',1,'','whether to see the final surface with image'],
#             ['vmtkRenderer','renderer','vmtkRenderer',1,'','external renderer']
        ])

        self.SetOutputMembers([
            ['Surface','o','vtkPolyData',1,'','the output surface','vmtksurfacewriter']
        ])
    

    def ShowInputImage(self,obj):

        # Turn opacity of surface and show image
        self.imageViewer = vmtkscripts.vmtkImageViewer()
        self.imageViewer.vmtkRenderer = self.vmtkRenderer
        self.imageViewer.Image = self.Image
        self.imageViewer.BuildView()


    def Execute(self):
        if self.Image == None:
            self.PrintError('Error: No Image.')
                
        if not self.Level:
            self.PrintError('Error: No Level')
        
#         if not self.vmtkRenderer:
#             self.vmtkRenderer = vmtkrenderer.vmtkRenderer()
#             self.vmtkRenderer.Initialize()
#             self.OwnRenderer = 1
                
        self.initializationImage = vmtkscripts.vmtkImageInitialization()
        self.initializationImage.Image = self.Image
        self.initializationImage.Method = 'isosurface'
        self.initializationImage.IsoSurfaceValue = self.Level
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
        self.marchingCubes.Level = self.Inflation
#         self.marchingCubes.Connectivity = 1
        self.marchingCubes.Execute()
       
        cleaner = vtk.vtkCleanPolyData()
        cleaner.SetInputData(self.marchingCubes.Surface)
        cleaner.Update()

        triangleFilter = vtk.vtkTriangleFilter()
        triangleFilter.SetInputConnection(cleaner.GetOutputPort())
        triangleFilter.Update()
        
        self.Surface = triangleFilter.GetOutput()

        # Extract largest connected surface
        surfaceConnected = vmtkscripts.vmtkSurfaceConnectivity()
        surfaceConnected.Surface = self.Surface
        surfaceConnected.Execute()

        # Get final surface
        self.Surface = surfaceConnected.Surface

        if self.ShowOutput:
            # Initialize renderer = surface + image
            self.vmtkRenderer = vmtkscripts.vmtkRenderer()
            self.vmtkRenderer.AddKeyBinding('space', 'Show input image', self.ShowInputImage)
            self.vmtkRenderer.Initialize()
            
            self.surfaceViewer = vmtkscripts.vmtkSurfaceViewer()
            self.surfaceViewer.vmtkRenderer = self.vmtkRenderer
            self.surfaceViewer.Surface = self.Surface
            self.surfaceViewer.Opacity = 0.5
            self.surfaceViewer.BuildView()
            

#             for level in self.Levels:
#                 self.marchingCubes.Level = level
#                 self.marchingCubes.Execute()
#                 self.Surface = self.marchingCubes.Surface
#                 self.SurfaceViewer.Surface = self.Surface
#                 self.SurfaceViewer.BuildView()
#             if self.OwnRenderer:
            self.vmtkRenderer.Deallocate()


if __name__ == '__main__':
    main = pypes.pypeMain()
    main.Arguments = sys.argv
    main.Execute()