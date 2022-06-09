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
            ['Image','i','vtkImageData',1,'',
                'the input image','vmtkimagereader'],

            ['Level','level','float',1,'',
                'graylevels to generate the isosurface at'],

            ['Inflation','inflation','float',1,'',
                'inflation parameters of the Marching Cubes algorithm'],

            ['ShowOutput','showoutput','bool',1,'',
                'whether to see the final surface with image'],

        ])

        self.SetOutputMembers([
            ['Surface','o','vtkPolyData',1,'',
                'the output surface','vmtksurfacewriter']
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
            
            self.vmtkRenderer.Deallocate()


if __name__ == '__main__':
    main = pypes.pypeMain()
    main.Arguments = sys.argv
    main.Execute()
