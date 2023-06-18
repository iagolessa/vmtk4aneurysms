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

vmtkextractrawsurface = 'vmtkExtractRawSurface'

class vmtkExtractRawSurface(pypes.pypeScript):
    def __init__(self):
        pypes.pypeScript.__init__(self)

        self.Image   = None
        self.Surface = None
        self.Inflation = 0.0
        self.Level = 0.0
        self.LevelSetsImage = None

        # Default values
        self.NumberOfIterations  = 300
        self.PropagationScaling  = 0.5
        self.CurvatureScaling    = 0.1
        self.AdvectionScaling    = 1.0
        self.SmoothingIterations = 20

        self.OwnRenderer = 0
        self.ShowOutput = False

        self.SetScriptName('vmtkextractrawsurface')
        self.SetScriptDoc('Automatically extract raw surface from image.')
        self.SetInputMembers([
            ['Image','i','vtkImageData',1,'',
                'the input image','vmtkimagereader'],

            ['Level','level','float',1,'',
                'graylevels to generate the isosurface at'],

            ['Inflation','inflation','float',1,'',
                'inflation parameters of the Marching Cubes algorithm'],

            ['NumberOfIterations','iterations','int',1,'(0,)',
                'number of iterations of level sets algorithm'],

            ['PropagationScaling','propagation','float',1,'(0.0,)',
                'propagation scaling of level sets algorithm'],

            ['CurvatureScaling','curvature','float',1,'(0.0,)',
                'curvature scaling of level sets algorithm'],

            ['AdvectionScaling','advection','float',1,'(0.0,)',
                'advection scaling of level sets algorithm'],

            ['SmoothingIterations','smoothingiterations','int',1,'(0,)',
                'smoothing iterations of level sets algorithm'],

            ['ShowOutput','showoutput','bool',1,'',
                'whether to see the final surface with image'],

        ])

        self.SetOutputMembers([
            ['LevelSetsImage','olevelsets','vtkImageData',1,'',
                'the output levels sets image','vmtkimagewriter'],

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

        self.imageLevelSets = vmtkscripts.vmtkLevelSetSegmentation()

        # Input image
        self.imageLevelSets.Image = self.initializationImage.Image
        self.imageLevelSets.InitialLevelSets = self.initializationImage.InitialLevelSets
        self.imageLevelSets.NumberOfIterations  = self.NumberOfIterations
        self.imageLevelSets.PropagationScaling  = self.PropagationScaling
        self.imageLevelSets.CurvatureScaling    = self.CurvatureScaling
        self.imageLevelSets.AdvectionScaling    = self.AdvectionScaling
        self.imageLevelSets.SmoothingIterations = self.SmoothingIterations
        self.imageLevelSets.Execute()

        self.LevelSetsImage = self.imageLevelSets.LevelSets

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
            self.vmtkRenderer.AddKeyBinding(
                'space',
                'Show input image',
                self.ShowInputImage
            )

            self.vmtkRenderer.Initialize()

            self.surfaceViewer = vmtkscripts.vmtkSurfaceViewer()
            self.surfaceViewer.vmtkRenderer = self.vmtkRenderer
            self.surfaceViewer.Surface = self.Surface
            self.surfaceViewer.Opacity = 0.4
            self.surfaceViewer.BuildView()

            self.vmtkRenderer.Deallocate()


if __name__ == '__main__':
    main = pypes.pypeMain()
    main.Arguments = sys.argv
    main.Execute()
