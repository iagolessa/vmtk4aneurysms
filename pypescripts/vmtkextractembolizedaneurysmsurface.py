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
from numpy import mean, where

from vmtk import pypes
from vmtk import vmtkscripts
from vmtk import vmtkrenderer
from vtk.numpy_interface import dataset_adapter as dsa

from vmtk4aneurysms.lib import polydatatools as tools

# Defining the relation between the class name 'customScrpt'
# and this file name
vmtkextractembolizedaneurysmsurface = 'vmtkExtractEmbolizedAneurysmSurface'

class vmtkExtractEmbolizedAneurysmSurface(pypes.pypeScript):
    def __init__(self):
        pypes.pypeScript.__init__(self)

        self.Image   = None
        self.Surface = None
        self.CoilSurface = None
        self.Inflation = 0.0
        self.Gamma = 1.0
        self.CoilRegionThreshold = 0.25
        self.VascularLevel = 0.0
        self.CoilsLevel = 0.0
        self.ArrayName = "VasculatureWithoutCoilScalars"

        self.vmtkRenderer = None
        self.OwnRenderer = 0
        self.ShowOutput = False

        self.SetScriptName('vmtkextractembolizedaneurysmsurface')

        self.SetScriptDoc(
            """Given an image with a vasculature with an embolized aneurysm,
            extract the 'wet surface' only, i.e. it extracts the portion with
            flow in it by also extracting the coil and assuming that there is
            no flow inside the coiled portion."""
        )

        self.SetInputMembers([
            ['Image','i','vtkImageData',1,'',
                'the input image','vmtkimagereader'],

            ['VascularLevel', 'vascularlevel', 'float', 1, '',
                'graylevels corresponding to the vascular isosurface'],

            ['CoilsLevel', 'coilslevel', 'float', 1, '',
                'graylevels roughly corresponding to the coils isosurface'],

            ['Gamma','gamma','float',1,'',
                'value of new map range for the filtered image'],

            ['CoilRegionThreshold','coilregionthreshold','float',1,'',
                'value (above zero) that delimits an extended region of coils'],

            ['Inflation','inflation','float',1,'',
                'inflation parameters of the Marching Cubes algorithm'],

            ['ShowOutput','showoutput','bool',1,'',
                'see the final surface with original image'],

        ])

        self.SetOutputMembers([
            ['Surface', 'o', 'vtkPolyData', 1, '',
                'the output surface','vmtksurfacewriter'],
        ])

    def ShowInputImage(self, obj):
        # Turn opacity of surface and show image
        self.imageViewer = vmtkscripts.vmtkImageViewer()
        self.imageViewer.vmtkRenderer = self.vmtkRenderer
        self.imageViewer.Image = self.Image
        self.imageViewer.BuildView()

    def ShowCoilSurface(self, obj):

        self.surfaceViewer2 = vmtkscripts.vmtkSurfaceViewer()
        self.surfaceViewer2.vmtkRenderer = self.vmtkRenderer
        self.surfaceViewer2.Surface = self.CoilSurface
        self.surfaceViewer2.Opacity = 0.4
        self.surfaceViewer2.Color = [0.0, 1.0, 0.0]
        self.surfaceViewer2.Display = 1
        self.surfaceViewer2.BuildView()


    def ViewImage(self, image):
        imageViewer = vmtkscripts.vmtkImageViewer()
        imageViewer.Image = image
        imageViewer.Execute()

    def GenerateImageLevelSet(self, image, level):
        # Get level sets image of the vascular and coiled portions
        initializationImage = vmtkscripts.vmtkImageInitialization()
        initializationImage.Image = image
        initializationImage.Method = 'isosurface'
        initializationImage.IsoSurfaceValue = level
        initializationImage.Interactive = 0
        initializationImage.Execute()

        imageLevelSets = vmtkscripts.vmtkLevelSetSegmentation()

        # Input image
        imageLevelSets.Image = initializationImage.Image
        imageLevelSets.InitialLevelSets = initializationImage.InitialLevelSets

        # Default parameters that work well overall
        imageLevelSets.NumberOfIterations  = 300
        imageLevelSets.PropagationScaling  = 0.5
        imageLevelSets.CurvatureScaling    = 0.1
        imageLevelSets.AdvectionScaling    = 1.0
        imageLevelSets.SmoothingIterations = 20
        imageLevelSets.Execute()

        return imageLevelSets.LevelSets

    def MapImageScalarsRange(self, image, new_range):
        """Given a VTK image, map the image scalars values to a
        specified range."""

        normalizeLevelSets = vmtkscripts.vmtkImageShiftScale()
        normalizeLevelSets.Image = image
        normalizeLevelSets.MapRanges = True
        normalizeLevelSets.OutputRange = new_range
        normalizeLevelSets.Execute()

        return normalizeLevelSets.Image

    def ExtractSurfaceMarchingCubes(
            self,
            image,
            level,
            array_name="ImageScalars"
        ):

        marchingCubes = vmtkscripts.vmtkMarchingCubes()
        marchingCubes.Image = image
        marchingCubes.ArrayName = array_name
        marchingCubes.Level = level
        marchingCubes.Execute()

        cleaner = vtk.vtkCleanPolyData()
        cleaner.SetInputData(marchingCubes.Surface)
        cleaner.Update()

        triangleFilter = vtk.vtkTriangleFilter()
        triangleFilter.SetInputConnection(cleaner.GetOutputPort())
        triangleFilter.Update()

        return triangleFilter.GetOutput()

    def Execute(self):
        if not self.Image:
            self.PrintError('Error: No Image.')

        if not self.VascularLevel:
            self.PrintError('Error: No Vascular Level')

        if not self.CoilsLevel:
            self.PrintError('Error: No Coils Level')

        self.OutputText("\nSegmenting coils surface region\n")
        imageLevelSetsCoils = self.GenerateImageLevelSet(
                                  self.Image,
                                  self.CoilsLevel
                              )

        self.OutputText("\nSegmenting vascular surface region\n")
        imageLevelSetsVascular = self.GenerateImageLevelSet(
                                     self.Image,
                                     self.VascularLevel
                                 )

        # Set new map range of ImageScalars
        newMapRange = [-self.Gamma, self.Gamma]

        imageLevelSetsCoils = self.MapImageScalarsRange(
                                  imageLevelSetsCoils,
                                  newMapRange
                              )

        imageLevelSetsVascular = self.MapImageScalarsRange(
                                     imageLevelSetsVascular,
                                     newMapRange
                                 )

        # self.ViewImage(imageLevelSetsCoils)
        # self.ViewImage(imageLevelSetsVascular)

        # Filter the vascular image with the coiled portion
        npLevelSetsCoils    = dsa.WrapDataObject(imageLevelSetsCoils)
        npLevelSetsVascular = dsa.WrapDataObject(imageLevelSetsVascular)

        scalarsCoils       = npLevelSetsCoils.GetPointData().GetArray("ImageScalars")
        scalarsVasculature = npLevelSetsVascular.GetPointData().GetArray("ImageScalars")

        # Filtering: set 0.25 (positive number in the coils region) filter a
        # positive number (different then zero so a larger portion of the coil
        # is removed)
        threshold = mean(newMapRange) + self.CoilRegionThreshold*max(newMapRange)
        coiledPortionSetValue = max(newMapRange)

        vasculatureNoCoilsFields = where(
                                       scalarsCoils > threshold,
                                       scalarsVasculature,
                                       coiledPortionSetValue
                                   )

        npLevelSetsVascular.PointData.append(
            vasculatureNoCoilsFields,
            self.ArrayName
        )

        imageLevelSetsVascular = npLevelSetsVascular.VTKObject

        # Extract surface
        self.Surface = self.ExtractSurfaceMarchingCubes(
                           imageLevelSetsVascular,
                           self.Inflation,
                           array_name=self.ArrayName
                       )

        # Extract largest connected surface
        surfaceConnected = vmtkscripts.vmtkSurfaceConnectivity()
        surfaceConnected.Surface = self.Surface
        surfaceConnected.Execute()

        # Get final surface
        self.Surface = surfaceConnected.Surface

        # Estimate the coil surface used
        self.CoilSurface = self.ExtractSurfaceMarchingCubes(
                               imageLevelSetsCoils,
                               0.0
                           )


        if self.ShowOutput:
            # Extract coil isosurface to show too
            # Initialize renderer = surface + image
            self.vmtkRenderer = vmtkscripts.vmtkRenderer()

            self.vmtkRenderer.AddKeyBinding(
                'space',
                'Show input image',
                self.ShowInputImage
            )

            self.vmtkRenderer.AddKeyBinding(
                'c',
                'Show estimated coil isosurface (green)',
                self.ShowCoilSurface
            )

            self.vmtkRenderer.Initialize()

            self.surfaceViewer = vmtkscripts.vmtkSurfaceViewer()
            self.surfaceViewer.vmtkRenderer = self.vmtkRenderer
            self.surfaceViewer.Surface = self.Surface
            self.surfaceViewer.Opacity = 0.35
            self.surfaceViewer.Color = [1.0, 0.0, 0.0]
            self.surfaceViewer.BuildView()

            self.vmtkRenderer.Deallocate()

if __name__ == '__main__':
    main = pypes.pypeMain()
    main.Arguments = sys.argv
    main.Execute()
