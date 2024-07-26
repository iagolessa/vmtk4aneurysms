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

import os
import sys
import vtk

from vmtk import pypes
from vmtk import vmtkscripts

from vmtk4aneurysms.lib import polydatatools as tools

vmtksurfacevasculatureflowanimation = 'vmtkSurfaceVasculatureFlowAnimation'

class vmtkSurfaceVasculatureFlowAnimation(pypes.pypeScript):

    def __init__(self):
        pypes.pypeScript.__init__(self)

        self.Surface = None
        self.SurfaceExtra = None
        self.Traces = None
        self.StoreVideo = False
        self.FramesDirectory = "/tmp/"
        self.VideoOutputFile = None
        self.ScaleSurface = True

        self.SetScriptName('vmtksurfacevasculatureflowanimation')
        self.SetScriptDoc('render animation of flow inside vessels')

        self.SetInputMembers([
            ['Surface','i', 'vtkPolyData', 1, '',
                'the input surface', 'vmtksurfacereader'],

            ['SurfaceExtra','iextra', 'vtkPolyData', 1, '',
                'extra input surface to appear in the rendering',
                'vmtksurfacereader'],

            ['Traces','itraces','vtkPolyData',1,'','the input traces',
                'vmtksurfacereader'],

            ['StoreVideo','storevideo','bool', 1, '',
             'indicate to store animation as video file'],

            ['FramesDirectory', 'framesdir', 'str', 1, '',
                'where to store the PNG animation files (not required)'],

            ['ScaleSurface','scalesurface','bool', 1, '',
             'to scale surface from meter to millimeter to match traces units',],

            ['VideoOutputFile', 'videofile', 'str', 1, '',
             'file path where to store video (with extension; MP4 or MKV)']
        ])

        self.SetOutputMembers([
        ])

    def Execute(self):
        if not self.Surface:
            self.PrintError('Error: no Surface.')

        # # Filter input surface
        # triangleFilter = vtk.vtkTriangleFilter()
        # triangleFilter.SetInputData(self.Surface)
        # triangleFilter.Update()

        # self.Surface = triangleFilter.GetOutput()

        if self.ScaleSurface:
            self.Surface = tools.ScaleVtkObject(self.Surface, 1e-3)
            self.SurfaceExtra = tools.ScaleVtkObject(self.SurfaceExtra, 1e-3)

        self.vmtkRenderer = vmtkscripts.vmtkRenderer()
        self.vmtkRenderer.Initialize()

        surfaceViewer1 = vmtkscripts.vmtkSurfaceViewer()
        surfaceViewer1.vmtkRenderer = self.vmtkRenderer
        surfaceViewer1.Surface = self.Surface
        surfaceViewer1.Opacity = 0.5
        surfaceViewer1.Color = [1.0, 1.0, 1.0]
        surfaceViewer1.Display = 0 if self.SurfaceExtra else 1
        surfaceViewer1.BuildView()

        if self.SurfaceExtra:
            surfaceViewer2 = vmtkscripts.vmtkSurfaceViewer()
            surfaceViewer2.vmtkRenderer = self.vmtkRenderer
            surfaceViewer2.Surface = self.SurfaceExtra
            surfaceViewer2.Opacity = 1
            surfaceViewer2.Color = [0.0, 1.0, 0.0]
            surfaceViewer2.Display = 1
            surfaceViewer2.BuildView()

        pathlineAnimator = vmtkscripts.vmtkPathLineAnimator()

        pathlineAnimator.InputTraces = self.Traces
        pathlineAnimator.Method = "streaklines"
        pathlineAnimator.StreakLineTimeLength = 0.05
        pathlineAnimator.Legend = 0
        pathlineAnimator.ColorMap = "blackbody"
        pathlineAnimator.MinTime = 0
        pathlineAnimator.MaxTime = 0.3 # 1.0
        pathlineAnimator.TimeStep = 0.005 # ideal 0.001
        pathlineAnimator.ImagesDirectory = self.FramesDirectory
        pathlineAnimator.WithScreenshots = self.StoreVideo
        pathlineAnimator.LineWidth = 8
        pathlineAnimator.ArrayMax = 1.5
        pathlineAnimator.ArrayUnit = "m/s"
        pathlineAnimator.vmtkRenderer = self.vmtkRenderer
        pathlineAnimator.Execute()

        # Store video
        if self.StoreVideo:

            os.system(
                "parallel -I {} convert -resize '20%' {} {} ::: " + self.FramesDirectory + "/frame_0*.png"
            )   

            os.system(
                "parallel -I {} convert {} -fill 'rgb\(24,24,31\)' -opaque 'rgb\(26,26,51\)' -flatten {} ::: " + self.FramesDirectory + "/frames_*.png"
            )

            os.system(
                "ffmpeg -r 30 -stream_loop 2 -i " + self.FramesDirectory + "/frame_%04d.png -vcodec libx264 -pix_fmt yuv420p " + self.VideoOutputFile
            )
            

if __name__ == '__main__':
    main = pypes.pypeMain()
    main.Arguments = sys.argv
    main.Execute()
