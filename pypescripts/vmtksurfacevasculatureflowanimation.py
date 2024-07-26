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
        self.Pattern = "frame_%05d.png"

        self.StreakLineTimeLength = 0.05
        self.Legend = 0
        self.ColorMap = 'blackbody'
        self.MinTime = 0.0
        self.MaxTime = 1.0
        self.TimeStep = 0.005
        self.ArrayMax = 1.5
        self.LineWidth = 8 # better for recording

        self.ImageResize = 60
        self.BackgroundColor = [24, 24, 31]
        self.VideoLoopCount = 3
        self.VideoFramesPerSec = 30

        self.SetScriptName('vmtksurfacevasculatureflowanimation')
        self.SetScriptDoc(
            """ render or store animation of vasculature flow, through
            streaklines, using traces previously computed and the vascular
            surface. Warning: if you choose to store a video, bear in mind that
            this screpts uses the 'convert' bin of ImageMagick and 'ffmpeg'
            commands as also "parallel" for parallel processing the batch of
            images generated. These are not part of a typical VMTK
            installation. You must have them installed in your system. Also,
            ffmpeg may used a lot of RAM memory, so try to resize the image by
            using the argument ImageResize.
            """
        )

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

            # Entries derived directly from vmtkpathlineanimator
            ['StreakLineTimeLength','streaklinetimelength','float',1,'(0.0,)'],
            ['Legend', 'legend', 'bool', 1, '', 'toggle scalar bar'],
            ['ColorMap', 'colormap', 'str', 1,
                '["rainbow","blackbody","cooltowarm","grayscale"]',
                'change the color map'],

            ['MinTime', 'mintime', 'float', 1, '(0.0,)'],
            ['MaxTime', 'maxtime', 'float', 1, '(0.0,)'],
            ['TimeStep', 'timestep', 'float', 1, '(0.0,)'],
            ['ArrayMax', 'arraymax', 'float', 1, '(0.0,)'],
            ['LineWidth', 'linewidth', 'int', 1, '(1,)'],

            ['ImageResize', 'imageresize', 'int', 1, '(100,0)',
                "resize image by this percentage"],

            ['BackgroundColor', 'backgroundcolor', 'int', -1, '',
                'color of the background video, in RGB format'],

            ['VideoLoopCount', 'videoloopcount', 'int', 1, '(,0)',
                "number of count of video loop"],

            ['VideoFramesPerSec', 'fps', 'int', 1, '(,0)',
                "video frames rate"],

            ['VideoOutputFile', 'videofile', 'str', 1, '',
             'file path where to store video (with extension; MP4 or MKV)']
        ])

        self.SetOutputMembers([
        ])

    def Execute(self):
        if not self.Surface:
            self.PrintError('Error: no Surface.')

        if self.ScaleSurface:
            self.Surface = tools.ScaleVtkObject(
                                self.Surface,
                                1.0e-3
                            )

            if self.SurfaceExtra:
                self.SurfaceExtra = tools.ScaleVtkObject(
                                        self.SurfaceExtra,
                                        1.0e-3
                                    )

        # Construct view
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

        # Animate
        pathlineAnimator = vmtkscripts.vmtkPathLineAnimator()

        pathlineAnimator.InputTraces = self.Traces
        pathlineAnimator.Method = "streaklines"
        pathlineAnimator.StreakLineTimeLength = self.StreakLineTimeLength
        pathlineAnimator.Legend = self.Legend
        pathlineAnimator.ColorMap = self.ColorMap
        pathlineAnimator.MinTime = self.MinTime
        pathlineAnimator.MaxTime = self.MaxTime
        pathlineAnimator.TimeStep = self.TimeStep
        pathlineAnimator.ImagesDirectory = self.FramesDirectory
        pathlineAnimator.WithScreenshots = self.StoreVideo
        pathlineAnimator.Pattern = self.Pattern
        pathlineAnimator.LineWidth = self.LineWidth
        pathlineAnimator.ArrayMax = self.ArrayMax
        pathlineAnimator.ArrayUnit = "m/s"
        pathlineAnimator.vmtkRenderer = self.vmtkRenderer
        pathlineAnimator.Execute()

        # Store video
        if self.StoreVideo:
            # Not the cleanest way, but it was quick and effective
            # TODO(?): turn this to pure python? It would uintroduce new
            # dependecies any way
            parallelCommandSuffix = "parallel -I {}"
            filesBashPattern = self.FramesDirectory + "/frame_0*.png"
            filesPattern = self.FramesDirectory + self.Pattern

            convertResizeOptions = "-resize '" + str(self.ImageResize) + "%'"

            convertFillOption = "-fill 'rgb\(" + \
                                ",".join(str(c) for c in self.BackgroundColor) + \
                                "\)'"

            ffmpegOtherOptions = "-vcodec libx264 -pix_fmt yuv420p"


            os.system(
                " ".join([
                    parallelCommandSuffix,
                    "convert",
                    convertResizeOptions,
                    "{} {} :::",
                    filesBashPattern
                ])
            )

            os.system(
                " ".join([
                    parallelCommandSuffix,
                    "convert {}",
                    convertFillOption,
                    "-opaque 'rgb\(26,26,51\)' -flatten",
                    "{} :::",
                    filesBashPattern
                ])
            )

            os.system(
                " ".join([
                    "ffmpeg",
                    "-r " + str(self.VideoFramesPerSec),
                    "-stream_loop " + str(self.VideoLoopCount - 1),
                    "-i " + filesPattern,
                    ffmpegOtherOptions,
                    self.VideoOutputFile
                ])
            )


if __name__ == '__main__':
    main = pypes.pypeMain()
    main.Arguments = sys.argv
    main.Execute()
