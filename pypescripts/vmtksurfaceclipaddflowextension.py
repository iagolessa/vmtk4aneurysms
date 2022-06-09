#!/usr/bin/env python

from __future__ import absolute_import #NEEDS TO STAY AS TOP LEVEL MODULE FOR Py2-3 COMPATIBILITY

import sys
from vmtk import vmtkscripts
from vmtk import pypes

vmtksurfaceclipaddflowextension = 'vmtkSurfaceClipAddFlowExtension'

class vmtkSurfaceClipAddFlowExtension(pypes.pypeScript):

    def __init__(self):

        pypes.pypeScript.__init__(self)

        self.Surface = None
        self.ClipMode = 'interactive'
        self.Remesh = False
        self.EdgeLength = 0.1
        self.Cap = False
        self.Interactive = False

        self.SetScriptName('vmtksurfaceclipaddflowextension')
        self.SetScriptDoc('Interactively clip a surface and add small flow '
                          'extension.')

        self.SetInputMembers([
            ['Surface', 'i', 'vtkPolyData', 1, '',
                 'the input surface', 'vmtksurfacereader'],

            ['Remesh' , 'remesh', 'bool', 1, '',
                'to apply remeshing procedure after fixing it'],

            ['EdgeLength' , 'edgelength', 'float', 1, '',
                'to edgelength for the remesh procedure'],

            ['Cap'   , 'cap'  ,'bool',1,'',
                'to cap surface after clipping'],

            ['ClipMode','clipmode', 'str' , 1, '["interactive","centerlinebased"]',
                 'clip mode'],

            ['Interactive' , 'interactive', 'bool', 1, '',
                'interactively choose the boundaries to add extension'],

        ])

        self.SetOutputMembers([
            ['Surface', 'o', 'vtkPolyData', 1, '',
                'the output surface', 'vmtksurfacewriter']
        ])


    def interactiveClip(self):
        surfaceClipper = vmtkscripts.vmtkSurfaceClipper()
        surfaceClipper.Surface = self.Surface
        surfaceClipper.InsideOut = 0
        surfaceClipper.WidgetType = 'box'
        surfaceClipper.Execute()

        self.Surface = surfaceClipper.Surface

    def centerlineClip(self):
        pass

    def Execute(self):
        if self.Surface == None:
            self.PrintError('Error: no Surface.')


        if self.ClipMode == 'interactive':
            self.interactiveClip()
        elif self.ClipMode == 'centerlinebased':
            self.centerlineClip()
        else:
            self.PrintError('Error: clip mode not recognized.')


        # Adding flow extensions
        surfaceFlowExtensions = vmtkscripts.vmtkFlowExtensions()
        surfaceFlowExtensions.Surface = self.Surface

        # Setup
        surfaceFlowExtensions.InterpolationMode = 'thinplatespline' # or linear
        surfaceFlowExtensions.ExtensionMode = 'boundarynormal'      # or centerlinedirection
        # boolean flag which enables computing the length of each
        # flowextension proportional to the mean profile radius
        surfaceFlowExtensions.AdaptiveExtensionLength = 1 # (bool)

        # The proportionality factor is set through 'extensionratio'
        surfaceFlowExtensions.ExtensionRatio = 1

        surfaceFlowExtensions.Interactive = self.Interactive
        surfaceFlowExtensions.TransitionRatio = 0.5
        surfaceFlowExtensions.AdaptiveExtensionRadius = 1
        surfaceFlowExtensions.AdaptiveNumberOfBoundaryPoints = 1
        surfaceFlowExtensions.TargetNumberOfBoundaryPoints = 50
        surfaceFlowExtensions.Sigma = 1.0
        surfaceFlowExtensions.Execute()

        self.Surface = surfaceFlowExtensions.Surface

        if self.Remesh:
            remesher = vmtkscripts.vmtkSurfaceRemeshing()
            remesher.Surface = self.Surface
            remesher.ElementSizeMode = "edgelength"
            remesher.TargetEdgeLength = self.EdgeLength
            remesher.OutputText("Remeshing procedure ...")
            remesher.Execute()

            self.Surface = remesher.Surface

        # Capping surface
        if self.Cap:
            capper = vmtkscripts.vmtkSurfaceCapper()
            capper.Surface = self.Surface
            capper.Method = 'centerpoint'
            capper.Interactive = 0
            capper.Execute()

            self.Surface = capper.Surface


if __name__=='__main__':
    main = pypes.pypeMain()
    main.Arguments = sys.argv
    main.Execute()
