#!/usr/bin/env python

from __future__ import absolute_import #NEEDS TO STAY AS TOP LEVEL MODULE FOR Py2-3 COMPATIBILITY

import sys
import vtk
from vmtk import vmtkscripts
from vmtk import pypes

vmtksurfaceremeshwithresolution = 'vmtkSurfaceRemeshWithResolution'

class vmtkSurfaceRemeshWithResolution(pypes.pypeScript):

    def __init__(self):

        pypes.pypeScript.__init__(self)

        self.Surface = None

        self.InsideValue  = 0.1
        self.OutsideValue = 0.2

        self.SetScriptName('vmtksurfaceremeshwithresolution')
        self.SetScriptDoc("Script to remesh a surface based on a" 
                           "resolution array defined on it created"     
                           "by the user: the user must draw the array"  
                           "on the surface with a drawing utility."
                           "The array scalar are the edge length factor"
                           "for the remeshing. The array is smoothed" 
                           "before the remeshing procedure.")

        self.SetInputMembers([
            ['Surface',	'i', 'vtkPolyData', 1, '', 'the input surface', 'vmtksurfacereader'],
            ['InsideValue', 'inside', 'float', 1, '(0.0,)', 'the inside value'],
            ['OutsideValue', 'outside', 'float', 1, '(0.0,)', 'the outside value'],
        ])

        self.SetOutputMembers([
            ['Surface', 'o', 'vtkPolyData', 1, '', 'the output surface', 'vmtksurfacewriter']
        ])


        
    def Execute(self):
        if self.Surface == None:
            self.PrintError('Error: no Surface.')

        # Creating resolution array 
        resolutionArrayCreator = vmtkscripts.vmtkSurfaceRegionDrawing()
        resolutionArrayCreator.Surface = self.Surface
        resolutionArrayCreator.Binary = 1
        resolutionArrayCreator.InsideValue = self.InsideValue
        resolutionArrayCreator.OutsideValue = self.OutsideValue
        resolutionArrayCreator.ContourScalarsArrayName = 'ResolutionArray'
        resolutionArrayCreator.Execute()

        # Smooth the resolution array 
        resolutionArraySmoothing = vmtkscripts.vmtkSurfaceArraySmoothing()
        resolutionArraySmoothing.Surface = resolutionArrayCreator.Surface
        resolutionArraySmoothing.SurfaceArrayName = resolutionArrayCreator.ContourScalarsArrayName
        resolutionArraySmoothing.Connexity = 1
        resolutionArraySmoothing.Relaxation = 1.0
        resolutionArraySmoothing.Iterations = 15
        resolutionArraySmoothing.OutputText("Smoothing resolution array...\n")
        resolutionArraySmoothing.Execute()

        # Remesh procedure
        surfaceRemesh = vmtkscripts.vmtkSurfaceRemeshing()
        surfaceRemesh.Surface = resolutionArraySmoothing.Surface 
        surfaceRemesh.ElementSizeMode = 'edgelengtharray'
        surfaceRemesh.TargetEdgeLengthArrayName = resolutionArraySmoothing.SurfaceArrayName
        surfaceRemesh.TargetEdgeLengthFactor = 1
        surfaceRemesh.OutputText("Remeshing... \n")
        surfaceRemesh.Execute()

        self.Surface = surfaceRemesh.Surface


if __name__=='__main__':
    main = pypes.pypeMain()
    main.Arguments = sys.argv
    main.Execute()