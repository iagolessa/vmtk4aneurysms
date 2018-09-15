#! /usr/bin/env python

import sys
import vtk

from vmtk import pypes
from vmtk import vmtkscripts

# Defining the relation between the class name 'customScrpt'
# and this file name
vmtksurfacefixer = 'vmtkSurfaceFixer'


class vmtkSurfaceFixer(pypes.pypeScript):
    def __init__(self):
        pypes.pypeScript.__init__(self)

        self.Surface = None
        self.Smooth  = False
        self.Method  = 'taubin'
        self.Subdivide = True
        
        self.SetScriptName('vmtksurfacefixer')
        self.SetScriptDoc("Function to interactively fix a surface of a"
                          "vessel segment with holes and other weird"
                          "artifacts. The function works internally with" 
                          "the region drawing script and clipper with"
                          "array")
        
        self.SetInputMembers([
            ['Surface', 'i','vtkPolyData',1,'','the input surface','vmtksurfacereader'],
            ['Smooth' , 'smooth','bool',1,'','if surface must be smoothed before fixing'],
            ['Method' , 'method','str',1,'["taubin","laplace"]','smoothing method'],
            ['Subdivide','subdivide','bool',1,'','if surface must be subdivided before fixing'],
        ])
        
        self.SetOutputMembers([
            ['Surface','o','vtkPolyData',1,'','the output surface','vmtksurfacewriter']
        ])
    

    def Execute(self):
        if self.Surface == None:
            self.PrintError('Error: No input surface.')
                
        # Clean input surface data
#         cleaner = vtk.vtkCleanPolyData()
#         cleaner.SetInputData(self.Surface)
#         cleaner.Update()

        if self.Smooth:
            self.surfaceSmoother = vmtkscripts.vmtkSurfaceSmoothing()
            self.surfaceSmoother.Surface = self.Surface #cleaner.GetOutputPort()
            self.surfaceSmoother.Method = self.Method
            
            if self.Method == 'taubin':
                self.surfaceSmoother.NumberOfIterations = 30
                self.surfaceSmoother.PassBand = 0.1
                
            self.surfaceSmoother.Execute()
                
        self.cutRegionsMarker = vmtkscripts.vmtkSurfaceRegionDrawing()
        self.cutRegionsMarker.Surface = self.surfaceSmoother.Surface
        self.cutRegionsMarker.Execute()

        # Initial Clipper to revome excess of surface
        self.arrayClipper = vmtkscripts.vmtkSurfaceClipper()
        self.arrayClipper.Surface = self.cutRegionsMarker.Surface
        self.arrayClipper.Interactive = 0
        self.arrayClipper.ClipArrayName = self.cutRegionsMarker.ContourScalarsArrayName
        self.arrayClipper.ClipValue = 0.5*(self.cutRegionsMarker.InsideValue + \
                                           self.cutRegionsMarker.OutsideValue)
        self.arrayClipper.Execute()

        # Extract largest connected surface
        self.surfaceConnected = vmtkscripts.vmtkSurfaceConnectivity()
        self.surfaceConnected.Surface = self.arrayClipper.Surface
        self.surfaceConnected.Execute()

        # Simple remesh procedure to increase quality at cut lines
        self.remesher = vmtkscripts.vmtkSurfaceRemeshing()
        self.remesher.Surface = self.surfaceConnected.Surface
        self.remesher.ElementSizeMode = "edgelength"
        self.remesher.TargetEdgeLength = 0.2
        self.remesher.Execute()

        self.surfaceFixer = vmtkscripts.vmtkSurfaceCapper()
        self.surfaceFixer.Surface = self.remesher.Surface
        self.surfaceFixer.Method = 'smooth'
        self.surfaceFixer.ConstraintFactor = 0.2
        self.surfaceFixer.NumberOfRings = 6
        self.surfaceFixer.Interactive = 0
        self.surfaceFixer.Execute()

        self.Surface = self.surfaceFixer.Surface

if __name__ == '__main__':
    main = pypes.pypeMain()
    main.Arguments = sys.argv
    main.Execute()
