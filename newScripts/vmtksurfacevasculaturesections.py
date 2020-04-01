#! /usr/bin/env python

import os
import sys

from vmtk import pypes
from vmtk import vmtkscripts

import vmtkfunctions as vf
vmtksurfacevasculaturesections = 'vmtkSurfaceVasculatureSections'

class vmtkSurfaceVasculatureSections(pypes.pypeScript):

    def __init__(self):

        pypes.pypeScript.__init__(self)

        self.Surface = None
        self.Remesh = False
        self.Clip = True
        self.SpheresDistance = 1
        self.ElementAreaSize = 0.001

        self.Centerlines = None
        self.RadiusArrayName = "MaximumIncribedSpheresRadiusArray"

        self.SetScriptName('vmtksurfacevasculaturesections')
        self.SetScriptDoc("Build vasculature sections separated by a given "
                          "number of incribed spheres.")

        self.SetInputMembers([
            ['Surface',	'i', 'vtkPolyData', 1, '', 
                'the input surface', 'vmtksurfacereader'],

            ['Remesh' , 'remesh', 'bool', 1, '', 
                'to apply remeshing procedure after fixing it'],

            ['Clip' , 'clip' ,'bool', 1,'',
                'to clip surface with a box before fixing it'],

            ['SpheresDistance',	'spheresdistance', 'int', 1, '', 
                'the number of spheres to be accouted as '
                'distance between the sections'],

            ['ElementAreaSize',	'elementareasize', 'float', 1, '', 
                'the size of area element if remeshing sections'],
        ])

        self.SetOutputMembers([
            ['Surface', 'o', 'vtkPolyData', 1, '', 
                'the output surface with the sections', 'vmtksurfacewriter']
        ])


    def ClipModel(self):
        """Clip surface on inlets and outlets."""

        clipper = vmtkscripts.vmtkSurfaceClipper()
        clipper.Surface = self.Surface
        clipper.Execute()

        self.Surface = clipper.Surface


    def AutomaticClipping(self):
        """Automatically clip models based on centerlines."""
        pass
        # # Procedure to automatically clip model
        # centerlines = vmtkscripts.vmtkCenterlines()
        # centerlines.SurfaceInputFileName = surfacesDir+'surfaceT1.vtp'
        # centerlines.IORead()
        # centerlines.Execute()

        # endPointExtractor = vmtkscripts.vmtkEndpointExtractor()
        # endPointExtractor.Centerlines = centerlines.Centerlines
        # endPointExtractor.RaiusArrayName = centerlines.RadiusArrayName
        # endPointExtractor.Execute()
        # endPointExtractor.PrintOutputMembers()

    def ComputeCenterlines(self):
        """Compute comdel centerlines."""

        centerlines = vmtkscripts.vmtkCenterlines()
        centerlines.Surface = self.Surface
        centerlines.CheckNonManifold = True
        centerlines.AppendEndPoints = False
#         centerlines.SeedSelector = 'openprofiles'
        centerlines.Resampling = True
        centerlines.ResamplingStepLength = 0.10
        centerlines.Execute()

        self.Centerlines = centerlines.Centerlines
        self.RadiusArrayName = centerlines.RadiusArrayName


    def Execute(self):

        if self.Clip:
            self.ClipModel()

        # Compute centerlines of model
        self.ComputeCenterlines()

        # Computing centerlines Frenet system
        cntGeometry = vmtkscripts.vmtkCenterlineGeometry()
        cntGeometry.Centerlines = self.Centerlines
        cntGeometry.LineSmoothing = 0
        cntGeometry.Execute()

        # Computation of centerlines attributes (parallel theory)
        cntAttributes = vmtkscripts.vmtkCenterlineAttributes()
        cntAttributes.Centerlines = cntGeometry.Centerlines
        cntAttributes.Execute()

        self.Centerlines = cntAttributes.Centerlines

        # Split surface into branches
        branchExtractor = vmtkscripts.vmtkBranchExtractor()
        branchExtractor.Centerlines = self.Centerlines
        branchExtractor.RadiusArrayName = self.RadiusArrayName
        branchExtractor.Execute()

        # Branch Clipper
        branchClipper = vmtkscripts.vmtkBranchClipper()
        branchClipper.Surface = self.Surface
        branchClipper.Centerlines = branchExtractor.Centerlines
        branchClipper.RadiusArrayName   = branchExtractor.RadiusArrayName
        branchClipper.GroupIdsArrayName = branchExtractor.GroupIdsArrayName
        branchClipper.BlankingArrayName = branchExtractor.BlankingArrayName
        branchClipper.Execute()

        # Compute vasculature sections
        sections = vmtkscripts.vmtkBranchSections()
        sections.Surface = branchClipper.Surface
        sections.Centerlines = branchExtractor.Centerlines

        sections.AbscissasArrayName = cntAttributes.AbscissasArrayName
        sections.NormalsArrayName   = cntAttributes.NormalsArrayName
        sections.GroupIdsArrayName  = branchExtractor.GroupIdsArrayName
        sections.BlankingArrayName  = branchExtractor.BlankingArrayName
        sections.RadiusArrayName    = branchExtractor.RadiusArrayName
        sections.TractIdsArrayName  = branchExtractor.TractIdsArrayName
        sections.BlankingArrayName  = branchExtractor.BlankingArrayName
        sections.CenterlineIdsArrayName = branchExtractor.CenterlineIdsArrayName

        sections.ReverseDirection = 0
        sections.NumberOfDistanceSpheres = self.SpheresDistance
        sections.Execute()
        
        triangulater = vmtkscripts.vmtkSurfaceTriangle()
        triangulater.Surface = sections.BranchSections
        triangulater.Execute()

        self.Surface = triangulater.Surface

        if self.Remesh:
            # Remeshing the surface with quality triangles
            remesher = vmtkscripts.vmtkSurfaceRemeshing()
            remesher.Surface = self.Surface
            remesher.ElementSizeMode = "area"
            remesher.TargetArea = self.ElementAreaSize 
            remesher.Execute()

            self.Surface = remesher.Surface

if __name__=='__main__':
    main = pypes.pypeMain()
    main.Arguments = sys.argv
    main.Execute()
