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
import numpy as np

from vmtk import pypes
from vmtk import vmtkscripts
from vmtk import vtkvmtk

from vmtk4aneurysms.lib import centerlines as cl

vmtksurfacevasculaturesections = 'vmtkSurfaceVasculatureSections'

class vmtkSurfaceVasculatureSections(pypes.pypeScript):

    def __init__(self):

        pypes.pypeScript.__init__(self)

        self.Surface = None
        self.Centerlines = None
        self.Remesh = False
        self.ClipSurface = False
        self.ClipSections = True
        self.SpheresDistance = 1
        self.ElementAreaSize = 0.001

        self.Centerlines = None
        self.RadiusArrayName = "MaximumInscribedSphereRadius"

        self.SetScriptName('vmtksurfacevasculaturesections')
        self.SetScriptDoc("Build vasculature sections separated by a given "
                          "number of incribed spheres.")

        self.SetInputMembers([
            ['Surface', 'i', 'vtkPolyData', 1, '',
                'the input surface', 'vmtksurfacereader'],

            ['Centerlines', 'icenterline', 'vtkPolyData', 1, '',
                'the centerlines of the input surface (optional; if not '\
                'passed, it is calculated automatically', 'vmtksurfacereader'],

            ['Remesh' , 'remesh', 'bool', 1, '',
                'apply remeshing procedure to the sections'],

            ['ClipSurface' , 'clipsurface' ,'bool', 1,'',
                'clip surface with a box before computing sections'],

            ['ClipSections' , 'clipsections' ,'bool', 1,'',
                'clip (potentially worng) sections with a box'],

            ['SpheresDistance', 'spheresdistance', 'int', 1, '',
                'the number of spheres to be accouted as '
                'distance between the sections'],

            ['ElementAreaSize', 'elementareasize', 'float', 1, '',
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

    def Execute(self):

        if self.ClipSurface:
            self.ClipModel()

        cleaner = vtk.vtkCleanPolyData()
        cleaner.SetInputData(self.Surface)
        cleaner.Update()

        triangulate = vtk.vtkTriangleFilter()
        triangulate.SetInputData(cleaner.GetOutput())
        triangulate.Update()

        self.Surface = triangulate.GetOutput()

        if not self.Centerlines:
            self.Centerlines = cl.GenerateCenterlines(self.Surface)

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

        # Include a final clip to remove eventual "slices" that can
        # renders the remeshing wrong
        if self.ClipSections:
            self.ClipModel()

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
