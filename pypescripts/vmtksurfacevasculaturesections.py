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


    def _get_inlet_and_outlets(self):
        """Compute inlet and outlets centers of vascular surface.

        Based on the surface and assuming that the inlet is the open profile
        with largest area, return a tuple with two lists: the first includes
        the coordinates of the inlet and the second the coordinates of the
        outlets.
        """

        boundaryRadiusArrayName  = "Radius"
        boundaryNormalsArrayName = "Normals"
        boundaryCentersArrayName = "Centers"

        boundarySystems = vtkvmtk.vtkvmtkBoundaryReferenceSystems()
        boundarySystems.SetInputData(self.Surface)
        boundarySystems.SetBoundaryRadiusArrayName(boundaryRadiusArrayName)
        boundarySystems.SetBoundaryNormalsArrayName(boundaryNormalsArrayName)
        boundarySystems.SetPoint1ArrayName(boundaryCentersArrayName)
        boundarySystems.SetPoint2ArrayName(boundaryCentersArrayName)
        boundarySystems.Update()

        referenceSystems = boundarySystems.GetOutput()

        nProfiles = referenceSystems.GetNumberOfPoints()

        # Get inlet center (larger radius)
        radiusArray = np.array([
            referenceSystems.GetPointData().GetArray(boundaryRadiusArrayName).GetTuple1(i)
            for i in range(nProfiles)
        ])

        maxRadius = max(radiusArray)
        inletId = int(np.where(radiusArray == maxRadius)[0])
        inletCenter = list(referenceSystems.GetPoint(inletId))

        # Get outlets
        outletCenters = list()

        # Get centers of outlets
        for profileId in range(nProfiles):

            if profileId != inletId:
                outletCenters += referenceSystems.GetPoint(profileId)

        return inletCenter, outletCenters

    # Code partially based on the vmtkcenterlines.py script
    def _generate_centerlines(self):
        """Compute centerlines automatically."""

        # Get inlet and outlet centers of surface
        sourcePoints, targetPoints = self._get_inlet_and_outlets()

        CapDisplacement = 0.0
        FlipNormals     = 0
        CostFunction    = '1/R'
        AppendEndPoints = 1
        CheckNonManifold = 0

        Resampling = 0
        ResamplingStepLength = 1.0
        SimplifyVoronoi = 0

        # Clean and triangulate
        surfaceCleaner = vtk.vtkCleanPolyData()
        surfaceCleaner.SetInputData(self.Surface)
        surfaceCleaner.Update()

        surfaceTriangulator = vtk.vtkTriangleFilter()
        surfaceTriangulator.SetInputConnection(surfaceCleaner.GetOutputPort())
        surfaceTriangulator.PassLinesOff()
        surfaceTriangulator.PassVertsOff()
        surfaceTriangulator.Update()

        # Cap surface
        surfaceCapper = vtkvmtk.vtkvmtkCapPolyData()
        surfaceCapper.SetInputConnection(surfaceTriangulator.GetOutputPort())
        surfaceCapper.SetDisplacement(CapDisplacement)
        surfaceCapper.SetInPlaneDisplacement(CapDisplacement)
        surfaceCapper.Update()

        centerlineInputSurface = surfaceCapper.GetOutput()

        # Get source and target ids of closest point
        sourceSeedIds = vtk.vtkIdList()
        targetSeedIds = vtk.vtkIdList()

        pointLocator = vtk.vtkPointLocator()
        pointLocator.SetDataSet(centerlineInputSurface)
        pointLocator.BuildLocator()

        for i in range(len(sourcePoints)//3):
            point = [sourcePoints[3*i + 0],
                     sourcePoints[3*i + 1],
                     sourcePoints[3*i + 2]]

            id_ = pointLocator.FindClosestPoint(point)
            sourceSeedIds.InsertNextId(id_)

        for i in range(len(targetPoints)//3):
            point = [targetPoints[3*i + 0],
                     targetPoints[3*i + 1],
                     targetPoints[3*i + 2]]

            id_ = pointLocator.FindClosestPoint(point)
            targetSeedIds.InsertNextId(id_)

        # Compute centerlines
        centerlineFilter = vtkvmtk.vtkvmtkPolyDataCenterlines()
        centerlineFilter.SetInputData(centerlineInputSurface)

        centerlineFilter.SetSourceSeedIds(sourceSeedIds)
        centerlineFilter.SetTargetSeedIds(targetSeedIds)

        centerlineFilter.SetRadiusArrayName(self.RadiusArrayName)
        centerlineFilter.SetCostFunction(CostFunction)
        centerlineFilter.SetFlipNormals(FlipNormals)
        centerlineFilter.SetAppendEndPointsToCenterlines(AppendEndPoints)
        centerlineFilter.SetSimplifyVoronoi(SimplifyVoronoi)

        centerlineFilter.SetCenterlineResampling(Resampling)
        centerlineFilter.SetResamplingStepLength(ResamplingStepLength)
        centerlineFilter.Update()

        self.Centerlines = centerlineFilter.GetOutput()

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
            self._generate_centerlines()

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
