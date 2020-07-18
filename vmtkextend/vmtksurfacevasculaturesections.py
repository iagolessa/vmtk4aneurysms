#! /usr/bin/env python

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



    def _get_inlet_and_outlets(self):
        """Compute inlet and outlets centers of vascular surface.

        Based on the surface and assuming that the inlet is the
        open profile with largest area, return a tuple with two lists:
        the first includes the coordinates of the inlet and the second 
        the coordinates of the outlets.
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

    def _generate_centerlines(self):
        """Compute centerlines automatically"""

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


    def Execute(self):

        if self.Clip:
            self.ClipModel()

        # Compute centerlines of model
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