#!/usr/bin/env python

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


from __future__ import absolute_import #NEEDS TO STAY AS TOP LEVEL MODULE FOR Py2-3 COMPATIBILITY

import os
import sys
import vtk

from vtk.numpy_interface import dataset_adapter as dsa
from vmtk import vtkvmtk
from vmtk import pypes
from vmtk import vmtkscripts

from vmtk4aneurysms.lib import names
from vmtk4aneurysms.lib import polydatatools as tools

vmtksurfaceclipaddflowextension = 'vmtkSurfaceClipAddFlowExtension'

class vmtkSurfaceClipAddFlowExtension(pypes.pypeScript):

    def __init__(self):

        pypes.pypeScript.__init__(self)

        self.Surface = None
        self.ClipMode = 'interactive'
        self.Clip = True
        self.Remesh = True
        self.EdgeLength = 0.1
        self.Interactive = True
        self.FlowExtensionRatio = 2

        self.InletPrefix = "inlet"
        self.WallPrefix = "wall"
        self.OutletPrefix = "outlet"
        self.SnappyHexMeshFilesDir = None
        self.SnappyFilesExtension = ".stl"
        self.OpenProfilesCentersFile = None

        self.SetScriptName('vmtksurfaceclipaddflowextension')
        self.SetScriptDoc(
            'Generate wall and profiles caps separately for snappyHexMeshing.'
            'The script allows for: interactively clipping the surface to open'
            ' inlet and outlet profiles, add flow extensions, remesh the '
            'extended surface, and write the wall and profiles caps to be '
            'used in CFD meshing with snappyHexMesh.'
        )

        self.SetInputMembers([
            ['Surface', 'i', 'vtkPolyData', 1, '',
                 'the input surface', 'vmtksurfacereader'],

            ['Clip' , 'clip', 'bool', 1, '',
                'to clip surface with a box before adding extensions'],

            ['Remesh' , 'remesh', 'bool', 1, '',
                'to apply remeshing procedure after adding extensions'],

            ['EdgeLength' , 'edgelength', 'float', 1, '',
                'to edgelength for the remesh procedure'],

            # ['ClipMode','clipmode', 'str' , 1,
            #     '["interactive", "centerlinebased"]', 'clip mode'],

            ['Interactive' , 'interactive', 'bool', 1, '',
                'interactively choose the boundaries to add extensions'],

            ['FlowExtensionRatio' , 'flowextensionratio', 'float', 1, '',
                'controls length of extension as number of profile radius'],

            ['SnappyHexMeshFilesDir','writedir', 'str' , 1, '',
                 'write directory path for snappyHexMesh stl files'],

            ['InletPrefix','inletprefix', 'str' , 1, '',
                'inlet files prefix (format: .stl)'],

            ['WallPrefix','wallprefix', 'str' , 1, '',
                'wall files prefix (format: .stl)'],

            ['OutletPrefix','outletprefix', 'str' , 1, '',
                'outlet files prefix (format: .stl)'],

            ['OpenProfilesCentersFile', 'ocentersfile', 'str', 1, '',
             'file to store the centers of each inlet and outlet (CSV extension)']
        ])

        self.SetOutputMembers([
            ['Surface', 'o', 'vtkPolyData', 1, '',
                'the output surface clipped, with flow extensions, and capped',
                'vmtksurfacewriter']
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
        if not self.Surface:
            self.PrintError('Error: no Surface.')

        if not self.SnappyHexMeshFilesDir:
            self.PrintError('Error: no write directory for snappy files.')

        if not self.OpenProfilesCentersFile:
            self.PrintError('Error: no name for open centers files.')

        # Clip surface
        if self.ClipMode == 'interactive':
            self.interactiveClip()

        elif self.ClipMode == 'centerlinebased':
            self.centerlineClip()

        else:
            self.PrintError('Error: clip mode not recognized.')

        # Adding flow extensions
        surfaceFlowExtensions = vmtkscripts.vmtkFlowExtensions()

        surfaceFlowExtensions.Surface = self.Surface
        surfaceFlowExtensions.InterpolationMode = 'thinplatespline' # or linear
        surfaceFlowExtensions.ExtensionMode = 'boundarynormal'      # or centerlinedirection

        # boolean flag which enables computing the length of each
        # flowextension proportional to the mean profile radius
        surfaceFlowExtensions.AdaptiveExtensionLength = 1
        surfaceFlowExtensions.ExtensionRatio = self.FlowExtensionRatio
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
            remesher.PreserveBoundaryEdges = 1
            remesher.Execute()

            self.Surface = remesher.Surface

        # Write wall file after remesh
        tools.WriteSurface(
            self.Surface,
            os.path.join(
                self.SnappyHexMeshFilesDir,
                self.WallPrefix + self.SnappyFilesExtension
            )
        )

        # Capping surface and saving the open profile caps to separate files
        # as required by snappyHexMesh
        capper = vtkvmtk.vtkvmtkCapPolyData()
        capper.SetInputData(self.Surface)
        capper.SetDisplacement(0.0)
        capper.SetInPlaneDisplacement(0.0)
        capper.SetCellEntityIdsArrayName(names.CellEntityIdsArrayName)
        capper.Update()

        cappedSurface = capper.GetOutput()

        # Interactively select the inlet points
        inletPickPoint = tools.PickPointSeedSelector()
        inletPickPoint.SetSurface(cappedSurface)
        inletPickPoint.InputInfo("Select a point on each inlet\n")
        inletPickPoint.Execute()

        inletSeeds  = inletPickPoint.PickedSeeds

        # Compute reference systems to identify outlets
        # Get complete ref systems
        boundarySystems = vtkvmtk.vtkvmtkBoundaryReferenceSystems()
        boundarySystems.SetInputData(self.Surface)
        boundarySystems.SetBoundaryRadiusArrayName('Radius')
        boundarySystems.SetBoundaryNormalsArrayName('BoundaryNormals')
        boundarySystems.SetPoint1ArrayName('Point1')
        boundarySystems.SetPoint2ArrayName('Point2')
        boundarySystems.Update()

        referenceSystems = boundarySystems.GetOutput()

        # get all open profile centers
        openProfilesCenters =  [referenceSystems.GetPoint(idx)
                                for idx in range(referenceSystems.GetNumberOfPoints())]

        # Local only inlets based on user input
        inletCenters = [tools.LocateClosestPointOnPolyData(
                            referenceSystems,
                            inletSeeds.GetPoint(idx)
                        ) for idx in range(inletSeeds.GetNumberOfPoints())]

        # Get outlet centers
        outletCenters = [point for point in openProfilesCenters
                         if point not in inletCenters]

        # Get boundaries contours
        boundaryExtractor = vtkvmtk.vtkvmtkPolyDataBoundaryExtractor()
        boundaryExtractor.SetInputData(self.Surface)
        boundaryExtractor.Update()

        boundaries = boundaryExtractor.GetOutput()

        # Select boundaries and write them with file name based on the
        # identification of inlet and outlet

        # Store each contour with its file name in a dict for later writing
        inletContours = {self.InletPrefix + str(idx + 1): (
                             tools.ExtractConnectedRegion(
                                 boundaries,
                                 method="closest",
                                 closest_point=center
                             ), center
                         ) for idx, center in enumerate(inletCenters)}

        outletContours = {self.OutletPrefix + str(idx + 1): (
                              tools.ExtractConnectedRegion(
                                  boundaries,
                                  method="closest",
                                  closest_point=center
                              ), center
                          ) for idx, center in enumerate(outletCenters)}

        inletContours.update(outletContours)

        # Write them
        for fname, (contour, _) in inletContours.items():

            tools.WriteSurface(
                tools.FillContourWithPlaneSurface(contour),
                os.path.join(
                    self.SnappyHexMeshFilesDir,
                    fname + self.SnappyFilesExtension
                )
            )

        # Write open profile centers info
        with open(self.OpenProfilesCentersFile, "a") as file_:

            file_.write("ProfileType, CenterX, CenterY, CenterZ\n")

            for fname, (_, center) in inletContours.items():

                file_.write(
                    "{}, {}, {}, {}\n".format(
                        fname,
                        center[0],
                        center[1],
                        center[2]
                    )
                )

        self.Surface = cappedSurface

if __name__=='__main__':
    main = pypes.pypeMain()
    main.Arguments = sys.argv
    main.Execute()
