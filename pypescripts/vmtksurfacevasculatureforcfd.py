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
from vmtk import pypes
from vmtk import vtkvmtk
from vmtk import vmtkscripts

from vmtk4aneurysms import vascular_operations as vscop
from vmtk4aneurysms.lib import names
from vmtk4aneurysms.lib import polydatatools as tools
from vmtk4aneurysms.pypescripts import v4aScripts

vmtksurfacevasculatureforcfd = 'vmtkSurfaceVasculatureForCFD'

class vmtkSurfaceVasculatureForCFD(pypes.pypeScript):

    def __init__(self):

        pypes.pypeScript.__init__(self)

        # Public member
        self.Surface = None
        self.Centerlines = None
        self.Aneurysm = True
        self.FlowExtensionRatio = 2
        self.Interactive = False

        self.MinResolutionValue = 0.125
        self.MaxResolutionValue = 0.250

        self.InletPrefix = "inlet"
        self.WallPrefix = "wall"
        self.OutletPrefix = "outlet"
        self.SnappyHexMeshFilesDir = None
        self.SnappyFilesExtension = ".stl"
        self.OpenProfilesCentersFile = None

        self.SetScriptName('vmtksurfacevasculatureforcfd')
        self.SetScriptDoc(
            "Treat a surface extracted from an DICOM image to be suitable for "
            "a CFD simulation. Remesh its structure with quality triangle "
            "based on its local diameter, possibly accounting for an existing "
            "aneurysm. Clip the model at specified locations based on its "
            "centerline and input location from the user."
            "The script also allows to write the output as separate files "
            "for use with snappyHexMesh of OpenFOAM."
        )

        self.SetInputMembers([
            ['Surface', 'i', 'vtkPolyData', 1, '',
                'the input surface', 'vmtksurfacereader'],

            ['Centerlines', 'icenterline', 'vtkPolyData', 1, '',
                'the centerlines of the input surface (optional; if not '\
                'passed, it is calculated automatically', 'vmtksurfacereader'],

            ['Aneurysm', 'aneurysm', 'bool', 1, '',
                'to indicate presence of an aneurysm'],

            ['MaxResolutionValue', 'maxresvalue', 'float', 1, '(0.0,)',
                'the maximum resolution value, to avoid large triangles'],

            ['MinResolutionValue', 'minresvalue', 'float', 1, '(0.0,)',
                'the minimum resolution value'],

            ['FlowExtensionRatio' , 'flowextensionratio', 'float', 1, '',
                'controls length of extension as number of profile radius'],

            ['Interactive' , 'interactive', 'bool', 1, '',
                'interactively choose the boundaries to add extensions'],

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
                'the output surface', 'vmtksurfacewriter']
        ])

    def Remesh(self, iterations=10):

        remesher = v4aScripts.vmtkSurfaceVasculatureRemeshing()
        remesher.Surface = tools.Cleaner(self.Surface)
        remesher.Centerlines = self.Centerlines
        remesher.Aneurysm = self.Aneurysm
        remesher.Iterations = iterations
        remesher.MinResolutionValue = self.MinResolutionValue
        remesher.MaxResolutionValue = self.MaxResolutionValue
        remesher.Execute()

        self.Surface = remesher.Surface


    def Execute(self):
        if self.Surface == None:
            self.PrintError('Error: no Surface.')

        # We will have to do 2 remeshings: one to get a better initial surface
        # Remesh with vasculature remshing script
        self.Remesh(iterations=5)

        # Smooth surface with Taubin's algorithm: allow change of boundary
        # points
        self.Surface = tools.SmoothSurface(
                           self.Surface,
                           smooth_boundary=True
                       )

        self.Surface = vscop.ClipVasculature(
                           self.Surface,
                           centerlines=self.Centerlines
                       )

        # Adding flow extensions
        flowExtensions = vmtkscripts.vmtkFlowExtensions()

        flowExtensions.Surface = self.Surface
        flowExtensions.Centerlines = self.Centerlines
        flowExtensions.InterpolationMode = 'thinplatespline'
        flowExtensions.ExtensionMode = 'centerlinedirection'

        # boolean flag which enables computing the length of each
        # flowextension proportional to the mean profile radius
        flowExtensions.AdaptiveExtensionLength = 1
        flowExtensions.ExtensionRatio = self.FlowExtensionRatio
        flowExtensions.Interactive = self.Interactive
        flowExtensions.TransitionRatio = 0.5
        flowExtensions.AdaptiveExtensionRadius = 1
        flowExtensions.AdaptiveNumberOfBoundaryPoints = 1
        flowExtensions.TargetNumberOfBoundaryPoints = 50
        flowExtensions.Sigma = 1.0
        flowExtensions.Execute()

        self.Surface = flowExtensions.Surface

        self.Remesh(iterations=5)

        # Capping surface and saving the open profile caps to separate files
        # as required by snappyHexMesh
        capper = vtkvmtk.vtkvmtkCapPolyData()
        capper.SetInputData(self.Surface)
        capper.SetDisplacement(0.0)
        capper.SetInPlaneDisplacement(0.0)
        capper.SetCellEntityIdsArrayName(names.CellEntityIdsArrayName)
        capper.Update()

        cappedSurface = capper.GetOutput()

        if self.SnappyHexMeshFilesDir is not None:
            # Write wall file after remesh
            tools.WriteSurface(
                self.Surface,
                os.path.join(
                    self.SnappyHexMeshFilesDir,
                    self.WallPrefix + self.SnappyFilesExtension
                )
            )

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

        self.Surface = tools.Cleaner(cappedSurface)

if __name__=='__main__':
    main = pypes.pypeMain()
    main.Arguments = sys.argv
    main.Execute()