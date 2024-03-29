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


from __future__ import absolute_import

import sys
import math
import vtk

from vmtk import vmtkscripts
from vmtk import pypes
from vmtk import vtkvmtk

vmtksurfacevasculaturetransform = 'vmtkSurfaceVasculatureTransform'

class vmtkSurfaceVasculatureTransform(pypes.pypeScript):

    def __init__(self):

        pypes.pypeScript.__init__(self)

        self.Surface = None
        self.Remesh = False
        self.Cap = False
        self.AddFlowExtensions = False
        self.Center = [0, 0, 0]
        self.Orientation = [0, 0, -1]

        self.SetScriptName('vmtksurfacevasculaturetransform')
        self.SetScriptDoc('Transform a vasculature surface by rotating and '
                          'translating it in such a way that the inlet '
                          'reference system coincides with the origin and '
                          'the negative z direction vector. The inlet is '
                          'identified as the largest radius open profile. '
                          'The script also adds flow extensions at each '
                          'boundary before the transformation.')

        self.SetInputMembers([
            ['Surface', 'i', 'vtkPolyData', 1, '',
                 'the input surface', 'vmtksurfacereader'],

            ['Remesh' , 'remesh', 'bool', 1, '',
                'to apply remeshing procedure before transforming the surface'],

            ['Cap' , 'cap', 'bool', 1, '',
                'to cap surface after transfrom it'],

            ['AddFlowExtensions' , 'addextensions', 'bool', 1, '',
                'to add short flow extensions before transforming the surface'],

            ['Orientation' , 'orientation', 'float', -1, '',
                'list with the desired orientation (default: [0,0,-1])'],

            ['Center' , 'center', 'float', -1, '',
                'list with the desired translation center (default: [0,0,0])']
        ])

        self.SetOutputMembers([
            ['Surface', 'o', 'vtkPolyData', 1, '',
                'the output surface', 'vmtksurfacewriter']
        ])

    def _compute_angle(self, vector1, vector2):

        vector1 = [vector1[0], vector1[1], vector1[2]]
        vector2 = [vector2[0], vector2[1], vector2[2]]

        vtk.vtkMath.Normalize(vector1)
        vtk.vtkMath.Normalize(vector2)

        dotProduct = sum((a*b) for a, b in zip(vector1, vector2))
        radAngle = math.acos(dotProduct)

        return 180.0*radAngle/vtk.vtkMath.Pi()

    def Execute(self):
        if self.Surface == None:
            self.PrintError('Error: no Surface.')

        if self.AddFlowExtensions:
            surfaceFlowExtensions = vmtkscripts.vmtkFlowExtensions()
            surfaceFlowExtensions.Surface = self.Surface
            surfaceFlowExtensions.InterpolationMode = 'thinplatespline'
            surfaceFlowExtensions.ExtensionMode = 'boundarynormal'
            # boolean flag which enables computing the length of each
            # flowextension proportional to the mean profile radius
            surfaceFlowExtensions.AdaptiveExtensionLength = 1
            # The proportionality factor is set through 'extensionratio'
            surfaceFlowExtensions.ExtensionRatio = 2
            surfaceFlowExtensions.Interactive = 0
            surfaceFlowExtensions.TransitionRatio = 0.5
            surfaceFlowExtensions.AdaptiveExtensionRadius = 1
            surfaceFlowExtensions.AdaptiveNumberOfBoundaryPoints = 1
            surfaceFlowExtensions.TargetNumberOfBoundaryPoints = 50
            surfaceFlowExtensions.Sigma = 1.0
            surfaceFlowExtensions.Execute()

            self.Surface = surfaceFlowExtensions.Surface

        if self.Remesh:
            surfaceRemesh = vmtkscripts.vmtkSurfaceRemeshing()
            surfaceRemesh.Surface = self.Surface
            surfaceRemesh.ElementSizeMode = 'edgelength'
            surfaceRemesh.TargetEdgeLength = 0.1
            surfaceRemesh.TargetEdgeLengthFactor = 1
            surfaceRemesh.PreserveBoundaryEdges = 1
            surfaceRemesh.Execute()

            self.Surface = surfaceRemesh.Surface

        boundaryReferenceSystems = vtkvmtk.vtkvmtkBoundaryReferenceSystems()
        boundaryReferenceSystems.SetInputData(self.Surface)
        boundaryReferenceSystems.SetBoundaryRadiusArrayName("BoundaryRadius")
        boundaryReferenceSystems.SetBoundaryNormalsArrayName("BoundaryNormals")
        boundaryReferenceSystems.SetPoint1ArrayName("Point1Array")
        boundaryReferenceSystems.SetPoint2ArrayName("Point2Array")
        boundaryReferenceSystems.Update()

        refSystems = boundaryReferenceSystems.GetOutput()

        # Getting point, normal, and radius of inlet (largest radius)
        maxRadius = 0
        idMaxRadius = 1

        for point_id in range(refSystems.GetPoints().GetNumberOfPoints()):
            radius = refSystems.GetPointData().GetArray("BoundaryRadius").GetValue(point_id)

            if radius > maxRadius:
                idMaxRadius = point_id
                maxRadius = radius

        # The new origin and outward normal of the inlet surface
        zero3dVector = [0.0, 0.0, 0.0]
        origin = self.Center
        orientation = self.Orientation

        currentOrigin = refSystems.GetPoint(idMaxRadius)
        currentNormal1 = refSystems.GetPointData().GetArray("BoundaryNormals").GetTuple3(idMaxRadius)
        currentNormal2 = refSystems.GetPointData().GetArray("BoundaryNormals").GetTuple3(idMaxRadius)

        # Transform surface with the inlet going to the origin of the system
        # and its outer normal became the negative z direction (0,0,-1)
        transform = vtk.vtkTransform()
        transform.PostMultiply()

        # Compute translation vector
        translation = zero3dVector
        translation[0] = origin[0] - currentOrigin[0]
        translation[1] = origin[1] - currentOrigin[1]
        translation[2] = origin[2] - currentOrigin[2]

        transform.Translate(translation)

        # Compute axis around which the rotation will happen
        cross = zero3dVector
        vtk.vtkMath.Cross(currentNormal1, orientation, cross)
        vtk.vtkMath.Normalize(cross)

        # Compute angle
        angle = self._compute_angle(currentNormal1, orientation)
        transform.RotateWXYZ(angle, cross)

        # Transform the second normal
        transformedNormal2 = transform.TransformNormal(currentNormal2)
        vtk.vtkMath.Cross(transformedNormal2, orientation, cross)
        vtk.vtkMath.Normalize(cross)
        angle = self._compute_angle(transformedNormal2, orientation)

        transform.RotateWXYZ(angle, cross)

        transformFilter = vtk.vtkTransformPolyDataFilter()
        transformFilter.SetInputData(self.Surface)
        transformFilter.SetTransform(transform)
        transformFilter.Update()

        surfaceTranformed = transformFilter.GetOutput()

        self.Surface = surfaceTranformed

        # cap surface automatically
        if self.Cap:
            capper = vmtkscripts.vmtkSurfaceCapper()
            capper.Surface = surfaceTranformed
            capper.Method  = 'centerpoint'
            capper.Interactive = 0
            capper.Execute()

            self.Surface = capper.Surface

if __name__=='__main__':
    main = pypes.pypeMain()
    main.Arguments = sys.argv
    main.Execute()
