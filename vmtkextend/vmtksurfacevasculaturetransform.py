#!/usr/bin/env python

from __future__ import absolute_import #NEEDS TO STAY AS TOP LEVEL MODULE FOR Py2-3 COMPATIBILITY

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

        self.SetScriptName('vmtksurfacevasculaturetransform')
        self.SetScriptDoc('Transform a vasculature surface by rotating and '
                          'translating it in such a way that the inlet '
                          'reference system coincides with the origin and '
                          'the negative z direction vector. The inlet is '
                          'identified as the largest radius open profile. '
                          'The script also adds flow extensions at each '
                          'boundary before the transformation.')

        self.SetInputMembers([
            ['Surface',	'i', 'vtkPolyData', 1, '', 
                 'the input surface', 'vmtksurfacereader'],

            ['Remesh' , 'remesh', 'bool', 1, '', 
                'to apply remeshing procedure before transforming the surface']
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

        # Gettig point, normal, and radius of inlet (largest radius)
        maxRadius = 0

        for point_id in range(refSystems.GetPoints().GetNumberOfPoints()):
            radius = refSystems.GetPointData().GetArray("BoundaryRadius").GetValue(point_id)

            if radius > maxRadius:
                idMaxRadius = point_id
                maxRadius = radius

        # The new origin and outward normal of the inlet surface
        # Eventual TODO: set then as user input
        zero3dVector = [0.0, 0.0, 0.0]
        origin = zero3dVector
        orientation = [0, 0, -1]

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

        # cap surface automatically
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
