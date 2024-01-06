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

import sys
import vtk
from vtk.numpy_interface import dataset_adapter as dsa

from vmtk import vmtkscripts
from vmtk import vtkvmtk
from vmtk import pypes

from vmtk4aneurysms import vascular_operations as vscop
from vmtk4aneurysms.lib import names

vmtksurfacevasculatureremeshing = 'vmtkSurfaceVasculatureRemeshing'

class vmtkSurfaceVasculatureRemeshing(pypes.pypeScript):

    def __init__(self):

        pypes.pypeScript.__init__(self)

        # Public member
        self.Surface = None
        self.Centerlines = None
        self.Aneurysm = True
        self.Iterations = 10

        self.MinResolutionValue = 0.15
        self.MaxResolutionValue = 0.30

        self.SetScriptName('vmtksurfacevasculatureremeshing')
        self.SetScriptDoc(
            "Script to remesh a surface based on a resolution array defined"
            "on it created based on the thickness array of a vascular"
            "surface. The result is radius-based resolution with the"
            "possibility of further refining an aneurysm if it is present on"
            "the vascular tree. In this case the user must draw the neck 3D"
            "contour on the surface with a drawing utility. The array is"
            "smoothed before the remeshing procedure."
        )

        self.SetInputMembers([
            ['Surface', 'i', 'vtkPolyData', 1, '',
                'the input surface', 'vmtksurfacereader'],

            ['Centerlines', 'icenterline', 'vtkPolyData', 1, '',
                'the centerlines of the input surface (optional; if not '\
                'passed, it is calculated automatically', 'vmtksurfacereader'],

            ['Aneurysm', 'aneurysm', 'bool', 1, '',
                'to indicate presence of an aneurysm'],

            ['Iterations', 'iterations', 'int', 1, '',
                'number of iterations of the remeshing step'],

            ['MaxResolutionValue', 'maxresvalue', 'float', 1, '(0.0,)',
                'the maximum resolution value, to avoid large triangles'],

            ['MinResolutionValue', 'minresvalue', 'float', 1, '(0.0,)',
                'the minimum resolution value']
        ])

        self.SetOutputMembers([
            ['Surface', 'o', 'vtkPolyData', 1, '',
                'the output surface', 'vmtksurfacewriter']
        ])



    def Execute(self):
        if self.Surface == None:
            self.PrintError('Error: no Surface.')

        resolutionArrayName = 'ResolutionArray'

        # Will operate on the triangulated one
        triangulate = vtk.vtkTriangleFilter()
        triangulate.SetInputData(self.Surface)
        triangulate.Update()

        # Clean before smoothing array
        cleaner = vtk.vtkCleanPolyData()
        cleaner.SetInputData(triangulate.GetOutput())
        cleaner.Update()

        self.Surface = cleaner.GetOutput()

        # Remesh based on thickness array on the surface: vessels will have
        # finer cells where the diameter is smaller, whereas the aneruysms gets
        # an intermediate value
        if self.Aneurysm:
            resolutionSurface = vscop.ComputeVasculatureThicknessWithAneurysm(
                                     self.Surface,
                                     self.Centerlines,
                                     thickness_field_name=names.ThicknessArrayName,
                                     neck_comp_mode="interactive",
                                 )

        else:
            resolutionSurface = vscop.ComputeVasculatureThickness(
                                   self.Surface,
                                   self.Centerlines,
                                   thickness_field_name=names.ThicknessArrayName
                                )

        # Map thickness array to resolution array between min and max resolution
        npResolutionSurface = dsa.WrapDataObject(resolutionSurface)

        thicknessArray = npResolutionSurface.GetPointData().GetArray(names.ThicknessArrayName)

        maxThickness = thicknessArray.max()
        minThickness = thicknessArray.min()

        angCoeff = (
                        self.MaxResolutionValue
                      - self.MinResolutionValue
                   )/(maxThickness - minThickness)

        resolutionArray = angCoeff*(thicknessArray - minThickness) + self.MinResolutionValue

        npResolutionSurface.PointData.append(
            resolutionArray,
            names.ThicknessArrayName
        )

        resolutionSurface = npResolutionSurface.VTKObject

        # Remesh procedure
        surfaceRemesh = vmtkscripts.vmtkSurfaceRemeshing()
        surfaceRemesh.Surface = resolutionSurface
        surfaceRemesh.ElementSizeMode = 'edgelengtharray'
        surfaceRemesh.TargetEdgeLengthArrayName = names.ThicknessArrayName
        surfaceRemesh.TargetEdgeLengthFactor = 1.0

        # Disable preserveing boundary edges, since this may impact extrusion
        surfaceRemesh.PreserveBoundaryEdges = 0
        surfaceRemesh.NumberOfIterations = self.Iterations
        surfaceRemesh.OutputText("Remeshing... \n")
        surfaceRemesh.Execute()

        self.Surface = surfaceRemesh.Surface

        # Remove spourious array from final surface
        cellData = self.Surface.GetCellData()
        pointData = self.Surface.GetPointData()

        cellArrays = [ cellData.GetArray(id_).GetName()
                       for id_ in range(cellData.GetNumberOfArrays()) ]

        pointArrays = [ pointData.GetArray(id_).GetName()
                       for id_ in range(pointData.GetNumberOfArrays()) ]

        for array in pointArrays:
            pointData.RemoveArray(array)

        for array in cellArrays:
            cellData.RemoveArray(array)

        # # The resolution array is deleted in the remesh procedure
        # # map it to the final surface to keep the array
        # surfaceProjection = vtkvmtk.vtkvmtkSurfaceProjection()
        # surfaceProjection.SetInputData(self.Surface)
        # surfaceProjection.SetReferenceSurface(resolutionSurface)
        # surfaceProjection.Update()

        # Clean before smoothing array
        cleaner = vtk.vtkCleanPolyData()
        cleaner.SetInputData(self.Surface)
        cleaner.Update()

        self.Surface = cleaner.GetOutput()

if __name__=='__main__':
    main = pypes.pypeMain()
    main.Arguments = sys.argv
    main.Execute()
