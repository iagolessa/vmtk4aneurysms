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
import vtk
import math
import sys
import numpy as np

from vmtk import vtkvmtk
from vmtk import pypes
from vmtk import vmtkscripts

from vmtk4aneurysms.lib import polydatatools as tools
from vmtk4aneurysms.lib import polydatageometry as geo
from vmtk4aneurysms.lib import centerlines as cl
from vmtk4aneurysms import aneurysms as an
from vmtk4aneurysms import vasculature as vsc
from vmtk4aneurysms.lib import polydatamath as pmath

from vtk.numpy_interface import dataset_adapter as dsa
from vmtk4aneurysms.lib import names

vmtkfoambifurcationflow = 'vmtkFoamBifurcationFlow'

class vmtkFoamBifurcationFlow(pypes.pypeScript):

    # Constructor
    def __init__(self):
        pypes.pypeScript.__init__(self)

        self.BifurcationSectionsSurface = None
        self.Surface = None
        self.Centerlines = None
        self.FlowMesh = None
        self.BifPoint = None
        self.InspectSections = False

        self.VelocityVectorArrayName = "U_peak_systole"
        self.SpheresDistance = 1

        self.SetScriptName('vmtkfoambifurcationflow')
        self.SetScriptDoc(
            """Compute flow rate in each sections off a vascular bifurcation."""
        )

        self.SetInputMembers([
            ['BifurcationSectionsSurface', 'isections', 'vtkPolyData', 1, '',
             'the bifurcation sections surface',
             'vmtksurfacereader'],

            ['Surface', 'i', 'vtkPolyData' , 1, '',
                'if the bifurcation sections are empty, compute from this surface',
                'vmtksurfacereader'],

            ['Centerlines', 'icenterlines', 'vtkPolyData' , 1, '',
                'centerline of the vascular surface',
                'vmtksurfacereader'],

            ['FlowMesh', 'flowmesh', 'vtkUnstructuredGrid' , 1, '',
                'mesh with the flow results', 'vmtkmeshreader'],

            ['SpheresDistance', 'spheresdistance', 'int', 1, '',
                'the number of spheres to be accouted as '
                'distance from the bifurcation to extract sections'],

            ['InspectSections', 'inspectsections', 'bool', 1, '',
                'activate visual inspection of sections']
        ])

        self.SetOutputMembers([
            ['BifurcationSectionsSurface', 'osections', 'vtkPolyData', 1, '',
             'the sections with the flow rate value',
             'vmtksurfacewriter']
        ])

    def Execute(self):

        if not self.BifurcationSectionsSurface and self.Surface:

            branchedCenterlines = cl.CenterlineBranching(self.Centerlines)

            # Branch vascular surface
            surfaceBrancher = vmtkscripts.vmtkBranchClipper()
            surfaceBrancher.Surface = tools.Cleaner(
                                          tools.CleanupArrays(self.Surface),
                                          # iterations=5,
                                          # target_cell_area=0.1,
                                          # preserve_boundary=False
                                      )

            surfaceBrancher.Centerlines = branchedCenterlines
            surfaceBrancher.Execute()

            # Compuse sections
            bifSections = vmtkscripts.vmtkBifurcationSections()
            bifSections.Surface = tools.Cleaner(surfaceBrancher.Surface)
            bifSections.Centerlines = branchedCenterlines
            bifSections.NumberOfDistanceSpheres = self.SpheresDistance
            # bifSections.RadiusArrayName = names.VascularRadiusArrayName
            # bifSections.GroupIdsArrayName = names.vmtkGroupIdsArrayName
            # bifSections.CenterlineIdsArrayName = names.vmtkCenterlineIdsArrayName
            bifSections.Execute()

            self.BifurcationSectionsSurface = bifSections.BifurcationSections

            # Filter the bif. sections to get the section closest to a
            # user-specified bifurcation point
            # Get bifrgucatio group id closest to point
            if not self.BifPoint:

                self.BifPoint = tools.SelectSurfacePoint(
                                    self.Surface,
                                    input_text="Select point at the bifurcation\n"
                                )

            bifGroupId = tools.GetFieldValueAtClosestPoint(
                             self.BifurcationSectionsSurface,
                             self.BifPoint,
                             bifSections.BifurcationSectionBifurcationGroupIdsArrayName
                         )

            self.BifurcationSectionsSurface = tools.ExtractPortion(
                                                   self.BifurcationSectionsSurface,
                                                   bifSections.BifurcationSectionBifurcationGroupIdsArrayName,
                                                   bifGroupId
                                               )

        GroupIdsArrayName = 'BifurcationSectionGroupIds'
        AreaArrayName = "BifurcationSectionArea"
        SectionNormals = "BifurcationSectionNormal"

        # Remesh surface to get field
        remeshedSections = tools.RemeshSurface(
                                self.BifurcationSectionsSurface,
                                target_cell_area=1.0e-3,
                                iterations=5
                            )

        remeshedSections = tools.ResampleFieldsToSurface(
                                self.FlowMesh,
                                remeshedSections,
                                self.VelocityVectorArrayName
                            )

        remeshedSections  = tools.ProjectCellArray(
                                remeshedSections,
                                self.BifurcationSectionsSurface,
                                SectionNormals
                            )

        if self.InspectSections:
            # Render surfaces
            self.vmtkRenderer = vmtkscripts.vmtkRenderer()
            self.vmtkRenderer.Initialize()

            surfaceViewer1 = vmtkscripts.vmtkSurfaceViewer()
            surfaceViewer1.vmtkRenderer = self.vmtkRenderer
            # Add normals here for surface visual smoothing
            surfaceViewer1.Surface = geo.Surface.Normals(self.Surface)
            surfaceViewer1.Opacity = 0.3
            surfaceViewer1.Color = [1.0, 1.0, 1.0]
            surfaceViewer1.Display = 0
            surfaceViewer1.BuildView()

            surfaceViewer2 = vmtkscripts.vmtkSurfaceViewer()
            surfaceViewer2.vmtkRenderer = self.vmtkRenderer
            surfaceViewer2.Surface = self.BifurcationSectionsSurface
            surfaceViewer2.Opacity = 1
            surfaceViewer2.Color = [1.0, 0.0, 0.0]
            surfaceViewer2.Display = 1
            surfaceViewer2.BuildView()

        # Compute normals
        # remeshedSections = geo.Surface.Normals(remeshedSections)
        npSection = dsa.WrapDataObject(remeshedSections)

        # Add normal component of velocity
        sectionNormals  = npSection.GetCellData().GetArray(
                              SectionNormals
                          )

        sectionVelocity = npSection.GetCellData().GetArray(
                              self.VelocityVectorArrayName
                          )

        normalVelocityMag = pmath.HadamardDot(
                                sectionNormals,
                                sectionVelocity
                            )

        normalVelocity = normalVelocityMag*sectionNormals

        npSection.CellData.append(
            normalVelocity,
            "U_normal"
        )

        npSection.CellData.append(
            normalVelocityMag,
            "U_magnitude_normal"
        )

        remeshedSections = npSection.VTKObject

        remeshedSections  = tools.ProjectCellArray(
                                remeshedSections,
                                self.BifurcationSectionsSurface,
                                GroupIdsArrayName
                            )

        npSectionSurface = dsa.WrapDataObject(self.BifurcationSectionsSurface)
        sectionIds = npSectionSurface.GetCellData().GetArray(GroupIdsArrayName)

        flowRateArray = {}

        for idx in sectionIds:
            # Get section surface
            section = tools.ExtractPortion(
                            self.BifurcationSectionsSurface,
                            GroupIdsArrayName,
                            idx
                      )

            sectionArea = section.GetCellData().GetArray(AreaArrayName).GetValue(0)

            remeshedSection = tools.ExtractPortion(
                                    remeshedSections,
                                    GroupIdsArrayName,
                                    idx
                              )

            # Add normal component of velocity
            flowRateArray.update(
                {idx: sectionArea*pmath.SurfaceAverage(
                                        remeshedSection,
                                        "U_magnitude_normal"
                                    )
            })

        npSectionSurface.CellData.append(
            dsa.VTKArray(list(flowRateArray.values())),
            "FlowRate"
        )

        self.BifurcationSectionsSurface = npSectionSurface.VTKObject


if __name__ == '__main__':
    main = pypes.pypeMain()
    main.Arguments = sys.argv
    main.Execute()
