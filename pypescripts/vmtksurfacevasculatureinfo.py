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

import sys
import vtk

from vmtk import pypes
from vmtk import vmtkscripts
from pprint import PrettyPrinter

from vmtk4aneurysms.lib.polydatatools import RemeshSurface
from vmtk4aneurysms.vasculature import Vasculature

vmtksurfacevasculatureinfo = 'vmtkSurfaceVasculatureInfo'

class vmtkSurfaceVasculatureInfo(pypes.pypeScript):

    # Constructor
    def __init__(self):
        pypes.pypeScript.__init__(self)

        self.Surface = None
        self.ComputationMode = "automatic"
        self.AneurysmType    = None
        self.AneurysmStatus  = None
        self.BifVectors = None

        self.ParentVesselSurface = None
        self.AneurysmSurface     = None
        self.OstiumSurface       = None
        self.HullSurface         = None

        self.ShowVascularModel = False

        self.SetScriptName('vmtksurfacevasculatureinfo')
        self.SetScriptDoc('extract vasculature infos')

        self.SetInputMembers([
            ['Surface','i', 'vtkPolyData', 1, '',
                'the input surface', 'vmtksurfacereader'],

            ['AneurysmType','type', 'str', 1, '["lateral", "bifurcation"]',
                'aneurysm type'],

            ['AneurysmStatus','status', 'str', 1, '["ruptured", "unruptured"]',
                'rupture status'],

            ['ComputationMode','mode', 'str', 1,
                '["interactive", "automatic", "plane"]',
                'mode of neck ostium computation'],

            ['ParentVesselSurface', 'iparentvessel', 'vtkPolyData', 1, '',
                'the parent vessel surface (if not passed, computed externally)',
                'vmtksurfacereader'],

            ['ShowVascularModel','showvascularmodel','bool', 1, '',
             'toggle visualization of the vascular model and aneurysm']
        ])

        self.SetOutputMembers([
            ['AneurysmSurface','oaneurysm','vtkPolyData',1,'',
             'the aneurysm sac surface', 'vmtksurfacewriter'],

            ['OstiumSurface','oostium','vtkPolyData',1,'',
             'the ostium surface generated from the contour scalar neck',
             'vmtksurfacewriter'],

            ['BifVectors', 'obifvectors', 'vtkPolyData', 1, '',
             'the bifurcation vectors', 'vmtksurfacewriter'],

            ['HullSurface','ohull','vtkPolyData',1,'',
             'the ostium surface generated from the contour scalar neck',
             'vmtksurfacewriter']
        ])

    def Execute(self):
        if not self.Surface:
            self.PrintError('Error: no Surface.')

        # Filter input surface
        triangleFilter = vtk.vtkTriangleFilter()
        triangleFilter.SetInputData(self.Surface)
        triangleFilter.Update()

        self.Surface = triangleFilter.GetOutput()

        # Generate an aneurysm object
        vascularModel = Vasculature(
                            self.Surface,
                            with_aneurysm=True,
                            clip_aneurysm_mode=self.ComputationMode,
                            parent_vascular_surface=self.ParentVesselSurface,
                            aneurysm_prop={
                                "aneurysm_type": self.AneurysmType,
                                "status": self.AneurysmStatus
                            }
                        )

        pp = PrettyPrinter(depth=3)

        pp.pprint(
            "Inlet centers: {}".format(
                vascularModel.GetInletCenters()
            )
        )

        pp.pprint(
            "Outlet centers: {}".format(
                vascularModel.GetOutletCenters()
            )
        )

        pp.pprint(
            "Number of bifurcations: {}".format(
                vascularModel.GetNumberOfBifurcations()
            )
        )

        for bid, branch in enumerate(vascularModel.GetBranches()):

            pp.pprint(
                "Branch {}: length {}| area {}".format(
                    bid,
                    branch.GetLength(),
                    branch.GetSurfaceArea()
                )
            )

        nBifs = vascularModel.GetNumberOfBifurcations()
        if nBifs > 1 and nBifs != 0:

            # If more than opne bifiurcation, we have to append all data
            # Append bifurcations together
            appendFilter = vtk.vtkAppendPolyData()

            for bif_id in range(vascularModel.GetNumberOfBifurcations()):

                bifurcation = vascularModel.GetBifurcations()[bif_id]

                pp.pprint(
                    "Angle between branches -> bif. {}: {} deg.".format(
                        bif_id,
                        bifurcation.GetDaugtherBranchesAngle()
                    )
                )

                appendFilter.AddInputData(
                    bifurcation.GetBifurcationVectorsObject()
                )

            appendFilter.Update()

            self.BifVectors = appendFilter.GetOutput()

        elif nBifs == 1:

            # The append with a single input was probably yield the wrong
            # result
            bifurcation = vascularModel.GetBifurcations()[0]

            pp.pprint(
                "Angle between branches: {} deg.".format(
                    bifurcation.GetDaugtherBranchesAngle()
                )
            )

            self.BifVectors = bifurcation.GetBifurcationVectorsObject()

        else:
            pass

        # Compute aneurysm properties
        self.OutputText("Computing metrics of aneurysm models.\n")

        aneurysmModel = vascularModel.GetAneurysm()

        self.AneurysmSurface = aneurysmModel.GetSurface()
        self.HullSurface = aneurysmModel.GetHullSurface()
        self.OstiumSurface = aneurysmModel.GetOstiumSurface()

        # Print aneurysm indices and metrics
        methods = [param for param in dir(aneurysmModel)
                   if param.startswith("Get")]

        # Remove metrics that are not analyzed
        methods.remove("GetSurface")
        methods.remove("GetOstiumSurface")
        methods.remove("GetHullSurface")

        attributes = {method.replace("Get",''): getattr(aneurysmModel, method)()
                      for method in methods}

        pp.pprint(
            attributes
        )

        if self.ShowVascularModel:
            # Render surfaces
            self.vmtkRenderer = vmtkscripts.vmtkRenderer()
            self.vmtkRenderer.Initialize()

            surfaceViewer1 = vmtkscripts.vmtkSurfaceViewer()
            surfaceViewer1.vmtkRenderer = self.vmtkRenderer
            surfaceViewer1.Surface = self.Surface
            surfaceViewer1.Opacity = 0.5
            surfaceViewer1.Color = [1.0, 1.0, 0.0]
            surfaceViewer1.Display = 0
            surfaceViewer1.BuildView()

            surfaceViewer2 = vmtkscripts.vmtkSurfaceViewer()
            surfaceViewer2.vmtkRenderer = self.vmtkRenderer
            surfaceViewer2.Surface = self.OstiumSurface
            surfaceViewer2.Opacity = 1
            surfaceViewer2.Color = [0.0, 1.0, 0.0]
            surfaceViewer2.Display = 0
            surfaceViewer2.BuildView()

            surfaceViewer3 = vmtkscripts.vmtkSurfaceViewer()
            surfaceViewer3.vmtkRenderer = self.vmtkRenderer
            surfaceViewer3.Surface = self.AneurysmSurface
            surfaceViewer3.Opacity = 1.0
            surfaceViewer3.Color = [1.0, 0.0, 0.0]
            surfaceViewer3.Display = 0
            surfaceViewer3.BuildView()

            surfaceViewer4 = vmtkscripts.vmtkSurfaceViewer()
            surfaceViewer4.vmtkRenderer = self.vmtkRenderer
            surfaceViewer4.Surface = self.HullSurface
            surfaceViewer4.Opacity = 0.4
            surfaceViewer4.Color = [1.0, 1.0, 1.0]
            surfaceViewer4.Display = 1
            surfaceViewer4.BuildView()

if __name__ == '__main__':
    main = pypes.pypeMain()
    main.Arguments = sys.argv
    main.Execute()
