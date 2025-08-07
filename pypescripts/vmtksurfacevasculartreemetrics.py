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
import pandas as pd

from vmtk import pypes
from vmtk import vmtkscripts
from pprint import PrettyPrinter

from vmtk4aneurysms.lib.common import FlattenDict
from vmtk4aneurysms.lib.names import DistanceToNeckArrayName
from vmtk4aneurysms.lib.polydatatools import GetPointArrays

from vmtk4aneurysms.vascular_classes import (
    VascularTree
)

from vmtk4aneurysms.aneurysms import (
    VascularTreeWithLateralAneurysm,
    VascularTreeWithBifurcationAneurysm
)

vmtksurfacevasculartreemetrics = 'vmtkSurfaceVascularTreeMetrics'

class vmtkSurfaceVascularTreeMetrics(pypes.pypeScript):

    # Constructor
    def __init__(self):
        pypes.pypeScript.__init__(self)

        self.Surface = None
        self.Centerlines = None
        self.Aneurysm = True
        self.ComputationMode = "interactive"
        self.AneurysmType    = None
        self.AneurysmStatus  = None
        self.AneurysmLabel   = ""
        self.DomePoint       = None
        self.BifVectors = None

        self.ParentVesselSurface = None
        self.AneurysmSurface     = None
        self.OstiumSurface       = None
        self.HullSurface         = None
        self.VascularInfoFile    = None
        self.AneurysmSacCenterline = None
        self.VascularTreeAttributesDict = None

        self.ShowVascularModel = False

        self.SetScriptName(self.__class__.__name__.lower())
        self.SetScriptDoc('extract vasculature metrics')

        self.SetInputMembers([
            ['Surface','i', 'vtkPolyData', 1, '',
                'the input surface', 'vmtksurfacereader'],

            ['Centerlines', 'icenterline', 'vtkPolyData', 1, '',
                'the centerlines of the input surface (optional; if not '\
                'passed, it is calculated automatically)', 'vmtksurfacereader'],

            ['Aneurysm','aneurysm','bool', 1, '',
             'indicate an aneurysm on the vascular tree'],

            ['AneurysmType','type', 'str', 1, '["lateral", "bifurcation"]',
                'aneurysm type'],

            ['AneurysmStatus','status', 'str', 1, '["ruptured", "unruptured"]',
                'rupture status'],

            ['AneurysmLabel','label', 'str', 1, '',
                'for larger studies, you can pass a label to the aneurysm'],

            ['DomePoint', 'domepoint', 'tuple', -1, '',
             'coordinates of aneurysm dome point'],

            ['ComputationMode','mode', 'str', 1,
                '["interactive", "automatic", "plane"]',
                'mode of neck ostium computation'],

            ['ParentVesselSurface', 'iparentvessel', 'vtkPolyData', 1, '',
                'the parent vessel surface (if not passed, computed externally)',
                'vmtksurfacereader'],

            ['ShowVascularModel','showvascularmodel','bool', 1, '',
             'toggle visualization of the vascular model and aneurysm'],

            ['VascularInfoFile', 'ovascularinfofile', 'str', 1, '',
             'tabular file (.csv) with morphological and hemodynamic data']
        ])

        self.SetOutputMembers([
            ['Surface','o', 'vtkPolyData', 1, '',
                'the output branched surface', 'vmtksurfacewriter'],

            ['AneurysmSurface','oaneurysm','vtkPolyData',1,'',
             'the aneurysm sac surface', 'vmtksurfacewriter'],

            ['OstiumSurface','oostium','vtkPolyData',1,'',
             'the ostium surface generated from the contour scalar neck',
             'vmtksurfacewriter'],

            ['BifVectors', 'obifvectors', 'vtkPolyData', 1, '',
             'the bifurcation vectors', 'vmtksurfacewriter'],

            ['HullSurface','ohull','vtkPolyData',1,'',
             'the ostium surface generated from the contour scalar neck',
             'vmtksurfacewriter'],

            ['AneurysmSacCenterline','osaccenterline','vtkPolyData',1,'',
             'the aneurysm sac centerline', 'vmtksurfacewriter']
        ])

    def Execute(self):
        if not self.Surface:
            self.PrintError('Error: no Surface.')

        if self.Aneurysm and not self.AneurysmType:
            self.PrintError('Error: Aneurysm type must be defined.')

        # Filter input surface
        triangleFilter = vtk.vtkTriangleFilter()
        triangleFilter.SetInputData(self.Surface)
        triangleFilter.Update()

        self.Surface = triangleFilter.GetOutput()

        # Check aneurysm type
        if self.Aneurysm:
            if self.AneurysmType == "lateral":
                vascularClassWithAneurysm = VascularTreeWithLateralAneurysm

            elif self.AneurysmType == "bifurcation":
                vascularClassWithAneurysm = VascularTreeWithBifurcationAneurysm

            else:
                self.PrintError(
                    'Error: Aneurysm type "{}" not recognized.'.format(
                        self.AneurysmType
                    )
                )

            # Generate an aneurysm object
            vascularModel = vascularClassWithAneurysm(
                                self.Surface,
                                centerlines_data=self.Centerlines,
                                clip_aneurysm_mode=self.ComputationMode,
                                dome_point=self.DomePoint
                            )

        else:
            # If no aneurysm, we can use the VascularTree model class
            vascularModel = VascularTree(
                                self.Surface,
                                centerlines_data=self.Centerlines
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

        for bid, branch in enumerate(vascularModel.GetBranches().values()):

            pp.pprint(
                "Branch {}: length {}| area {}".format(
                    bid,
                    branch.GetLength(),
                    branch.GetSurfaceArea()
                )
            )

        nBifs = vascularModel.GetNumberOfBifurcations()

        # Store the data into dict
        dictArterialTreeAttributes = {
            "nBifurcations": nBifs
        }

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

        # Store the angle of first bifurcation
        # dictArterialTreeAttributes.update({
        #     "bifAngle": vascularModel.GetBifurcations()[0].GetDaugtherBranchesAngle()[0]
        # })

        # Compute aneurysm properties
        self.OutputText("Computing metrics of aneurysm models.\n")

        # self.Surface = vascularModel.GetBranchedSurface()
        self.Surface = vascularModel.GetVascularSurface()

        if self.Aneurysm:

            aneurysmModel = vascularModel.GetAneurysm()
            aneurysmModel.ComputeSacRegionsField()

            self.AneurysmSurface = aneurysmModel.GetSurface()
            self.HullSurface     = aneurysmModel.GetHullSurface()
            self.OstiumSurface   = aneurysmModel.GetOstiumSurface()
            self.AneurysmSacCenterline = aneurysmModel.GetSacCenterline()

            aneurysmAttributes = aneurysmModel.GetMorphologyMetrics()

            pp.pprint(
                aneurysmAttributes
            )

            hemodynamicAttributes = aneurysmModel.GetHemodynamicStats()

            pp.pprint(
                hemodynamicAttributes
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

        dictArterialTreeAttributes.update(
            aneurysmAttributes
        )

        dictArterialTreeAttributes.update(
            dict(
                FlattenDict(
                    aneurysmModel.GetHemodynamicStats()
                )
            )
        )

        self.VascularTreeAttributesDict = dictArterialTreeAttributes

        if self.VascularInfoFile is not None:

            pd.DataFrame.from_dict(
                dictArterialTreeAttributes,
                orient="index",
                # columns=["IaLabel"])
            ).to_csv(
                self.VascularInfoFile,
                index=True
            )

if __name__ == '__main__':
    main = pypes.pypeMain()
    main.Arguments = sys.argv
    main.Execute()
