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
"""Test the vasculature.py module."""

import os
import sys
import vtk
import unittest

import numpy as np
import pandas as pd

import vasculature as vsc
from lib import polydatatools as tools

# TODO: I need to build this geometry here, on the fly with some geo module
modelFile = "./tests/example-data/aneurysm_vascular_model.stl"
modelHealthyFile = "./tests/example-data/aneurysm_model_healthy_vasculature.stl"

class TestVasculatureModule(unittest.TestCase):

    def test_VasculatureWithoutAneurysm(self):

        # Generate a report on the vasculature being loaded
        vascularModel = vsc.Vasculature.from_file(
                            modelHealthyFile,
                            with_aneurysm=False
                        )

        # # Inspection
        # tools.ViewSurface(vascularModel.GetSurface().GetSurfaceObject(),
        #                   array_name="Local_Shape_Type")

        # tools.ViewSurface(vascularModel.GetCenterlines())

        # vascularModel.ComputeWallThicknessArray()
        # tools.ViewSurface(vascularModel.GetSurface().GetSurfaceObject(),
        #                   array_name="Thickness")

        # if withAneurysm:
        #     tools.ViewSurface(vascularModel.GetAneurysm().GetSurface())
        #     tools.ViewSurface(vascularModel.GetAneurysm().GetHullSurface())

        necessaryArraysInCenterlines = ["MaximumInscribedSphereRadius",
                                        "Curvature", "Torsion",
                                        "FrenetTangent", "FrenetNormal",
                                        "FrenetBinormal", "Abscissas",
                                        "ParallelTransportNormals"]

        arrays = [vascularModel.GetCenterlines().GetPointData().GetArray(idx).GetName()
                  for idx in range(vascularModel.GetCenterlines().GetPointData().GetNumberOfArrays())]

        self.assertTrue(
            set(arrays) == set(necessaryArraysInCenterlines)
        )

        nBifurcations = vascularModel.GetNumberOfBifurcations()

        self.assertTrue(
             nBifurcations == 1
        )

        self.assertTrue(
            len(vascularModel.GetBranches()) == 2*nBifurcations + 1
        )

    def test_VasculatureWithAneurysm(self):

        # Generate a report on the vasculature being loaded
        vascularModel = vsc.Vasculature.from_file(
                            modelFile,
                            with_aneurysm=True,
                            clip_aneurysm_mode="automatic",
                            parent_vascular_surface=tools.ReadSurface(modelHealthyFile),
                            aneurysm_prop={
                                "aneurysm_type": "bifurcation",
                                "status": "unruptured",
                                "label": "BifurcationModel"
                            }
                        )

        # # Inspection
        # tools.ViewSurface(vascularModel.GetSurface().GetSurfaceObject(),
        #                   array_name="Local_Shape_Type")

        # tools.ViewSurface(vascularModel.GetCenterlines())

        # vascularModel.ComputeWallThicknessArray()
        # tools.ViewSurface(vascularModel.GetSurface().GetSurfaceObject(),
        #                   array_name="Thickness")

        # if withAneurysm:
        #     tools.ViewSurface(vascularModel.GetAneurysm().GetSurface())
        #     tools.ViewSurface(vascularModel.GetAneurysm().GetHullSurface())

        necessaryArraysInCenterlines = ["MaximumInscribedSphereRadius",
                                        "Curvature", "Torsion",
                                        "FrenetTangent", "FrenetNormal",
                                        "FrenetBinormal", "Abscissas",
                                        "ParallelTransportNormals"]

        arraysInCenterline = []

        for index in range(vascularModel.GetCenterlines().GetPointData().GetNumberOfArrays()):
            arrayName = vascularModel.GetCenterlines().GetPointData().GetArray(index).GetName()

            arraysInCenterline.append(arrayName)

        self.assertTrue(
            set(arraysInCenterline) == set(necessaryArraysInCenterlines)
        )

        nBifurcations = vascularModel.GetNumberOfBifurcations()

        self.assertTrue(
             nBifurcations == 1
        )

        self.assertTrue(
            len(vascularModel.GetBranches()) == 2*nBifurcations + 1
        )

if __name__=='__main__':
    unittest.main()
