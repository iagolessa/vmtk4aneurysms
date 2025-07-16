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
"""Test the vascular_classes.py module."""

import os
import sys
import vtk
import unittest

import numpy as np
import pandas as pd

import vascular_classes as vcls

from lib import names
from lib import polydatatools as tools

# The unit test depends on the functions to load the aneurisk database files,
# available in the aneurisk module in
# https://github.com/iagolessa/AneuriskDatabase
# Make sure that the aneurisk folder is in your PYTHONPATH
from aneurisk import filenames as fn

# TODO: Need to build this geometry here, on the fly
bifurcationModelFile = "./tests/example-data/bifurcation_model.stl"

class TestVasculatureModule(unittest.TestCase):

    # def test_CenterlineClass(self):
    #     """Test the case C0001 of the aneurisk repository for 6 bends.

    #     The algorithm proposed by Piccinelli's work (see docstring of function
    #     vascular_operations.ComputeICABendsLimits) finds 7 bends for what seems
    #     to be the case C0001 of the Aneurisk database of intracranial aneurysms
    #     (a fork is available here
    #     https://github.com/iagolessa/AneuriskDatabase). This functions tests
    #     the algorithm implemented in the library for that result.
    #     """

    #     caseId = 1
    #     correctNumberOfBifurcations = 7

    #     # Load case 1 of aneurisk
    #     vascularSurface = tools.ReadSurface(
    #                            fn.path_to_vascular_model_file(caseId)
    #                        )

    #     centerlines = tools.ReadSurface(
    #                       fn.path_to_model_centerline_file(caseId)
    #                   )

    #     # Get ICA ref. point for case 1
    #     icaPoint = fn.get_ref_bif_point(caseId)

    #     offsetCenterlines = cl.ComputeCenterlinePropertiesOffBifurcation(
    #                             cl.SmoothCenterline(centerlines),
    #                             icaPoint
    #                         )

    #     bendLimits = cl.ComputeICABendsLimits(offsetCenterlines)

    #     self.assertTrue(
    #         len(bendLimits) == correctNumberOfBifurcations
    #     )

    def test_VascularSurface(self):
        """Test the vascular surface class."""

        expectedArraysOnSurface = [
            names.normals,
            names.GaussCurvatureArrayName,
            names.MeanCurvatureArrayName,
            names.LocalShapeTypeArrayName
        ]

        # First test a bifurcation model
        bifurcationSurfaceModel = vcls.VascularSurface.from_file(
                                      bifurcationModelFile
                                  )

        nInlets = len(bifurcationSurfaceModel.GetInletCenters())
        nOutlets = len(bifurcationSurfaceModel.GetOutletCenters())
        arraysOnSurface = bifurcationSurfaceModel.GetCellFields()
        bifurcationVolume = bifurcationSurfaceModel.GetVolume()

        print(f"\nTesting the bifurcation model")

        print(f"Number of inlets: {nInlets}")
        print(f"Number of outlets: {nOutlets}")

        self.assertTrue(
            nInlets == 1
        )

        self.assertTrue(
            nOutlets == 2
        )

        self.assertTrue(
            set(arraysOnSurface) == set(expectedArraysOnSurface)
        )

        # Volume measured by creating a tetra-mesh of the geometry with VMTK
        # Mesh target edge length of mesh setup was 0.1 mm
        refVolume = 1156.52  # mm3
        print(f"Real Bifurcation volume: {bifurcationVolume:.2f} mm3")
        print(f"Ref. bifurcation volume: {refVolume:.2f} mm3")

        self.assertTrue(
           abs((bifurcationVolume - refVolume)/refVolume) < 1.0e-3
        )

        # Load a real case from Aneurisk
        vascularSurfaceModel = vcls.VascularSurface.from_file(
                                   fn.path_to_vascular_model_file(1)
                               )

        nInlets = len(vascularSurfaceModel.GetInletCenters())
        nOutlets = len(vascularSurfaceModel.GetOutletCenters())

        print(f"Testing the real case")

        print(f"Number of inlets: {nInlets}")
        print(f"Number of outlets: {nOutlets}")

        self.assertTrue(
            nInlets == 1
        )

        self.assertTrue(
            nOutlets == 7
        )

        arraysOnSurface = vascularSurfaceModel.GetCellFields()

        self.assertTrue(
            set(arraysOnSurface) == set(expectedArraysOnSurface)
        )

        # Check if the surface has a valid polydata object
        self.assertIsInstance(
            vascularSurfaceModel.GetSurface(),
            names.polyDataType
        )

    def test_VascularTree(self):

        # Generate a report on the vasculature being loaded
        vascularTreeModel = vcls.VascularTree.from_file(bifurcationModelFile)

        # # Inspection
        # tools.ViewSurface(vascularModel.GetSurface().GetSurfaceObject(),
        #                   array_name="Local_Shape_Type")

        # tools.ViewSurface(vascularModel.GetCenterlines())

        expectedFieldsInCenterlines = [
            names.VascularRadiusArrayName,
            names.CurvatureArrayName,
            names.TorsionArrayName,
            names.vmtkFrenetTangentArrayName,
            names.vmtkFrenetNormalArrayName,
            names.vmtkFrenetBinormalArrayName,
            names.vmtkAbscissasArrayName,
            names.vmtkParallelTransportArrayName
        ]

        vascCenterlines = vascularTreeModel.GetCenterlinesObject()

        self.assertTrue(
            set(vascCenterlines.GetPointFields()) == set(expectedFieldsInCenterlines)
        )

        nBifurcations = vascularTreeModel.GetNumberOfBifurcations()

        self.assertTrue(
             nBifurcations == 1
        )

        self.assertTrue(
            len(vascularTreeModel.GetBranches()) == 2*nBifurcations + 1
        )

if __name__=='__main__':
    unittest.main()
