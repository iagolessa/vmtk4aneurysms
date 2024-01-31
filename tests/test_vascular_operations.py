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
"""Test the vascular_operations.py module."""

import os
import sys
import vtk
import unittest

import vmtk4aneurysms.vascular_operations as vscop
from vmtk4aneurysms.lib import polydatatools as tools
from vmtk4aneurysms.lib import centerlines as cl
from vmtk4aneurysms.lib import names

# The unit test depends on the functions to load the aneurisk database files,
# available in the aneurisk module in
# https://github.com/iagolessa/AneuriskDatabase
# Make sure that the aneurisk folder is in your PYTHONPATH
from aneurisk import filenames as fn

class TestVascularOperationsModule(unittest.TestCase):

    def test_ComputeICABendLimits(self):
        """Test the case C0001 of the aneurisk repository for 6 bends.

        The algorithm proposed by Piccinelli's work (see docstring of function
        vascular_operations.ComputeICABendsLimits) finds 7 bends for what seems
        to be the case C0001 of the Aneurisk database of intracranial aneurysms
        (a fork is available here
        https://github.com/iagolessa/AneuriskDatabase). This functions tests
        the algorithm implemented in the library for that result.
        """

        caseId = 1
        correctNumberOfBifurcations = 7

        # Load case 1 of aneurisk
        vascularSurface = tools.ReadSurface(
                               fn.path_to_vascular_model_file(caseId)
                           )

        centerlines = tools.ReadSurface(
                          fn.path_to_model_centerline_file(caseId)
                      )

        # Get ICA ref. point for case 1
        icaPoint = fn.get_ref_bif_point(caseId)

        geoCenterlines = cl.ComputeCenterlineGeometry(
                            cl.SmoothCenterline(centerlines)
                        )

        geoCenterlines = cl.CenterlineBranching(geoCenterlines)

        referenceSystems = cl.CenterlineReferenceSystems(geoCenterlines)

        icaBifGroupId = int(
                            vscop._get_field_value_at_closest_point(
                                referenceSystems,
                                icaPoint,
                                names.vmtkGroupIdsArrayName
                            )
                        )

        offsetCenterlines = vscop._robust_offset_centerline(
                                geoCenterlines,
                                referenceSystems,
                                icaBifGroupId
                            )

        bendLimits = vscop.ComputeICABendsLimits(offsetCenterlines)

        self.assertTrue(
            len(bendLimits) == correctNumberOfBifurcations
        )

if __name__=='__main__':
    unittest.main()
