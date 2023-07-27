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
"""Test the aneurysms.py."""

import os
import re
import sys
import vtk
import unittest

import numpy as np
import pandas as pd
import aneurysms as ia
import lib.constants as const
import lib.polydatageometry as geo
import lib.polydatatools as tools

from lib import names
from tests.test_aneurysms import relative_diff, absolute_diff

_SMALL = 1e-6

# Tolerance used for onedim. measures
_1dTol = 1.0e-3
_2dTol = 1.0e-3
_3dTol = 1.0e-3

class TestPolyDataGeometry(unittest.TestCase):

    def test_ContourFunctions(self):
        surfResolution = 200
        origin = (0.0, 0.0, 0.0)

        sphereRadius = 4.0
        ellipsoidMinorAxis = sphereRadius
        ellipsoidMajorAxis = 2.0*ellipsoidMinorAxis

        # "true" values
        contourPerimeter  = const.two*const.pi*sphereRadius
        contourBarycenter = origin
        contourPlaneArea  = const.pi*(sphereRadius**const.two)

        contourHydraulicDiameter = const.two*sphereRadius
        contourAverageDiameter   = const.two*sphereRadius

        sphereSurface = geo.GenerateSphereSurface(
                            sphereRadius,
                            resolution=surfResolution
                        )

        # Create contour: the result is a circle
        contour = tools.ContourCutWithPlane(
                    sphereSurface,
                    (0, 0, 0),
                    (0, 0, 1) #zAxis
                )

        self.assertTrue(
            relative_diff(
                geo.ContourHydraulicDiameter(contour),
                contourHydraulicDiameter
            ) <= _1dTol
        )

        self.assertTrue(
            relative_diff(
                geo.ContourAverageDiameter(contour),
                contourAverageDiameter
            ) <= _1dTol
        )

        self.assertTrue(
            relative_diff(
                geo.ContourPerimeter(contour),
                contourPerimeter
            ) <= _1dTol
        )

        self.assertTrue(
            relative_diff(
                geo.ContourPlaneArea(contour),
                contourPlaneArea
            ) <= _2dTol
        )

        self.assertTrue(
            np.all(
                np.array(
                    relative_diff(
                        np.array(geo.ContourBarycenter(contour)),
                        np.array(contourBarycenter)
                    )
                ) <= np.array(3*[_1dTol])
            )
        )

if __name__=='__main__':
    unittest.main()
