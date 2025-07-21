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
from healthy_vessel_reconstruction import (
    BifurcationAneurysmReconstruction,
    LateralAneurysmReconstruction
)

bifSurfaceFile = "./tests/example-data/bifurcation_model_with_aneurysm.stl"
latSurfaceFile = "./tests/example-data/lateral_model_with_aneurysm.stl"

class TestHealthyVesselReconstructionModule(unittest.TestCase):

    def test_Reconstruction(self):
        vascularSurfaceWithAneurysm = tools.ReadSurface(bifSurfaceFile)
        vascularSurfaceLateralAneurysm = tools.ReadSurface(latSurfaceFile)

        bifDomePoint = (33.40, 0.1731, -0.1597)
        latDomePoint = (0.3626, 26.75, -0.1255)

        bifStrategy = BifurcationAneurysmReconstruction(
                          vascularSurfaceWithAneurysm,
                          dome_point=bifDomePoint
                      )

        print("Reconstructing bifurcation aneurysm healthy vessel...")
        bifHealthyVessel = bifStrategy.Reconstruct()

        latStrategy = LateralAneurysmReconstruction(
                            vascularSurfaceLateralAneurysm,
                            dome_point=latDomePoint
                      )
        print("Reconstructing lateral aneurysm healthy vessel...")
        latHealthyVessel = latStrategy.Reconstruct()

        tools.ViewSurface(bifHealthyVessel)
        tools.ViewSurface(latHealthyVessel)

if __name__=='__main__':
    unittest.main()
