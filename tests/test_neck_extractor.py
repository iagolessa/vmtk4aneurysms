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
from collections import Counter

import lib.constants as const
import lib.polydatatools as tools

from vmtk import vmtkscripts
from lib import names

from neck_extractor import (
    LateralAneurysmRegionExtractor,
    BifurcationAneurysmRegionExtractor,
    InteractiveNeckIdentification,
    Automatic3DNeckIdentification,
    PlaneNeckIdentification
)

bifWithFieldsFile = "./tests/example-data/bifurcation_model_with_aneurysm.vtp"
latSurfaceFile = "./tests/example-data/lateral_model_with_aneurysm.stl"

def compareObjects(s, t):
    """Compare hashable objects s and t."""

    return Counter(s) == Counter(t)

class TestNeckExtractorModule(unittest.TestCase):

    def test_LateralAneurysmNeckExtractor(self):
        vascularSurfaceLateralAneurysm = tools.ReadSurface(latSurfaceFile)
        latDomePoint = (0.3626, 26.75, -0.1255)

        latAneurysmExtractor = LateralAneurysmRegionExtractor(
                                   vascularSurfaceLateralAneurysm,
                                   latDomePoint#,
                                   # healthyVessel -> test without healthy vessel
                               )

        latAneurysmalSurface = latAneurysmExtractor.ExtractAneurysmalRegion()
        latInceptionSurface = latAneurysmExtractor.ExtractAneurysmInceptionRegion()

        # Render surfaces
        self.vmtkRenderer = vmtkscripts.vmtkRenderer()
        self.vmtkRenderer.Initialize()

        # Original vascular surface
        surfaceViewer1 = vmtkscripts.vmtkSurfaceViewer()
        surfaceViewer1.vmtkRenderer = self.vmtkRenderer
        surfaceViewer1.Surface = vascularSurfaceLateralAneurysm
        surfaceViewer1.Opacity = 0.3
        surfaceViewer1.Color = [1.0, 1.0, 1.0]
        surfaceViewer1.Display = 0
        surfaceViewer1.BuildView()

        # Clipped Aneurysmal surface
        clippedAneurysmalSurface = tools.ClipWithScalar(
            latAneurysmalSurface,
            names.AneurysmalRegionArrayName,
            const.zero
        )

        surfaceViewer2 = vmtkscripts.vmtkSurfaceViewer()
        surfaceViewer2.vmtkRenderer = self.vmtkRenderer
        surfaceViewer2.Surface = clippedAneurysmalSurface
        surfaceViewer2.Opacity = 1
        surfaceViewer2.Color = [1.0, 0.0, 0.0]
        surfaceViewer2.Display = 0
        surfaceViewer2.BuildView()

        # Inception region
        surfaceViewer3 = vmtkscripts.vmtkSurfaceViewer()
        surfaceViewer3.vmtkRenderer = self.vmtkRenderer
        surfaceViewer3.Surface = latInceptionSurface
        surfaceViewer3.Opacity = 1.0
        surfaceViewer3.Color = [0.0, 1.0, 0.0]
        surfaceViewer3.Display = 1
        surfaceViewer3.BuildView()

    def test_BifurcationAneurysmNeckExtractor(self):
        vascularSurfaceWithAneurysm = tools.ReadSurface(bifWithFieldsFile)
        bifDomePoint = (33.40, 0.1731, -0.1597)

        bifAneurysmExtractor = BifurcationAneurysmRegionExtractor(
                                   vascularSurfaceWithAneurysm,
                                   bifDomePoint#,
                                   # healthyVessel -> test without healthy vessel
                               )

        bifAneurysmalSurface = bifAneurysmExtractor.ExtractAneurysmalRegion()
        bifInceptionSurface = bifAneurysmExtractor.ExtractAneurysmInceptionRegion()

        # Render surfaces
        self.vmtkRenderer = vmtkscripts.vmtkRenderer()
        self.vmtkRenderer.Initialize()

        # Original vascular surface
        surfaceViewer1 = vmtkscripts.vmtkSurfaceViewer()
        surfaceViewer1.vmtkRenderer = self.vmtkRenderer
        surfaceViewer1.Surface = vascularSurfaceWithAneurysm
        surfaceViewer1.Opacity = 0.3
        surfaceViewer1.Color = [1.0, 1.0, 1.0]
        surfaceViewer1.Display = 0
        surfaceViewer1.BuildView()

        # Clipped Aneurysmal surface
        clippedAneurysmalSurface = tools.ClipWithScalar(
            bifAneurysmalSurface,
            names.AneurysmalRegionArrayName,
            const.zero
        )

        surfaceViewer2 = vmtkscripts.vmtkSurfaceViewer()
        surfaceViewer2.vmtkRenderer = self.vmtkRenderer
        surfaceViewer2.Surface = clippedAneurysmalSurface
        surfaceViewer2.Opacity = 1
        surfaceViewer2.Color = [1.0, 0.0, 0.0]
        surfaceViewer2.Display = 0
        surfaceViewer2.BuildView()

        # Inception region
        surfaceViewer3 = vmtkscripts.vmtkSurfaceViewer()
        surfaceViewer3.vmtkRenderer = self.vmtkRenderer
        surfaceViewer3.Surface = bifInceptionSurface
        surfaceViewer3.Opacity = 1.0
        surfaceViewer3.Color = [0.0, 1.0, 0.0]
        surfaceViewer3.Display = 1
        surfaceViewer3.BuildView()

    def _view_distance_to_neck_field(self, markedNeckSurface):
        # View Marked surface with array
        surfaceViewer = vmtkscripts.vmtkSurfaceViewer()
        surfaceViewer.Surface = markedNeckSurface
        surfaceViewer.ArrayName = names.DistanceToNeckArrayName
        surfaceViewer.ColorMap = 'cooltowarm'
        surfaceViewer.Legend = 1
        surfaceViewer.LegendTitle = 'Distance to Neck'
        surfaceViewer.ScalarRange = [-1, 1]
        surfaceViewer.Execute()

    def _view_neck_clipped_surfaces(self, sacSurface, noSacSurface):
        # Render clipped aneurysm surface
        self.vmtkRenderer = vmtkscripts.vmtkRenderer()
        self.vmtkRenderer.Initialize()

        # Original vascular surface
        surfaceViewer1 = vmtkscripts.vmtkSurfaceViewer()
        surfaceViewer1.vmtkRenderer = self.vmtkRenderer
        surfaceViewer1.Surface = sacSurface
        surfaceViewer1.Opacity = 1
        surfaceViewer1.Color = [1.0, 0.0, 0.0]
        surfaceViewer1.Display = 0
        surfaceViewer1.BuildView()

        surfaceViewer2 = vmtkscripts.vmtkSurfaceViewer()
        surfaceViewer2.vmtkRenderer = self.vmtkRenderer
        surfaceViewer2.Surface = noSacSurface
        surfaceViewer2.Opacity = 1
        surfaceViewer2.Color = [0.0, 1.0, 0.0]
        surfaceViewer2.Display = 1
        surfaceViewer2.BuildView()

    # Testing neck identification strategies
    def test_AneurysmInteractiveNeckStrategy(self):
        vascularSurfaceLateralAneurysm = tools.ReadSurface(latSurfaceFile)

        interactiveNeckStrat = InteractiveNeckIdentification(
                                   vascularSurfaceLateralAneurysm
                               )

        markedNeckSurface = interactiveNeckStrat.MarkAneurysmNeck()
        sacSurface, noSacSurface = interactiveNeckStrat.ClipSac()

        self._view_distance_to_neck_field(markedNeckSurface)
        self._view_neck_clipped_surfaces(sacSurface, noSacSurface)

    def test_LateralAneurysmAutomatic3DNeckStrategy(self):
        print("\nTesting automatic 3D neck identification strategy: lateral aneurysm...")

        vascularSurfaceLateralAneurysm = tools.ReadSurface(latSurfaceFile)
        latDomePoint = (0.3626, 26.75, -0.1255)

        # Build aneurysm region extractor
        latAneurysmExtractor = LateralAneurysmRegionExtractor(
                                   vascularSurfaceLateralAneurysm,
                                   latDomePoint#,
                                   # healthyVessel -> test without healthy vessel
                               )

        automaticNeckStrat = Automatic3DNeckIdentification(
                                   vascularSurfaceLateralAneurysm,
                                   latAneurysmExtractor
                               )

        markedNeckSurface = automaticNeckStrat.MarkAneurysmNeck()
        sacSurface, noSacSurface = automaticNeckStrat.ClipSac()

        self._view_distance_to_neck_field(markedNeckSurface)
        self._view_neck_clipped_surfaces(sacSurface, noSacSurface)

    def test_LateralAneurysmPlaneNeckStrategy(self):
        print("\nTesting plane neck identification strategy: lateral aneurysm...")

        vascularSurfaceLateralAneurysm = tools.ReadSurface(latSurfaceFile)
        latDomePoint = (0.3626, 26.75, -0.1255)

        # Build aneurysm region extractor
        latAneurysmExtractor = LateralAneurysmRegionExtractor(
                                   vascularSurfaceLateralAneurysm,
                                   latDomePoint#,
                                   # healthyVessel -> test without healthy vessel
                               )

        planeNeckStrat = PlaneNeckIdentification(
                               vascularSurfaceLateralAneurysm,
                               latAneurysmExtractor
                           )

        markedNeckSurface = planeNeckStrat.MarkAneurysmNeck()
        sacSurface, noSacSurface = planeNeckStrat.ClipSac()

        self._view_distance_to_neck_field(markedNeckSurface)
        self._view_neck_clipped_surfaces(sacSurface, noSacSurface)

    def test_BifAneurysmWithFieldsInteractiveNeckStrategy(self):
        vascularSurfaceBifAneurysm = tools.ReadSurface(bifWithFieldsFile)


        # Get Fields before
        fieldsOnSurface = tools.GetCellArrays(vascularSurfaceBifAneurysm) + \
                          tools.GetPointArrays(vascularSurfaceBifAneurysm)

        interactiveNeckStrat = InteractiveNeckIdentification(
                                   vascularSurfaceBifAneurysm
                               )

        markedNeckSurface = interactiveNeckStrat.MarkAneurysmNeck()
        sacSurface, noSacSurface = interactiveNeckStrat.ClipSac()

        fieldsAfter = tools.GetCellArrays(markedNeckSurface) + \
                      tools.GetPointArrays(markedNeckSurface)

        print("Fields before: ", fieldsOnSurface)
        print("Fields after: ", fieldsAfter)

        self.assertEqual(
            compareObjects(
                fieldsOnSurface  + [names.DistanceToNeckArrayName],
                fieldsAfter
            ),
            True,
            "Number of fields before and after are correct."
        )

        self._view_distance_to_neck_field(markedNeckSurface)
        self._view_neck_clipped_surfaces(sacSurface, noSacSurface)

    def test_BifurcationAneurysmAutomatic3DNeckStrategy(self):
        print("\nTesting automatic 3D neck identification strategy: bifurcation aneurysm...")

        vascularSurfaceWithAneurysm = tools.ReadSurface(bifWithFieldsFile)
        bifDomePoint = (33.40, 0.1731, -0.1597)

        # Get Fields before
        fieldsOnSurface = tools.GetCellArrays(vascularSurfaceWithAneurysm) + \
                          tools.GetPointArrays(vascularSurfaceWithAneurysm)

        bifAneurysmExtractor = BifurcationAneurysmRegionExtractor(
                                   vascularSurfaceWithAneurysm,
                                   bifDomePoint#,
                                   # healthyVessel -> test without healthy vessel
                               )

        automaticNeckStrat = Automatic3DNeckIdentification(
                                   vascularSurfaceWithAneurysm,
                                   bifAneurysmExtractor
                               )

        markedNeckSurface = automaticNeckStrat.MarkAneurysmNeck()
        sacSurface, noSacSurface = automaticNeckStrat.ClipSac()

        fieldsAfter = tools.GetCellArrays(markedNeckSurface) + \
                      tools.GetPointArrays(markedNeckSurface)

        print("Fields before: ", fieldsOnSurface)
        print("Fields after: ", fieldsAfter)

        self.assertEqual(
            compareObjects(
                fieldsOnSurface  + [names.DistanceToNeckArrayName],
                fieldsAfter
            ),
            True,
            "Number of fields before and after are correct."
        )

        self._view_distance_to_neck_field(markedNeckSurface)
        self._view_neck_clipped_surfaces(sacSurface, noSacSurface)

    def test_BifurcationAneurysmPlaneNeckStrategy(self):
        print("\nTesting plane neck identification strategy: bifurcation aneurysm...")

        vascularSurfaceWithAneurysm = tools.ReadSurface(bifWithFieldsFile)
        bifDomePoint = (33.40, 0.1731, -0.1597)

        # Get Fields before
        fieldsOnSurface = tools.GetCellArrays(vascularSurfaceWithAneurysm) + \
                          tools.GetPointArrays(vascularSurfaceWithAneurysm)

        bifAneurysmExtractor = BifurcationAneurysmRegionExtractor(
                                   vascularSurfaceWithAneurysm,
                                   bifDomePoint#,
                                   # healthyVessel -> test without healthy vessel
                               )

        planeNeckStrat = PlaneNeckIdentification(
                               vascularSurfaceWithAneurysm,
                               bifAneurysmExtractor
                           )

        markedNeckSurface = planeNeckStrat.MarkAneurysmNeck()
        sacSurface, noSacSurface = planeNeckStrat.ClipSac()

        fieldsAfter = tools.GetCellArrays(markedNeckSurface) + \
                      tools.GetPointArrays(markedNeckSurface)

        print("Fields before: ", fieldsOnSurface)
        print("Fields after: ", fieldsAfter)

        self.assertEqual(
            compareObjects(
                fieldsOnSurface  + [names.DistanceToNeckArrayName],
                fieldsAfter
            ),
            True,
            "Number of fields before and after are correct."
        )

        self._view_distance_to_neck_field(markedNeckSurface)
        self._view_neck_clipped_surfaces(sacSurface, noSacSurface)

if __name__=='__main__':
    # Run all test methods
    # unittest.main()

    suite = unittest.TestSuite()
    # suite.addTest(
    #     TestNeckExtractorModule("test_AneurysmInteractiveNeckStrategy")
    # )

    suite.addTest(
        TestNeckExtractorModule("test_BifurcationAneurysmNeckExtractor")
    )
    suite.addTest(
        TestNeckExtractorModule("test_BifAneurysmWithFieldsInteractiveNeckStrategy")
    )
    suite.addTest(
        TestNeckExtractorModule("test_BifurcationAneurysmAutomatic3DNeckStrategy")
    )
    suite.addTest(
        TestNeckExtractorModule("test_BifurcationAneurysmPlaneNeckStrategy")
    )

    runner = unittest.TextTestRunner()
    runner.run(suite)
