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
import lib.polydatageometry as geo

from vascular_models import (
        HemisphereAneurysm,
        HemiEllipsoidAneurysm,
        ThreeFourthEllipsoidAneurysm
    )

_SMALL = 1e-6

def addSpaceCapitals(word):
    return re.sub(r"(\w)([A-Z])", r"\1 \2", word)

def relative_diff(value, reference):
    return abs(value - reference)/(reference + _SMALL)

def absolute_diff(value, reference):
    return abs(value - reference)

class TestAneurysmModule(unittest.TestCase):

    def test_ComputeMetrics(self):
        # Aneurysm model ref. volume used by B. Ma, R. E. Harbaugh, e M. L.
        # Raghavan, “Three-dimensional geometrical characterization of cerebral
        # aneurysms.”, Annals of Biomedical Engineering, vol. 32, nº 2, p.
        # 264–273, 2004, doi: 10.1023/B:ABME.0000012746.31343.92
        refVolume = 67.64 # mm3

        radius = 4.0
        majorAxis = 2.0*radius
        center = (0, 0, 0)
        surfResolution = 300

        # Build aneurysm surface models
        iaModel1 = HemisphereAneurysm(
                       radius,
                       center,
                       surface_resolution=surfResolution
                   )

        iaModel2 = HemiEllipsoidAneurysm(
                       radius,
                       majorAxis,
                       center,
                       surface_resolution=surfResolution
                   )

        iaModel3 = ThreeFourthEllipsoidAneurysm(
                       radius,
                       majorAxis,
                       center,
                       surface_resolution=surfResolution
                   )

        modelSurfaces = {"hemisphere": iaModel1.GetSurface(),
                         "half-ellipsoid": iaModel2.GetSurface(),
                         "three-fourth-ellipsoid": iaModel3.GetSurface()}

        # HEMISPHERE
        hemisphereModel = {
            "MaximumDiameter":     iaModel1.GetMaximumDiameter(),
            "NeckDiameter":        iaModel1.GetNeckDiameter(),
            "MaximumNormalHeight": iaModel1.GetMaximumNormalHeight(),
            "AneurysmVolume":      iaModel1.GetAneurysmVolume(),
            "HullVolume":          iaModel1.GetHullVolume(),
            "OstiumArea":          iaModel1.GetOstiumArea(),
            "AneurysmSurfaceArea": iaModel1.GetAneurysmSurfaceArea(),
            "HullSurfaceArea":     iaModel1.GetHullSurfaceArea(),
            "AspectRatio":         iaModel1.GetAspectRatio(),
            "ConicityParameter":   iaModel1.GetConicityParameter(),
            "BottleneckFactor":    iaModel1.GetBottleneckFactor(),
            "NonSphericityIndex":  iaModel1.GetNonSphericityIndex(),
            "EllipticityIndex":    iaModel1.GetEllipticityIndex(),
            "UndulationIndex":     iaModel1.GetUndulationIndex(),
            "DomeTipPoint":        iaModel1.GetDomeTipPoint()
        }

        # HALF ELLIPSOIDE
        halfEllipsoidModel = {
            "MaximumDiameter":     iaModel2.GetMaximumDiameter(),
            "NeckDiameter":        iaModel2.GetNeckDiameter(),
            "MaximumNormalHeight": iaModel2.GetMaximumNormalHeight(),
            "AneurysmVolume":      iaModel2.GetAneurysmVolume(),
            "HullVolume":          iaModel2.GetHullVolume(),
            "OstiumArea":          iaModel2.GetOstiumArea(),
            "AneurysmSurfaceArea": iaModel2.GetAneurysmSurfaceArea(),
            "HullSurfaceArea":     iaModel2.GetHullSurfaceArea(),
            "AspectRatio":         iaModel2.GetAspectRatio(),
            "ConicityParameter":   iaModel2.GetConicityParameter(),
            "BottleneckFactor":    iaModel2.GetBottleneckFactor(),
            "NonSphericityIndex":  iaModel2.GetNonSphericityIndex(),
            "EllipticityIndex":    iaModel2.GetEllipticityIndex(),
            "UndulationIndex":     iaModel2.GetUndulationIndex(),
            "DomeTipPoint":        iaModel2.GetDomeTipPoint()
        }

        # Three-fourth ellipsoid
        threeFourthEllipsoidModel = {
            "MaximumDiameter":     iaModel3.GetMaximumDiameter(),
            "NeckDiameter":        iaModel3.GetNeckDiameter(),
            "MaximumNormalHeight": iaModel3.GetMaximumNormalHeight(),
            "AneurysmVolume":      iaModel3.GetAneurysmVolume(),
            "HullVolume":          iaModel3.GetHullVolume(),
            "OstiumArea":          iaModel3.GetOstiumArea(),
            "AneurysmSurfaceArea": iaModel3.GetAneurysmSurfaceArea(),
            "HullSurfaceArea":     iaModel3.GetHullSurfaceArea(),
            "AspectRatio":         iaModel3.GetAspectRatio(),
            "ConicityParameter":   iaModel3.GetConicityParameter(),
            "BottleneckFactor":    iaModel3.GetBottleneckFactor(),
            "NonSphericityIndex":  iaModel3.GetNonSphericityIndex(),
            "EllipticityIndex":    iaModel3.GetEllipticityIndex(),
            "UndulationIndex":     iaModel3.GetUndulationIndex(),
            "DomeTipPoint":        iaModel3.GetDomeTipPoint()
        }


        # Computation using vmtk4aneurysms
        methods = [param for param in dir(ia.Aneurysm)
                   if param.startswith("Get")]

        # Remove metrics that are not analyzed
        methods.remove("GetCurvatureMetrics")
        methods.remove("GetSurface")
        methods.remove("GetOstiumSurface")
        methods.remove("GetHullSurface")
        methods.remove("GetHemodynamicStats")
        methods.remove("GetLowTAWSSArea")
        methods.remove("GetDomeTipPoint")

        # Get methods of each aneurysm model
        print("Computing metrics of aneurysm models.", end="\n")

        dictMorphology = {}

        for label, surface in modelSurfaces.items():

            # Initiate aneurysm model with surfaces
            aneurysm = ia.Aneurysm(surface, label=label)

            attributes = {}
            for method in methods:

                try:
                    # Get only float (exclude surfaces)
                    attr = getattr(aneurysm, method)()

                    attributes.update({method.replace("Get", ''): attr})

                except:
                    print('Error for case' + aneurysm.label + ' in param ' + param)


            # Store all cases
            dictMorphology.update({label + "-measured": attributes})

        # Add values of models
        dictMorphology["hemisphere-model"] = hemisphereModel
        dictMorphology["half-ellipsoid-model"] = halfEllipsoidModel
        dictMorphology["three-fourth-ellipsoid-model"] = threeFourthEllipsoidModel

        # Get metrics names
        metrics = [param.replace("Get", '')
                   for param in methods]

        # Store only the higher order ones
        higherOrderMetrics = [param
                              for param in metrics
                              if param.endswith("Volume") or param.endswith("Area")]

        def select_diff_measure(parameter):
            return relative_diff if parameter in higherOrderMetrics else absolute_diff

        # This tolerance was set as an upper threshold, and applied to all
        # metrics irrespective of their nature (either 1D, 2D or 3D) or how the
        # difference was computed (either absolute or relative) so this should
        # be taken into account when judging the validity of each one of the
        # metrics. The highest values of difference measured were for the 2D
        # (area) and 3D (volume) metrics, which is expected.

        # Note that the difference will get smaller as the dicretization used
        # for the surface models decreases.
        tol = 1e-2

        for label in modelSurfaces.keys():
            print("\nInspecting {}:".format(label), end="\n")
            print("\t Parameter              Model   Measured    Diff.", end="\n")
            print("\t ---------------------  -----   --------    -----", end="\n")

            for param in metrics:
                measured = dictMorphology[label + "-measured"][param]
                model = dictMorphology[label + "-model"][param]

                diff = select_diff_measure(param)(measured, model)

                print(
                    "\t {:22s}    {:4.2f}    {:4.2f}    {:4.2e}".format(
                        addSpaceCapitals(param),
                        model,
                        measured,
                        diff
                    ),
                    end="\n"
                )

                self.assertTrue(diff < tol)


if __name__=='__main__':
    unittest.main()
