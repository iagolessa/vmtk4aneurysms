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

from vascular_models import HemisphereAneurysm, HemiEllipsoidAneurysm

_SMALL = 1e-6

def addSpaceCapitals(word):
    return re.sub(r"(\w)([A-Z])", r"\1 \2", word)

def relative_diff(value, reference):
    return abs(value - reference)/(reference + _SMALL)

def absolute_diff(value, reference):
    return abs(value - reference)

def sphericity_index(surface_area, volume):

    aux = (18.0*np.pi)**(1.0/3.0)

    return 1.0 - aux*(volume**(2.0/3.0)/surface_area)

class TestAneurysmModule(unittest.TestCase):

    def test_ComputeMetrics(self):
        radius = 4.0
        majorAxis = 2.0*radius
        center = (0, 0, 0)

        # Build aneurysm surface models
        hemisphereIaModel = HemisphereAneurysm(
                                radius,
                                center
                            )

        hemiEllipsoidIaModel = HemiEllipsoidAneurysm(
                                   radius,
                                   majorAxis,
                                   center
                               )

        threeFourthEllipsoid = geo.GenerateThreeFourthEllipsoid(
                                   radius,
                                   majorAxis,
                                   center
                               )

        modelSurfaces = {"hemisphere": hemisphereIaModel.GetSurface(),
                         "half-ellipsoid": hemiEllipsoidIaModel.GetSurface(),
                         "three-fourth-ellipsoid": threeFourthEllipsoid}


        # Define the models of aneurysms: hemisphere and half ellipsoid.
        # Here, the correct values of the metrics assessed are computed analytically

        # HEMISPHERE
        hemisphereModel = {
            "MaximumDiameter":     hemisphereIaModel.GetMaximumDiameter(),
            "NeckDiameter":        hemisphereIaModel.GetNeckDiameter(),
            "MaximumNormalHeight": hemisphereIaModel.GetMaximumNormalHeight(),
            "AneurysmVolume":      hemisphereIaModel.GetAneurysmVolume(),
            "HullVolume":          hemisphereIaModel.GetHullVolume(),
            "OstiumArea":          hemisphereIaModel.GetOstiumArea(),
            "AneurysmSurfaceArea": hemisphereIaModel.GetAneurysmSurfaceArea(),
            "HullSurfaceArea":     hemisphereIaModel.GetHullSurfaceArea(),
            "AspectRatio":         hemisphereIaModel.GetAspectRatio(),
            "ConicityParameter":   hemisphereIaModel.GetConicityParameter(),
            "BottleneckFactor":    hemisphereIaModel.GetBottleneckFactor(),
            "NonSphericityIndex":  hemisphereIaModel.GetNonSphericityIndex(),
            "EllipticityIndex":    hemisphereIaModel.GetEllipticityIndex(),
            "UndulationIndex":     hemisphereIaModel.GetUndulationIndex(),
            "DomeTipPoint":        hemisphereIaModel.GetDomeTipPoint()
        }

        # HALF ELLIPSOIDE
        # Values for the half and the next (three fourth ellipsoid)
        a = radius
        b = majorAxis

        halfEllipsoidModel = {
            "MaximumDiameter":     hemiEllipsoidIaModel.GetMaximumDiameter(),
            "NeckDiameter":        hemiEllipsoidIaModel.GetNeckDiameter(),
            "MaximumNormalHeight": hemiEllipsoidIaModel.GetMaximumNormalHeight(),
            "AneurysmVolume":      hemiEllipsoidIaModel.GetAneurysmVolume(),
            "HullVolume":          hemiEllipsoidIaModel.GetHullVolume(),
            "OstiumArea":          hemiEllipsoidIaModel.GetOstiumArea(),
            "AneurysmSurfaceArea": hemiEllipsoidIaModel.GetAneurysmSurfaceArea(),
            "HullSurfaceArea":     hemiEllipsoidIaModel.GetHullSurfaceArea(),
            "AspectRatio":         hemiEllipsoidIaModel.GetAspectRatio(),
            "ConicityParameter":   hemiEllipsoidIaModel.GetConicityParameter(),
            "BottleneckFactor":    hemiEllipsoidIaModel.GetBottleneckFactor(),
            "NonSphericityIndex":  hemiEllipsoidIaModel.GetNonSphericityIndex(),
            "EllipticityIndex":    hemiEllipsoidIaModel.GetEllipticityIndex(),
            "UndulationIndex":     hemiEllipsoidIaModel.GetUndulationIndex(),
            "DomeTipPoint":        hemiEllipsoidIaModel.GetDomeTipPoint()
        }

        # Three-fourth ellipsoid
        # The computation of the surface area is a little bit mode complecated
        # So this is kind of cheating because in the code the area is also
        # computed with IntegrateAttributes
        tfeSurfaceArea = 269.069 # calculated in ParaView with integrate attributes

        d = 0.5*b
        baseRadius = a*np.sqrt(1.0 - (d/b)**2)

        # Volume calculated based on the volume of an ellipsoid cap
        # https://keisan.casio.com/keisan/image/volume%20of%20an%20ellipsoidal%20cap.pdf
        tfeVolume = (4.0/3.0)*np.pi*(a**2)*b - ((np.pi*(a**2)*(d**2))/(3.0*b**2))*(3.0*b - d)

        threeFourthEllipsoidModel = {
            "AneurysmSurfaceArea": tfeSurfaceArea,
            "HullVolume": tfeVolume,
            "NonSphericityIndex": sphericity_index(tfeSurfaceArea, tfeVolume),
            "MaximumDiameter": 2.0*a,
            "NeckDiameter": 2*baseRadius,
            "OstiumArea": np.pi*(baseRadius**2.0),
            "EllipticityIndex": sphericity_index(tfeSurfaceArea, tfeVolume),
            "UndulationIndex": 0.0,
            "MaximumNormalHeight": 1.5*b,
            "AneurysmVolume": tfeVolume,
            "BottleneckFactor": 2.0/np.sqrt(3),
            "AspectRatio": 3.0/np.sqrt(3),
            "HullSurfaceArea": tfeSurfaceArea,
            "ConicityParameter": 0.1667,
            "DomeTipPoint": (0,0,12)
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
