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
        surfResolution = 200

        # Aneurysm model ref. volume used by B. Ma, R. E. Harbaugh, e M. L.
        # Raghavan, “Three-dimensional geometrical characterization of cerebral
        # aneurysms.”, Annals of Biomedical Engineering, vol. 32, nº 2, p.
        # 264–273, 2004, doi: 10.1023/B:ABME.0000012746.31343.92
        refVolume  = 67.64 # mm3
        normVolume = const.three*refVolume/const.pi

        hemisphereRadius     = (normVolume/const.two)**(const.one/const.three)
        hEllipsoidMinorAxis  = (normVolume/const.four)**(const.one/const.three)
        tfEllipsoidMinorAxis = (
                                    const.four*refVolume/(const.nine*const.pi)
                               )**(const.one/const.three)

        def addCurvatureArrays(model_surface, label):

            # Get a copy of model surface
            model_surface = tools.CopyVtkObject(model_surface)

            # Create complete sphere and ellipsoid surfaces anc compute
            # numerical curvature
            if label == "hemisphere":
                curvSurface = geo.Surface.Curvatures(
                                  geo.GenerateSphereSurface(
                                      hemisphereRadius,
                                      resolution=surfResolution
                                  )
                              )

            elif label == "hemi-ellipsoid":
                curvSurface = geo.Surface.Curvatures(
                                  geo.GenerateEllipsoid(
                                      hEllipsoidMinorAxis,
                                      const.two*hEllipsoidMinorAxis,
                                      resolution=surfResolution
                                  )
                              )

            elif label == "three-fourth-ellipsoid":
                curvSurface = geo.Surface.Curvatures(
                                  geo.GenerateEllipsoid(
                                      tfEllipsoidMinorAxis,
                                      const.two*tfEllipsoidMinorAxis,
                                      resolution=surfResolution
                                  )
                              )

            else:
                raise ValueError("Unrecognized surface model.")

            for fname in [names.GaussCurvatureArrayName,
                          names.MeanCurvatureArrayName]:

                model_surface = tools.ProjectCellArray(
                                    model_surface,
                                    curvSurface,
                                    fname
                                )

            return model_surface

        # Build aneurysm surface models
        iaModel1 = HemisphereAneurysm(
                       hemisphereRadius,
                       surface_resolution=surfResolution
                   )

        iaModel2 = HemiEllipsoidAneurysm(
                       hEllipsoidMinorAxis,
                       const.two*hEllipsoidMinorAxis,
                       surface_resolution=surfResolution
                   )

        iaModel3 = ThreeFourthEllipsoidAneurysm(
                       tfEllipsoidMinorAxis,
                       const.two*tfEllipsoidMinorAxis,
                       surface_resolution=surfResolution
                   )

        iaModels = {model.GetLabel(): model
                    for model in [iaModel1, iaModel2, iaModel3]}

        # Computation using vmtk4aneurysms
        methods = [param for param in dir(ia.Aneurysm)
                   if param.startswith("Get")]

        # Remove metrics that are not analyzed
        methods.remove("GetSurface")
        methods.remove("GetLabel")
        methods.remove("GetOstiumSurface")
        methods.remove("GetHullSurface")
        methods.remove("GetHemodynamicStats")
        methods.remove("GetLowTAWSSArea")
        methods.remove("GetDomeTipPoint")
        methods.remove("GetCurvatureMetrics")

        # Get methods of each aneurysm model
        print("Computing metrics of aneurysm models.", end="\n")

        dictMorphology = {}

        for label, iaModel in iaModels.items():

            # Initiate aneurysm "measured" with the model surfaces
            iaMeasured = ia.Aneurysm(
                           addCurvatureArrays(
                               iaModel.GetSurface(),
                               label
                           ),
                           label=label + "-measured"
                       )

            modelAttributes = {}
            measuredAttributes = {}

            for method in methods:

                try:
                    # Get attributes of model and measured
                    attrModel    = getattr(iaModel, method)()
                    attrMeasured = getattr(iaMeasured, method)()

                    modelAttributes.update(
                        {method.replace("Get", ''): attrModel}
                    )

                    measuredAttributes.update(
                        {method.replace("Get", ''): attrMeasured}
                    )

                except:
                    print(
                        'Error for case' + iaModel.label + ' in param ' + param
                    )

            # Add the curvature metrics separately
            modelAttributes.update(
                iaModel.GetCurvatureMetrics()
            )

            # Add the curvature metrics separately
            measuredAttributes.update(
                iaMeasured.GetCurvatureMetrics()
            )

            # Store all cases
            dictMorphology.update({
                iaModel.GetLabel() + "-model": modelAttributes,
                iaMeasured.GetLabel(): measuredAttributes,
            })

        # Get metrics names
        metrics = [param.replace("Get", '')
                   for param in methods]

        # Append curvature metrics
        for curv in names.curvMetricsList:
            metrics.append(curv)

        # Store only the higher order ones
        higherOrderMetrics = [param for param in metrics
                              if param.endswith("Volume") \
                              or param.endswith("Area")]

        def select_diff_measure(parameter):
            if parameter in higherOrderMetrics: return relative_diff
            else: return absolute_diff

        # This tolerance was set as an upper threshold, and applied to all
        # metrics irrespective of their nature (either 1D, 2D or 3D) or how the
        # difference was computed (either absolute or relative) so this should
        # be taken into account when judging the validity of each one of the
        # metrics. The highest values of difference measured were for the 2D
        # (area) and 3D (volume) metrics, which is expected.

        # Note that the difference will get smaller as the dicretization used
        # for the surface models decreases.
        tol = 1.0e-2

        for label in iaModels.keys():

            print("\nInspecting {}:".format(label), end="\n")
            print("\t Parameter              Model   Measured    Diff.", end="\n")
            print("\t ---------------------  -----   --------    -----", end="\n")

            for param in metrics:
                model    = dictMorphology[label + "-model"][param]
                measured = dictMorphology[label + "-measured"][param]

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
