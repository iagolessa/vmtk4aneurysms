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

import os
import sys
import vtk

from vmtk import pypes

from vmtk4aneurysms.lib import polydatatools as tools
from vmtk4aneurysms.lib import polydatageometry as geo
from vmtk4aneurysms.lib import foamtovtk as fvtk

from vmtk4aneurysms import hemodynamics as hm

vmtkfoamcomputehemodynamics = 'vmtkFoamComputeHemodynamics'

class vmtkFoamComputeHemodynamics(pypes.pypeScript):

    # Constructor
    def __init__(self):
        pypes.pypeScript.__init__(self)

        self.FoamCasePath = None
        self.WallPatchName = None
        self.PeakSystoleInstant = None
        self.LowDiastoleInstant = None

        self.HemodynamicsSurface = None
        self.PressureSurface = None
        self.TemporalDataFile = None

        self.BloodDensity = 1056.0
        self.ComputePressureStats = False
        self.ComputeGon = False
        self.ComputeAfi = False
        self.ScaleSurface = True

        self.FoamMultiRegion = False
        self.RegionName = None

        self.SetScriptName('vmtkfoamcomputehemodynamics')
        self.SetScriptDoc(
            """compute the hemodynamic wall parameters relted to the WSS from a
            simulation performed in OpenFOAM."""
        )

        self.SetInputMembers([
            ['FoamCasePath', 'foamcasepath', 'str' , 1, '',
                'the path to the OpenFOAM file (.foam) case simulation data'],

            ['WallPatchName', 'wallpatchname', 'str' , 1, '',
                'wall patch name where to compute the hemodynamics'],

            ['PeakSystoleInstant', 'peaksystoleinstant', 'float' , 1, '',
                'peak-systole instant'],

            ['LowDiastoleInstant', 'lowdiastoleinstant', 'float' , 1, '',
                'low-diastole instant'],

            ['BloodDensity', 'blooddensity', 'float', 1, '',
                'the density of blood'],

            ['ComputePressureStats', 'computepressurestats', 'bool', 1, '',
                'activates computations of pressure field stats'],

            ['ComputeGon', 'computegon', 'bool', 1, '',
                'activates computations of the gradient oscillatory number'],

            ['ComputeAfi', 'computeafi', 'bool', 1, '',
                'activates computations of the aneurysm formation indicator'],

            ['ScaleSurface', 'scalesurface', 'bool', 1, '',
                'scale geometry back to millimeters units'],

            ['FoamMultiRegion', 'foammultiregion', 'bool', 1, '',
                'Indicate that the simulation has multiregions'],

            ['RegionName', 'region', 'str' , 1, '',
                'Name of region if multiregion is activated'],

            ['TemporalDataFile', 'otemporaldatafile', 'str', 1, '',
             'file to store the WSS surface-average over time (CSV extension)']
        ])

        self.SetOutputMembers([
            ['HemodynamicsSurface', 'ohemodynamics', 'vtkPolyData', 1, '',
             'the output surface with hemodynamics data', 'vmtksurfacewriter'],

            ['PressureSurface', 'opressure', 'vtkPolyData', 1, '',
             'the output surface with pressure data', 'vmtksurfacewriter']
        ])

    def Execute(self):
        if self.FoamMultiRegion and not self.RegionName:
            raise NameError("Provide valid region name.")

        # Get peak systole and low diastole instant per case
        # For backward compatibiliy
        if not self.PeakSystoleInstant and not self.LowDiastoleInstant:

            instants = hm.GetCardiacCyclePeakAndDiastoleInstants(
                           os.path.dirname(self.FoamCasePath)
                       )

            self.PeakSystoleInstant, self.LowDiastoleInstant = instants

        self.OutputText(
            "Peak and diastole instants {}s {}s".format(
                self.PeakSystoleInstant,
                self.LowDiastoleInstant
            )
        )

        self.HemodynamicsSurface = hm.Hemodynamics(
                                       self.FoamCasePath,
                                       self.PeakSystoleInstant,
                                       self.LowDiastoleInstant,
                                       density=self.BloodDensity,
                                       patch=self.WallPatchName,
                                       compute_gon=self.ComputeGon,
                                       compute_afi=self.ComputeAfi,
                                       multi_region=self.FoamMultiRegion,
                                       region_name=self.RegionName
                                   )

        # Scale the surface back to millimeters (fields are not scaled)
        # before computing the curvatures
        if self.ScaleSurface:
            self.HemodynamicsSurface = tools.ScaleVtkObject(
                                           self.HemodynamicsSurface,
                                           1.0e3
                                       )

        if self.ComputePressureStats:
            self.PressureSurface = hm.PressureTemporalStats(
                                       self.FoamCasePath,
                                       self.PeakSystoleInstant,
                                       self.LowDiastoleInstant,
                                       density=self.BloodDensity,
                                       patch=self.WallPatchName,
                                       multi_region=self.FoamMultiRegion,
                                       region_name=self.RegionName
                                   )


            if self.ScaleSurface:
                self.PressureSurface = tools.ScaleVtkObject(
                                           self.PressureSurface,
                                           1.0e3
                                       )

        # Compute temporal evolution of WSS over the *whole surface*
        fieldAvgOverTime = hm.WssSurfaceAverage(
                               self.FoamCasePath,
                               density=self.BloodDensity,
                               multi_region=self.FoamMultiRegion,
                               region_name=self.RegionName
                           )

        with open(self.TemporalDataFile, "a") as file_:
            file_.write("Time, surfaceAveragedWSS\n")

            for time in sorted(fieldAvgOverTime.keys()):
                value = fieldAvgOverTime.get(time)

                file_.write(
                    "{},{}\n".format(time, value)
                )

if __name__ == '__main__':
    main = pypes.pypeMain()
    main.Arguments = sys.argv
    main.Execute()
