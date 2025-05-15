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
import numpy as np

from vmtk import pypes

from vmtk4aneurysms import hemodynamics as hm
from vmtk4aneurysms.lib import constants as const

vmtkfoamgenerateflowrateprofile = 'vmtkFoamGenerateFlowrateProfile'

class vmtkFoamGenerateFlowrateProfile(pypes.pypeScript):

    # Constructor
    def __init__(self):
        pypes.pypeScript.__init__(self)

        self.NCycles = 3
        self.ThresholdAge = const.ThresholdAge
        self.TimeIncrement = 0.01
        self.BloodDensity = const.bloodDensity
        self.ScaledPressureByDensity = True
        self.MassFlowRate = False

        self.PatientAge = None
        self.InletLocation = None # ica or basilar if posterior circulation

        self.FlowRateDataFile = None
        self.OutflowPressureDataFile = None

        self.SetScriptName('vmtkfoamgenerateflowrateprofile')
        self.SetScriptDoc(
            """generate patient-specific flowrate profile to be used
            with a simulation performed in OpenFOAM. The profile is
            generated based on the patient's age and the location of the
            aneurysm."""
        )

        self.SetInputMembers([
            ['NCycles', 'ncycles', 'int' , 1, '',
                'number of cardiac cycles to save'],

            ['ThresholdAge', 'thresholdage', 'int' , 1, '',
                'threshold age separating young and older patients'],

            ['TimeIncrement', 'timeincrement', 'float' , 1, '',
                'time increment to store the flow rate'],

            ['PatientAge', 'patientage', 'int' , 1, '',
                'patient age'],

            ['InletLocation', 'inletlocation', 'str', 1, '',
                'artery where the inlet flow is located (ica, ba, mca, aca, pca, va, oa)'],

            ['BloodDensity', 'blooddensity', 'float', 1, '',
                'the density of blood'],

            ['ScaledPressureByDensity', 'scaledpressurebydensity', 'bool', 1, '',
                'to divide the pressure by the density of blood'],

            ['MassFlowRate', 'massflowrate', 'bool', 1, '',
                'to output the mass flow rate (kg/s) instead of volume flow rate'],

            ['FlowRateDataFile', 'oflowratefile', 'str', 1, '',
             'text file to store the flow rate profile (no extension)'],

            ['OutflowPressureDataFile', 'opressurefile', 'str', 1, '',
             'text file to store the resistance pressure profile (no extension)'],
        ])

        self.SetOutputMembers([])

    def Execute(self):

        # Generate the inlet flow profile according to age and aneurysm location
        flowRateWaveform = hm.GenerateBloodFlowRateProfile(
                               self.PatientAge,
                               self.InletLocation,
                               time_step=self.TimeIncrement,
                               ncycles=self.NCycles,
                               mass_flow_rate=self.MassFlowRate,
                               blood_density=self.BloodDensity
                           )

        # Get pressure profile (for incompressible OF simulation)
        pressureProfile = hm.ResistanceOutflowPressure(
                                flowRateWaveform,
                                scale_pressure_by_density=self.ScaledPressureByDensity,
                          )

        # Write profiles in OpenFOAM list syntax
        np.savetxt(
            self.FlowRateDataFile,
            flowRateWaveform,
            fmt='\t(%5.3f\t%5.4e)',
            header='//Flow rate profile at {} of a patient with {} years,'\
                   'measured in {}\n{}FlowRate table\n('.format(
                        self.InletLocation.upper(),
                        str(self.PatientAge),
                        "kg/s" if self.MassFlowRate else "m3/s",
                        "mass" if self.MassFlowRate else "volumetric"
                    ),
            footer=');',
            comments=''
        )

        np.savetxt(
            self.OutflowPressureDataFile,
            pressureProfile,
            fmt='\t(%5.3f\t%5.4f)',
            header='//Pressure profile at {} of a patient with {} years [{}]\n'\
                   'uniformValue table\n('.format(
                        self.InletLocation.upper(),
                        str(self.PatientAge),
                        "m^2/s^2" if self.ScaledPressureByDensity else "Pa"
                    ),
            footer=');',
            comments=''
        )

        # Get cardiac period and peak systole and low diastole instants
        cardiacPeriod = hm.GetCardiacCyclePeriod(
                           self.PatientAge,
                           self.InletLocation,
                        )

        ldInstant, psInstant = hm.GetCardiacCyclePeakAndDiastoleInstants(
                                   self.PatientAge,
                                   self.InletLocation,
                                   ncycles=self.NCycles
                               )

        self.OutputText(
            "\nCardiac cycle period: {:.2f} s\n"\
            "Peak-systole instant: {:.2f} s\n"\
            "Low-diastole instant: {:.2f} s\n\n".format(
                cardiacPeriod,
                psInstant,
                ldInstant
            )
        )

if __name__ == '__main__':
    main = pypes.pypeMain()
    main.Arguments = sys.argv
    main.Execute()
