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
        self.ThresholdAge = 50
        self.TimeIncrement = 0.01
        self.BloodDensity = const.bloodDensity
        self.ScaledPressureByDensity = True

        self.PatientAge = None
        self.AneurysmLocation = None # ica or basilar if posterior circulation

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

            ['AneurysmLocation', 'aneurysmlocation', 'str', 1, '',
                'aneurysm main tree location (ica or basilar)'],

            ['BloodDensity', 'blooddensity', 'float', 1, '',
                'the density of blood'],

            ['FlowRateDataFile', 'oflowratefile', 'str', 1, '',
             'text file to store the flow rate profile (no extension)'],

            ['OutflowPressureDataFile', 'opressurefile', 'str', 1, '',
             'text file to store the resistance pressure profile (no extension)'],
        ])

        self.SetOutputMembers([])

    def _patient_is_older(self, age: int) -> bool:
        """Return whether patine tis older.

        Older patients are classified here based on Hoi et al. (2010) study whose
        subjects had an average age of 68 +- 8 years and Ford et al. (2005) study
        whose subjects had an average age of 28 +- 7 years. We classified, then,
        young patients with age < 40 and older patients with age > 40, for
        classification purposes.
        """

        return age >= self.ThresholdAge

    def _patient_is_young(self, age: int) -> bool:
        """Return whether patine is young.

        Older patients are classified here based on Hoi et al. (2010) study whose
        subjects had an average age of 68 +- 8 years and Ford et al. (2005) study
        whose subjects had an average age of 28 +- 7 years. We classified, then,
        young patients with age < 40 and older patients with age > 40, for
        classification purposes.
        """

        return age < self.ThresholdAge

    def _select_norm_flow_rate(
            self,
            patient_age,
            ia_location
        ):

        if self._patient_is_older(patient_age) and ia_location == "basilar":
            # Profile not avaible for older patients. Use the ICA one
            profileType = "measured_ica_older"
            Qavg = const.BfrAvgBAOlderAdults

        elif self._patient_is_young(patient_age) and ia_location == "basilar":
            profileType = "measured_va_young"
            Qavg = const.BfrAvgBAYoungAdults

        elif self._patient_is_older(patient_age) and ia_location != "basilar":
            profileType = "measured_ica_older"
            Qavg = const.BfrAvgICAOlderAdults

        elif self._patient_is_young(patient_age) and ia_location != "basilar":
            profileType = "measured_ica_young"
            Qavg = const.BfrAvgICAYoungAdults

        else:
            raise ValueError("Patient age and aneurysm location not identified.")

        return profileType, Qavg

    def Execute(self):

        # Generate blood flow rate profile for the case
        profileType, Qavg = self._select_norm_flow_rate(
                                self.PatientAge,
                                self.AneurysmLocation
                            )

        # Generate the inlet flow profile according to age and aneurysm location
        normFlowProfile = hm.GenerateBloodFlowRateProfile(
                              time_step=self.TimeIncrement,
                              ncycles=self.NCycles,
                              profile_type=profileType,
                              Qavg=Qavg
                          )

        # Get pressure profile (for incompressible OF simulation)
        pressureProfile = hm.ResistanceOutflowPressure(
                                normFlowProfile,
                                scale_pressure_by_density=self.ScaledPressureByDensity,
                          )

        # Write profiles in OpenFOAM list syntax
        np.savetxt(
            self.FlowRateDataFile,
            normFlowProfile,
            fmt='\t(%5.3f\t%5.4e)',
            header='//Flow rate profile at {} of {} patients dimensionalized by '\
                   '{:.3e} m3/s\nvolumetricFlowRate table\n('.format(
                        self.AneurysmLocation.upper(),
                        "older" if self._patient_is_older(self.PatientAge) else "young",
                        Qavg
                    ),
            footer=');',
            comments=''
        )

        np.savetxt(
            self.OutflowPressureDataFile,
            pressureProfile,
            fmt='\t(%5.3f\t%5.4f)',
            header='//Pressure profile at {} of {} patients\n'\
                   'uniformValue table\n('.format(
                        self.AneurysmLocation.upper(),
                        "older" if self._patient_is_older(self.PatientAge) else "young"
                    ),
            footer=');',
            comments=''
        )

        ldInstant, psInstant = hm.GetCardiacCyclePeakAndDiastoleInstants(
                                   profile_type=profileType,
                                   ncycles=self.NCycles
                               )

        self.OutputText(
            "\nPeak-systole instant: {:.3f} s\n"\
            "Low-diastole instant: {:.3f} s\n\n".format(
                psInstant,
                ldInstant
            )
        )

if __name__ == '__main__':
    main = pypes.pypeMain()
    main.Arguments = sys.argv
    main.Execute()
