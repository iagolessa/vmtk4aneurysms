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

from vmtk4aneurysms.lib import constants as const
from vmtk4aneurysms.lib import polydatatools as tools
from vmtk4aneurysms.lib import foamtovtk as fvtk
from vmtk4aneurysms.hemodynamics import GetCardiacCyclePeakAndDiastoleInstants
from vmtk4aneurysms.pypescripts import v4aScripts

vmtkfoamcomputeflowsections = 'vmtkFoamComputeFlowSections'

class vmtkFoamComputeFlowSections(pypes.pypeScript):

    # Constructor
    def __init__(self):
        pypes.pypeScript.__init__(self)

        self.FoamCasePath = None
        self.FieldNames = ["U", "p"]
        self.PeakSystoleInstant = None
        self.LowDiastoleInstant = None
        self.Mesh = None

        self.FoamMultiRegion = False
        self.RegionName = None

        self.SetScriptName('vmtkfoamcomputeflowsections')
        self.SetScriptDoc(
            """get the velocity and pressure fields to the sections from the
            volumetric field (assumes that the flow was computed in
            OpenFOAM) at peak systole and low diastole instants."""
        )

        self.SetInputMembers([
            ['FoamCasePath', 'foamcasepath', 'str' , 1, '',
                'the path to the OpenFOAM file (.foam) case simulation data'],

            ['PeakSystoleInstant', 'peaksystoleinstant', 'float' , 1, '',
                'peak-systole instant'],

            ['LowDiastoleInstant', 'lowdiastoleinstant', 'float' , 1, '',
                'low-diastole instant'],

            ['FieldNames', 'fieldnames', 'str' , -1, '',
                'list of OpenFOAM volumetric field names to be extracted'],

            ['FoamMultiRegion', 'foammultiregion', 'bool', 1, '',
                'Indicate that the simulation has multiregions'],

            ['RegionName', 'region', 'str' , 1, '',
                'Name of region if multiregion is activated']
        ])

        self.SetOutputMembers([
            ['Mesh', 'o', 'vtkUnstructuredGrid', 1, '', 'the output mesh',
             'vmtkmeshwriter']
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
            "Peak and diastole instants {}s {}s \n".format(
                self.PeakSystoleInstant,
                self.LowDiastoleInstant
            )
        )

        # Computing surface temporal statistics
        # Get selected fields from the simulation results
        self.OutputText(
            "Getting fields from OF simulation\n"
        )

        # Read hemodynamics surface
        # Compute volume stats
        self.Mesh, fields = fvtk.GetPatchFieldOverTime(
                                self.FoamCasePath,
                                field_names=self.FieldNames,
                                active_patch_name="", # internalMesh
                                multi_region=self.FoamMultiRegion,
                                region_name=self.RegionName
                            )

        # Computes the statistics of each field
        self.Mesh = fvtk.FieldTimeStats(
                          self.Mesh, # FieldTimeStats operates on a copy, so fine
                          fields,
                          self.PeakSystoleInstant,
                          self.LowDiastoleInstant
                      )

        # Scale the geometry back to millimeters (fields are not scaled)
        self.Mesh = tools.ScaleVtkObject(
                        self.Mesh,
                        const.millimeterToMeterFactor
                    )

if __name__ == '__main__':
    main = pypes.pypeMain()
    main.Arguments = sys.argv
    main.Execute()
