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
from vmtk4aneurysms.pypescripts import v4aScripts

vmtkfoamcomputeflowsections = 'vmtkFoamComputeFlowSections'

class vmtkFoamComputeFlowSections(pypes.pypeScript):

    # Constructor
    def __init__(self):
        pypes.pypeScript.__init__(self)

        self.FoamCasePath = None
        self.WallPatchName = None
        self.FieldNames = ["U", "p"]

        self.SectionsSurface = None
        self.SpheresDistance = 1

        self.BloodDensity = 1056.0
        self.FoamMultiRegion = False
        self.RegionName = None

        self.SetScriptName('vmtkfoamcomputeflowsections')
        self.SetScriptDoc(
            """compute the sections of a vascular surface and interpolate the
            velocity and pressure fields to the sections from the volumetric
            field (assumes that the flow was computed in OpenFOAM)."""
        )

        self.SetInputMembers([
            ['FoamCasePath', 'foamcasepath', 'str' , 1, '',
                'the path to the OpenFOAM file (.foam) case simulation data'],

            ['WallPatchName', 'wallpatchname', 'str' , 1, '',
                'wall patch name where to compute the hemodynamics'],

            ['FieldNames', 'fieldnames', 'str' , -1, '',
                'list of OpenFOAM volumetric field names to be extracted'],

            ['SectionsSurface', 'isections', 'vtkPolyData', 1, '',
             'the vasculature sections surface (if not passed, computed automatically)',
             'vmtksurfacereader'],

            ['SpheresDistance', 'spheresdistance', 'int', 1, '',
                'the number of spheres to be accouted as '
                'distance between the sections (if file sections not passed)'],

            ['FoamMultiRegion', 'foammultiregion', 'bool', 1, '',
                'Indicate that the simulation has multiregions'],

            ['RegionName', 'region', 'str' , 1, '',
                'Name of region if multiregion is activated']
        ])

        self.SetOutputMembers([
            ['SectionsSurface', 'osections', 'vtkPolyData', 1, '',
             'the vasculature sections surface with the velocity and pressure',
             'vmtksurfacewriter']
        ])

    def _folders_in(self, path_to_parent):
        """List directories in a path."""

        for fname in os.listdir(path_to_parent):
            if os.path.isdir(os.path.join(path_to_parent,fname)):
                yield fname

    def _get_peak_instant(self, case_folder):
        """Get peak and low diastole instants of aneurysm simulation."""

        timeFolders = list(self._folders_in(case_folder))

        if '2' in timeFolders:
            return (2, 2.81)

        elif '1.06' in timeFolders:
            return (1.06, 1.87)

        else:
            return (0.12, 0.93)

    def Execute(self):
        if self.FoamMultiRegion and not self.RegionName:
            raise NameError("Provide valid region name.")

        # Get peak systole and low diastole instant per case
        peakSystoleInstant, lowDiastoleInstant = self._get_peak_instant(
                                                     os.path.dirname(self.FoamCasePath)
                                                 )

        self.OutputText(
            "Peak and diastole instants {}s {}s \n".format(
                peakSystoleInstant,
                lowDiastoleInstant
            )
        )

        # Computing surface temporal statistics
        # Get selected fields from the simulation results
        self.OutputText(
            "Getting fields from OF simulation\n"
        )

        # Read hemodynamics surface
        # Compute volume stats
        emptyVolume, fields = fvtk.GetPatchFieldOverTime(
                                  self.FoamCasePath,
                                  field_names=self.FieldNames,
                                  active_patch_name="", # internalMesh
                                  multi_region=self.FoamMultiRegion,
                                  region_name=self.RegionName
                              )

        # Computes the statistics of each field
        statsVolume = fvtk.FieldTimeStats(
                          emptyVolume, # FieldTimeStats operates on a copy, so fine
                          fields,
                          peakSystoleInstant,
                          lowDiastoleInstant
                      )

        # Scale the geometry back to millimeters (fields are not scaled)
        statsVolume = tools.ScaleVtkObject(
                          statsVolume,
                          const.millimeterToMeterFactor
                      )

        if not self.SectionsSurface:
            self.OutputText(
                "No sections file passed. Computing it automatically.\n"
            )

            # Get wall surface
            wallSurface, _ = fvtk.GetPatchFieldOverTime(
                                      self.FoamCasePath,
                                      field_names=[],
                                      active_patch_name=self.WallPatchName,
                                      multi_region=self.FoamMultiRegion,
                                      region_name=self.RegionName
                                  )

            wallSurface = tools.ScaleVtkObject(
                              wallSurface,
                              const.millimeterToMeterFactor
                          )

            # Compute the sections surface
            computeSections = v4aScripts.vmtkSurfaceVasculatureSections()
            computeSections.Surface = wallSurface
            computeSections.Remesh  = True
            computeSections.ClipBefore = False
            computeSections.ClipSections = False
            computeSections.SpheresDistance = self.SpheresDistance
            computeSections.Execute()

            self.SectionsSurface = computeSections.Surface

        # Finally, resample to mid surface
        self.SectionsSurface = tools.ResampleFieldsToSurface(
                                   statsVolume,
                                   self.SectionsSurface
                               )

if __name__ == '__main__':
    main = pypes.pypeMain()
    main.Arguments = sys.argv
    main.Execute()
