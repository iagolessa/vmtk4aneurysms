#! /usr/bin/env python3

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


import argparse

def generate_arg_parser():
    """Creates and return a parser object for this app."""

    parser = argparse.ArgumentParser(
                 description="""Compute a projection of velocity and pressure
                             to sections surface of a vascular surface.

                             The script builds a surface with the projection
                             of the velocity and pressure fields on a surface
                             composed of sections to a the vascular surface
                             where the flow was computed. """
             )

    parser.add_argument(
        '--case',
        help="The path to the OpenFOAM case",
        type=str,
        required=True
    )

    parser.add_argument(
        '--ofile',
        help="Output sections file with U and p fields.",
        type=str,
        required=True
    )

    # Optional
    parser.add_argument(
        '--patch',
        help="The patch where to compute the hemodynamics (default: 'wall')",
        type=str,
        required=False,
        default="wall"
    )

    parser.add_argument(
        '--sectionsfile',
        help="The sections surface of the simulated case (in millimeters)",
        type=str,
        required=False,
        default=None
    )

    parser.add_argument(
        '--multiregion',
        help="Indicate that the simulation has multiregions",
        dest="multiregion",
        action="store_true",
    )

    parser.add_argument(
        '--no-multiregion',
        help="Indicate that the simulation has a single region",
        dest="multiregion",
        action="store_false",
    )

    parser.set_defaults(multiregion=False)

    parser.add_argument(
        '--region',
        help="Name of region if multiregion is True",
        type=str,
        choices=["solid", "fluid"],
        required=False,
        default=""
    )

    return parser

# Parse arguments
argsParser = generate_arg_parser()
args = argsParser.parse_args()

import os
import vtk
import sys

from pathlib import Path
from vmtk import vmtkscripts

from vmtk4aneurysms.pypescripts import v4aScripts
import vmtk4aneurysms.lib.polydatatools as tools
import vmtk4aneurysms.lib.foamtovtk as fvtk

def folders_in(path_to_parent):
    """List directories in a path."""

    for fname in os.listdir(path_to_parent):
        if os.path.isdir(os.path.join(path_to_parent,fname)):
            yield fname

def get_peak_instant(case_folder):
    timeFolders = list(folders_in(case_folder))

    if '2' in timeFolders:
        return (2, 2.81)

    elif '1.06' in timeFolders:
        return (1.06, 1.87)

    else:
        return (0.12, 0.93)

# Names
foamFolder       = args.case
surfacePatch     = args.patch
outSectionsFile  = args.ofile
multiRegion      = args.multiregion

sectionsFile     = args.sectionsfile

if multiRegion == True and args.region == "":
    raise NameError("Provide valid region name.")
else:
    regionName = args.region

fieldNames = ["U", "p"]

# Create dumb case.foam file, if does not exist
# Pathlib will just update mod time if the file already exists
foamFile = os.path.join(foamFolder, "case.foam")
Path(foamFile).touch(exist_ok=True)

# Get peak systole and low diastole instant per case
peakSystoleInstant, lowDiastoleInstant = get_peak_instant(foamFolder)

print(
    "Peak and diastole instants {}s {}s".format(
        peakSystoleInstant,
        lowDiastoleInstant
    ),
    end="\n"
)

# Computing surface temporal statistics
# Get selected fields from the simulation results
print(
    "Getting fields from OF simulation...",
    end="\n"
)

# Read hemodynamics surface
# Compute volume stats
emptyVolume, fields = fvtk.GetPatchFieldOverTime(
                          foamFile,
                          field_names=fieldNames,
                          active_patch_name="", # internalMesh
                          multi_region=multiRegion,
                          region_name=regionName
                      )

# Computes the statistics of each field
statsVolume = fvtk.FieldTimeStats(
                  emptyVolume, # FieldTimeStats operates on a copy, so fine
                  fields,
                  peakSystoleInstant,
                  lowDiastoleInstant
              )

# Scale the surface back to millimeters (fields are not scaled)
scaleMeterToMM = 1.0e3
statsVolume = tools.ScaleVtkObject(statsVolume, scaleMeterToMM)

if not sectionsFile:
    print(
        "No sections file passed, computing it based on", surfacePatch, "patch",
        end="\n"
    )

    # Get wall surface
    wallSurface, _ = fvtk.GetPatchFieldOverTime(
                              foamFile,
                              field_names=[],
                              active_patch_name=surfacePatch,
                              multi_region=multiRegion,
                              region_name=regionName
                          )

    wallSurface = tools.ScaleVtkObject(wallSurface, scaleMeterToMM)

    # Compute the sections surface
    computeSections = v4aScripts.vmtkSurfaceVasculatureSections()
    computeSections.Surface = wallSurface
    computeSections.Remesh  = True
    computeSections.ClipBefore = False
    computeSections.ClipSections = False
    computeSections.Execute()

    sectionsSurface = computeSections.Surface

else:
    sectionsSurface = tools.ReadSurface(sectionsFile)


# Finally, resample to mid surface
statsSectionsSurface = tools.ResampleFieldsToSurface(
                              statsVolume,
                              sectionsSurface
                          )

# Write mid surface
tools.WriteSurface(
    statsSectionsSurface,
    outSectionsFile
)
