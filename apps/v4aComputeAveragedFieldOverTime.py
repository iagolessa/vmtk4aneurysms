#! /usr/bin/env python3
"""Compute surface-average field over aneurysm surface over time."""

import argparse

def generate_arg_parser():
    """Creates and return a parser object for this app."""

    parser = argparse.ArgumentParser(
                 description="Compute the complete hemodynamics variables "\
                             " given the OpenFOAM directory with the results "\
                             " of a simulation in a vascular geometry."
             )

    # Required
    parser.add_argument(
        '--case',
        help="The path to the OpenFOAM case",
        type=str,
        required=True
    )

    parser.add_argument(
        '--patch',
        help="The patch where to compute the hemodynamics",
        type=str,
        required=True
    )

    parser.add_argument(
        '--field',
        help="Name of the field to integrate",
        type=str,
        required=True
    )

    parser.add_argument(
        '--ofile',
        help="Output file to store surface-averaged values over time (.csv)",
        type=str,
        required=True
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

    parser.set_defaults(multiregion=True)

    # Optional
    parser.add_argument(
        '--patchfile',
        help="Name of the surface file if to integrate on patch of surface",
        type=str,
        required=False,
        default=None
    )

    parser.add_argument(
        '--state',
        help="In case of multiple aneurysms, key to identify on aneurysm",
        type=str,
        choices=["ruptured", "unruptured"],
        required=False,
        default=""
    )

    parser.add_argument(
        '--region',
        help="Name of region if multiregion is True",
        type=str,
        choices=["solid", "fluid"],
        required=False,
        default=""
    )

    return parser


# Get arguments first
parser = generate_arg_parser()
args   = parser.parse_args()

# Begin
import os
import vtk
import sys
import pandas as pd

from pathlib import Path
from vmtk import vtkvmtk
from vmtk import vmtkscripts

import vmtk4aneurysms.lib.constants as const
import vmtk4aneurysms.lib.polydatatools as tools
import vmtk4aneurysms.lib.polydatageometry as geo
import vmtk4aneurysms.lib.foamtovtk as fvtk

import vmtk4aneurysms.hemodynamics as hm
import vmtk4aneurysms.aneurysms as an
import vmtk4aneurysms.wallmotion as wm

# Names
foamFolder       = args.case
surfacePatchName = args.patch
fieldName        = args.field
temporalDataFile = args.ofile
neckSurface      = tools.ReadSurface(args.patchfile) \
                    if args.patchfile is not None else None
aneurysmState    = args.state
multiRegion      = args.multiregion

if multiRegion == True and args.region == "":
    raise NameError("Provide valid region name.")
else:
    regionName = args.region

# Compute the surface-averaged stress over time
print(
    "Computing temporal average for field {}...".format(
        fieldName
    ),
    end="\n"
)

# Compute surface average over time.
# In the case where the vasculature has >= 2 aneurysms (case4, so far)
# specify which state. Code assumes that the name is prepended by the
# aneruysmNeckArray name
arrayName = aneurysmState + an.AneurysmNeckArrayName

foamFile = os.path.join(foamFolder, "case.foam")

# Create dumb case.foam file, if does not exist
# Pathlib will just update mod time if the file already exists
Path(foamFile).touch(exist_ok=True)

fieldNames = [fieldName]

emptySurface, fields = fvtk.GetPatchFieldOverTime(
                           foamFile,
                           field_names=fieldNames,
                           active_patch_name=surfacePatchName,
                           multi_region=multiRegion,
                           region_name=regionName
                       )

scaleMeterToMM = 1.0e3
# Now, compute the surface-average of each field over time
fieldSurfAvg = fvtk.FieldSurfaceAverageOnPatch(
                   tools.ScaleVtkObject(
                       emptySurface,
                       scaleMeterToMM
                   ),
                   fields,
                   patch_surface_id=neckSurface,
                   patch_array_name=an.AneurysmNeckArrayName,
                   patch_boundary_value=0.5
               )

# Write temporal data
with open(temporalDataFile, "a") as tfile:

    tfile.write(
        ",".join(
            ["Time"] + fieldNames
        ) + "\n"
    )

    for time in sorted(fieldSurfAvg[fieldNames[0]].keys()):

        tfile.write(
            ",".join(
                [str(time)] + [str(fieldSurfAvg[fname].get(time))
                               for fname in fieldNames]
            ) + "\n"
        )
