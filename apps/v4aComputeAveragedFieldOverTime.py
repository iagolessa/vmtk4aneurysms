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
surfacePatch     = args.patch
fieldName        = args.field
temporalDataFile = args.ofile
neckSurfaceFile  = args.patchfile
aneurysmState    = args.state
multiRegion      = args.multiregion

if multiRegion == True and args.region == "":
    raise NameError("Provide valid region name.")
else:
    regionName = args.region

# Scale the surface back to millimeters (fields are not scaled)
neckSurface = None

if neckSurfaceFile is not None:
    scaling = vmtkscripts.vmtkSurfaceScaling()
    scaling.Surface = tools.ReadSurface(neckSurfaceFile)
    scaling.ScaleFactor = 1.0e-3
    scaling.Execute()

    neckSurface = scaling.Surface

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

fieldAvgOverTime = fvtk.FieldSurfaceAverageOnPatch(
                       foamFile,
                       fieldName,
                       surfacePatch,
                       patch_surface_id=neckSurface,
                       patch_array_name=arrayName,
                       patch_boundary_value=0.5,
                       multi_region=multiRegion,
                       region_name=regionName
                   )

# Write temporal data
with open(temporalDataFile, "a") as file_:
    file_.write(
        "Time,{}\n".format(fieldName)
    )

    for time in sorted(fieldAvgOverTime.keys()):
        file_.write(
            "{},{}\n".format(
                time, 
                fieldAvgOverTime.get(time)
            )
        )
