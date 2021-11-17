#! /usr/bin/env python3

import argparse

def generate_arg_parser():
    """Creates and return a parser object for this app."""

    parser = argparse.ArgumentParser(
                 description="""Compute temporal statistics of selected fields
                             on vascular surface.

                             The script builds a surface with the temporal
                             statistics of the fields on the surface related to
                             the wall dynamics.  The final surface
                             'statsSurface' here have all these time statistics
                             fields, which allow us to compute the aneurysm
                             statistics, similar to what we do for the
                             hemodynamics fields. Note that this surface can
                             also be used to compute the pulsatility index. 
                             This script also computes the surface-average
                             of the fields over time on the whole surface or, 
                             optionally, on the aneurysm if a patch surface
                             is passed, i.e., the lumen surface with the 
                             aneurysm neck array."""
             )

    parser.add_argument(
        '--case',
        help="The path to the OpenFOAM case",
        type=str,
        required=True
    )

    parser.add_argument(
        '--patch',
        help="""The patch where to compute the wall dynamics (either the patch
             name, or 'volumeMesh' to get the internal mesh)""",
        type=str,
        required=True
    )

    parser.add_argument(
        '--ofile',
        help="Output wall dynamics file as a vtkPolyData object (.vtp)",
        type=str,
        required=True
    )

    parser.add_argument(
        '--otemporalfile',
        help="Output the surface averaged fields as a CSV file (.csv)",
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

    parser.add_argument(
        '--region',
        help="Name of region if multiregion is True",
        type=str,
        choices=["solid", "fluid"],
        required=False,
        default=""
    )

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

    return parser

# Parse arguments
argsParser = generate_arg_parser()
args = argsParser.parse_args()

import os
import vtk
import sys
import pandas as pd

from pathlib import Path
from vtk.numpy_interface import dataset_adapter as dsa
from vmtk import vtkvmtk
from vmtk import vmtkscripts

import vmtk4aneurysms.lib.constants as const
import vmtk4aneurysms.lib.polydatatools as tools
import vmtk4aneurysms.lib.polydatageometry as geo
import vmtk4aneurysms.lib.foamtovtk as fvtk

import vmtk4aneurysms.hemodynamics as hm
import vmtk4aneurysms.aneurysms as aneu
import vmtk4aneurysms.wallmotion as wm

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
surfacePatchName = "" if args.patch == "volumeMesh" else args.patch
wallDynamicsFile = args.ofile
temporalDataFile = args.otemporalfile
multiRegion      = args.multiregion

neckSurfaceFile  = args.patchfile
aneurysmState    = args.state

if multiRegion == True and args.region == "":
    raise NameError("Provide valid region name.")
else:
    regionName = args.region

if args.patch == "volumeMesh" and wallDynamicsFile.endswith(".vtp"):
    raise ValueError("If volume mesh is extract, passa a file name with extension .vtk")

fieldNames = ["D",
              "sigmaEq",
              "sigmaMax",
              # "sigmaMid",
              # "sigmaMin",
              "stretchMax"]
              # "stretchMid",
              # "stretchMin"
              # ]

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

emptySurface, fields = fvtk.GetPatchFieldOverTime(
                           foamFile,
                           field_names=fieldNames,
                           active_patch_name=surfacePatchName,
                           multi_region=multiRegion,
                           region_name=regionName
                       )

# Computes the statistics of each field
statsSurface = fvtk.FieldTimeStats(
                   emptySurface, # FieldTimeStats operates on a copy, so fine
                   fields,
                   peakSystoleInstant,
                   lowDiastoleInstant
               )

# Scale the surface back to millimeters (fields are not scaled)
scaleMeterToMM = 1.0e3
emptySurface = tools.ScaleVtkObject(emptySurface, scaleMeterToMM)
statsSurface = tools.ScaleVtkObject(statsSurface, scaleMeterToMM)

# Scale displacement arrays too
displacementArrays = [arrayName
                      for arrayName in tools.GetCellArrays(statsSurface)
                      if arrayName.startswith("D_")]

npSurface = dsa.WrapDataObject(statsSurface)

for arrayName in displacementArrays:
    array = scaleMeterToMM*npSurface.CellData.GetArray(arrayName)

    npSurface.CellData.append(array, arrayName)

if args.patch == "volumeMesh":
    tools.WriteUnsGrid(
        npSurface.VTKObject,
        wallDynamicsFile
    )

else:
    tools.WriteSurface(
        npSurface.VTKObject,
        wallDynamicsFile
    )

# Now, compute the surface-average of each field over time
fieldSurfAvg = fvtk.FieldSurfaceAverageOnPatch(
                   emptySurface,
                   fields,
                   patch_surface_id=tools.ReadSurface(neckSurfaceFile) \
                                    if neckSurfaceFile is not None \
                                    else None,
                   patch_array_name=aneu.AneurysmNeckArrayName,
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
