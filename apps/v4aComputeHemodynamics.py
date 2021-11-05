#! /usr/bin/env python3

import argparse

def generate_arg_parser():
    """Creates and return a parser object for this app."""

    parser = argparse.ArgumentParser(
                 description="Compute the complete hemodynamics variables "\
                             " given the OpenFOAM directory with the results "\
                             " of a simulation in a vascular geometry."
             )

    parser.add_argument(
        '--case',
        help="The path to the OpenFOAM case",
        type=str,
        required=True
    )

    parser.add_argument(
        '--patch',
        help="The patch where to compute the hemodynamics (default: 'wall')",
        type=str,
        default="wall"
    )

    parser.add_argument(
        '--ofile',
        help="Output hemodynamics file as a vtkPolyData object (.vtp)",
        type=str,
        required=True
    )

    parser.add_argument(
        '--otemporalfile',
        help="Output surface-averaged WSS over time(.csv)",
        type=str,
        required=True
    )

    # Optional
    parser.add_argument(
        '--density',
        help="density of blood (default 1056.0 kg/m3)",
        type=float,
        default=1056.0
    )

    parser.add_argument(
        '--computegon',
        help="Flag to activate GON computation (computationally expensive)",
        dest="computegon",
        action="store_true",
    )

    parser.add_argument(
        '--computeafi',
        help="Flag to activate AFI computation (computationally expensive)",
        dest="computeafi",
        action="store_true",
    )

    parser.set_defaults(
        computegon=False,
        computeafi=False
    )

    return parser

# Begin: parse arguments
argsParser = generate_arg_parser()
args = argsParser.parse_args()

import os
import sys
import pandas as pd

from vmtk import vmtkscripts
from vmtk4aneurysms.lib import polydatatools as tools
from vmtk4aneurysms.lib import polydatageometry as geo
from vmtk4aneurysms.lib import foamtovtk as fvtk

from vmtk4aneurysms import hemodynamics as hm


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
hemodynamicsFile = args.ofile
temporalDataFile = args.otemporalfile
bloodDensity     = args.density

boolComputeGon   = args.computegon
boolComputeAfi   = args.computeafi

# Get peak systole and low diastole instant per case
foamFile = os.path.join(foamFolder, "case.foam")
peakSystoleInstant, lowDiastoleInstant = get_peak_instant(foamFolder)

print(
    "Peak and diastole instants {}s {}s".format(
        peakSystoleInstant,
        lowDiastoleInstant
    ),
    end="\n"
)

# TODO: include multiregion support
hemodynamicsSurface = hm.Hemodynamics(
                          foamFile,
                          peakSystoleInstant,
                          lowDiastoleInstant,
                          patch=surfacePatch,
                          compute_gon=boolComputeGon,
                          compute_afi=boolComputeAfi
                      )

# Scale the surface back to millimeters (fields are not scaled)
# before computing the curvatures
hemodynamicsSurface = tools.ScaleVtkObject(
                          hemodynamicsSurface,
                          1.0e3
                      )

tools.WriteSurface(
    hemodynamicsSurface,
    hemodynamicsFile
)

# Compute temporal evolution of WSS over the *whole surface*
fieldAvgOverTime = hm.WssSurfaceAverage(foamFile)

with open(temporalDataFile, "a") as file_:
    file_.write("Time, surfaceAveragedWSS\n")

    for time in sorted(fieldAvgOverTime.keys()):
        value = fieldAvgOverTime.get(time)

        file_.write(
            "{},{}\n".format(time, value)
        )
