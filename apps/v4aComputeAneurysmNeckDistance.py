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
                 description="Compute the hypothetically healthy parent "\
                             "vascular surface and the aneurysm neck plane. "
             )

    parser.add_argument(
        '--vascularsurface',
        help="The path to the vascular surface file (.vtp)",
        type=str,
        required=True
    )

    parser.add_argument(
        '--aneurysmtype',
        help="The type of the aneurysm (lateral or bifurcation)",
        type=str,
        choices=["bifurcation", "lateral"],
        required=True
    )

    parser.add_argument(
        '--oparentsurface',
        help="Output parent vessel surface (.vtp)",
        type=str,
        required=True
    )

    parser.add_argument(
        '--oaneurysmsurface',
        help="Output aneurysm surface clipped by the computed neck plane (.vtp)",
        type=str,
        required=True
    )

    parser.add_argument(
        '--domepoint',
        help="A tuple indicating the three coordinates of the top dome point",
        type=tuple,
        required=False,
        default=None
    )

    parser.add_argument(
        '--clippedvascularsurface',
        help="Output file for the clipped vascular surface file (.vtp)",
        type=str,
        required=False,
        default=None
    )

    return parser

# Begin: parse arguments
argsParser = generate_arg_parser()
args = argsParser.parse_args()

import os
import sys
import vtk

from vmtk import vmtkscripts
from vmtk4aneurysms.lib import polydatatools as tools
from vmtk4aneurysms import vascular_operations as vscop

# Names
vascularSurfaceFile = args.vascularsurface
aneurysmType        = args.aneurysmtype
parentSurfaceFile   = args.oparentsurface
aneurysmSurfaceFile = args.oaneurysmsurface
domePoint           = args.domepoint
clippedVascularSurfaceFile = args.clippedvascularsurface

vascularSurface = tools.ReadSurface(vascularSurfaceFile)

# This first clip is to reduce the vasculature to a single bifurcation
clippedVascSurface = vscop.ClipVasculature(vascularSurface)

# Select dome point if not already passed
if domePoint is None:
    domePoint = tools.SelectSurfacePoint(clippedVascSurface)

parentSurface = vscop.HealthyVesselReconstruction(
                    clippedVascSurface,
                    aneurysmType,
                    domePoint
                )

# Clip the parent vascular surface
clipper = vmtkscripts.vmtkSurfaceClipper()
clipper.Surface = parentSurface
clipper.InsideOut = False
clipper.Execute()

parentSurface = clipper.Surface

# Compute aneurysm neck plane
aneurysmalSurface = vscop.ComputeGeodesicDistanceToAneurysmNeck(
                       clippedVascSurface,
                       mode="automatic",
                       aneurysm_type=aneurysmType,
                       parent_vascular_surface=parentSurface
                   )

# White files
if clippedVascularSurfaceFile is not None:
    tools.WriteSurface(
        clippedVascSurface,
        clippedVascularSurfaceFile
    )

tools.WriteSurface(
    parentSurface,
    parentSurfaceFile
)

tools.WriteSurface(
    aneurysmalSurface,
    aneurysmSurfaceFile
)
