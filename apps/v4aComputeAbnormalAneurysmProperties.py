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


import os
import sys
import vtk
import argparse

from vmtk import vtkvmtk
from vmtk import vmtkscripts
from vmtk4aneurysms.pypescripts import v4aScripts

import vmtk4aneurysms.lib.polydatatools as tools
import vmtk4aneurysms.vascular_operations as vscop

def generate_arg_parser():
    """Creates and return a parser object for this app."""

    parser = argparse.ArgumentParser(
        description="""Compute the abnormal thickness and abnormal aneurysm
            properties of a vascular geometry.

            Given the surface generated by
            'v4aComputeUniformAneurysmProperties.py' with the fields of
            thickness and elasticities uniform on the aneurysm sac, 'upates'
            the fields there to account for the abnormal hemodynamics patterns
            obtained from the TAWSS and OSI fields, provided separately as also
            a surface of the vasculature. The result are the fields updated and
            also a field called 'WallType' that identifies the patches with
            type I and II morphology according to the hemodynamics."""
        )

    parser.add_argument(
        '--surfacefile',
        help="""File path to the surface file with uniform fields (surface
            dimensions units must me in millimeters)""",
        type=str,
        required=True
    )

    parser.add_argument(
        '--hemodynamicsfile',
        help="""File path to the surface with hemodynamics of the vasculature
            (surface dimensions units must me in millimeters)""",
        type=str,
        required=True
    )

    # Optional
    parser.add_argument(
        '--case',
        help="The path to the OpenFOAM case where to store the files (default: cwd)",
        type=str,
        required=False,
        default=os.getcwd()
    )

    parser.add_argument(
        '--ofilename',
        help="""Output file name with thickness and material constants fields
            (stored in the case directory, default: surfaceWithAbnormalAneurysmArrays.vtp)""",
        type=str,
        required=False,
        default="surfaceWithAbnormalAneurysmArrays.vtp"
    )

    parser.add_argument(
        '--typeIthicknessscale',
        help="""Scale factor to control thickness in red regions
            (type I, default: 0.95)""",
        type=float,
        default=0.95,
        required=False
    )

    parser.add_argument(
        '--typeIIthicknessscale',
        help="""Scale factor to control thickness in atherosclerotic regions
            (type I, default: 1.20)""",
        type=float,
        default=1.20,
        required=False
    )

    parser.add_argument(
        '--elasticitiesabnormalfactor',
        help="""Scale factor to control abnormal elasticities patches
            (default: 1.20)""",
        type=float,
        default=1.20,
        required=False
    )

    parser.add_argument(
        '--inspectwalltype',
        help="""Flag indicating to visualize the WallType array prior
             to fields update""",
        dest="inspectwalltype",
        action="store_true",
    )

    parser.add_argument(
        '--furthersmoothing',
        help="""Further smooth the fields defined on the input surface with the
            uniform wall morphology. This is recommended to have the same level
            of smoothing of the resulting surface with the abnormal wall
            morphology fields""",
        dest="furthersmoothing",
        action="store_true",
    )

    parser.set_defaults(
        inspectwalltype=False,
        furthersmoothing=False
    )

    return parser

# Get arguments
parser = generate_arg_parser()
args = parser.parse_args()

uniformSurface   = tools.ReadSurface(args.surfacefile)
hemoSurface      = tools.ReadSurface(args.hemodynamicsfile)
casePath         = args.case
abnormalOutFile  = os.path.join(
                      args.case,
                      args.ofilename
                   )


typeIThicknessFactor  = args.typeIthicknessscale  # red regions
typeIIThicknessFactor = args.typeIIthicknessscale # atherosclerotic regions

# For the elasticities, both type I and type II regions are stiffer
elasticitiesAbnormalFactor = args.elasticitiesabnormalfactor

inspectWallType  = args.inspectwalltype
furtherSmoothing = args.furthersmoothing

def wallTypeHemoClassification(
        hemo_surface: vtk.vtkCommonDataModelPython.vtkPolyData,
        surface_with_thickness: vtk.vtkCommonDataModelPython.vtkPolyData
    ) -> vtk.vtkCommonDataModelPython.vtkPolyData:
    """Classify aneurysm wall based on hemodynamics.

    With the surface with the hemodynamics results of a flow simulation of a
    vasculature with an aneurysm, classifies its wall based on specific
    combinations of hemodynamics variables, creating a discrete array called
    WallType. Then, project this array to the surface with the original
    Thickness array.
    """

    # Compute WallType array
    wallTypeSurface = vscop.WallTypeClassification(hemo_surface)

    # Debug
    if inspectWallType:
        tools.ViewSurface(
            wallTypeSurface,
            array_name="WallType"
        )

    # Project cell array WallType as a cell array on surface with thickness
    return tools.ProjectCellArray(
               surface_with_thickness,
               wallTypeSurface,
               "WallType"
           )

def updateElasticity(
        surface: vtk.vtkCommonDataModelPython.vtkPolyData,
        elasticty: str
    )   -> vtk.vtkCommonDataModelPython.vtkPolyData:

    updateElasticity = v4aScripts.vmtkSurfaceAneurysmElasticity()
    updateElasticity.Surface = surface
    updateElasticity.ElasticityArrayName = elasticity

    updateElasticity.OnlyUpdateElasticity        = True
    updateElasticity.AbnormalHemodynamicsRegions = True
    updateElasticity.UniformElasticity           = False

    updateElasticity.AtheroscleroticFactor = elasticitiesAbnormalFactor
    updateElasticity.RedRegionsFactor      = elasticitiesAbnormalFactor
    updateElasticity.Execute()

    return updateElasticity.Surface


if "WallType" not in tools.GetCellArrays(uniformSurface):
    print("Projecting wall type array", end="\n")

    # Project the WallType array to the local surface
    # This script will also sabe the WallType to the original surface
    vascularSurface = wallTypeHemoClassification(
                          hemoSurface,
                          uniformSurface
                      )

else:
    if inspectWallType:
        tools.ViewSurface(
            uniformSurface,
            array_name="WallType"
        )

    vascularSurface = uniformSurface

# Update thickness array
print("Updating the thickness", end="\n")
updateThickness = v4aScripts.vmtkSurfaceVasculatureThickness()
updateThickness.Surface = vascularSurface

updateThickness.OnlyUpdateThickness         = True
updateThickness.AbnormalHemodynamicsRegions = True
updateThickness.UniformWallToLumenRatio     = False

updateThickness.RedRegionsFactor      = typeIThicknessFactor
updateThickness.AtheroscleroticFactor = typeIIThicknessFactor
updateThickness.Execute()

vascularSurface = updateThickness.Surface

# Update elasticities arrays
elasticities = ["E", "c1Fung",
                "c1Yeoh", "c2Yeoh",
                "c10", "c01", "c11"]

for elasticity in elasticities:
    print("Updating {}".format(elasticity), end="\n")
    vascularSurface = updateElasticity(
                          vascularSurface,
                          elasticity
                      )

tools.WriteSurface(
    vascularSurface,
    abnormalOutFile
)

if furtherSmoothing:
    # The abnormal arrays are smoothier than the original uniform arrays.
    # Hence, we apply a further smoothing in the uniform-aneurysm Thickness
    # array so the branches thickness is the same as the resulting
    # abnormal-aneurysm Thickness array
    print(
        "Updating thickness with uniform aneurysm thickness...",
        end='\n'
    )

    triangulate = vmtkscripts.vmtkSurfaceTriangle()
    triangulate.Surface = uniformSurface
    triangulate.Execute()

    arraySmoothing = vmtkscripts.vmtkSurfaceArraySmoothing()
    arraySmoothing.Surface = triangulate.Surface
    arraySmoothing.Iterations = 10
    arraySmoothing.SurfaceArrayName = "Thickness"
    arraySmoothing.Execute()

    tools.WriteSurface(
        arraySmoothing.Surface,
        args.surfacefile
    )
