#! /usr/bin/env python3

import os
import sys
import vtk
import argparse

from vmtk import vtkvmtk
from vmtk import vmtkscripts
from vmtk4aneurysms.vmtkextend import customscripts

import vmtk4aneurysms.lib.polydatatools as tools
import vmtk4aneurysms.wallmotion as wm

def generate_arg_parser():
    """Creates and return a parser object for this app."""

    parser = argparse.ArgumentParser(
        description="""Compute the abnormal thickness and abnormal aneurysm
            properties of a vascular geometry."""
    )

    parser.add_argument(
        '--surfacefile',
        help="File path to the surface file with uniform fields",
        type=str,
        required=True
    )

    parser.add_argument(
        '--hemodynamicsfile',
        help="File path to the surface with hemodynamics of the vasculature",
        type=str,
        required=True
    )

    parser.add_argument(
        '--case',
        help="The path to the OpenFOAM case where to store the files",
        type=str,
        required=False,
        default=os.getcwd()
    )

    parser.add_argument(
        '--ofile',
        help="Output file with thickness and material constants fields",
        type=str,
        required=False,
        default=os.path.join(
                    os.getcwd(), 
                    "surfaceWithAbnormalAneurysmArrays.vtp"
                )
    )

    parser.add_argument(
        '--typeIthicknessscale',
        help="Scale factor to control thickness in red regions (type I)",
        type=float,
        default=0.95,
        required=False
    )

    parser.add_argument(
        '--typeIIthicknessscale',
        help="Scale factor to control thickness in atherosclerotic regions (type I)",
        type=float,
        default=1.20,
        required=False
    )

    parser.add_argument(
        '--elasticitiesabnormalfactor',
        help="Scale factor to control abnormal elasticities patches",
        type=float,
        default=1.20,
        required=False
    )

    return parser

# Get arguments
parser = generate_arg_parser()
args = parser.parse_args()

uniformSurface   = tools.ReadSurface(args.surfacefile)
hemoSurface      = tools.ReadSurface(args.hemodynamicsfile)
abnormalOutFile  = args.ofile
casePath         = args.case

typeIThicknessFactor  = args.typeIthicknessscale  # red regions
typeIIThicknessFactor = args.typeIIthicknessscale # atherosclerotic regions

# For the elasticities, both type I and type II regions are stiffer 
elasticitiesAbnormalFactor = args.elasticitiesabnormalfactor


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
    wallTypeSurface = wm.WallTypeClassification(hemo_surface)

    # The surface with WallType array came from the flow simulation so its
    # dimensions are in meters -> scale it to match surface in millimeters
    # (maybe I should default this analysis to always have the surfaces in mm)
    scaling = vmtkscripts.vmtkSurfaceScaling()
    scaling.Surface = wallTypeSurface
    scaling.ScaleFactor = 1000.0
    scaling.Execute()

    # Debug
    tools.ViewSurface(
        scaling.Surface,
        array_name="WallType"
    )

    # Project cell array WallType as a cell array on surface with thickness
    projector = vtkvmtk.vtkvmtkSurfaceProjectCellArray()
    projector.SetInputData(surface_with_thickness)
    projector.SetReferenceSurface(scaling.Surface)
    projector.SetProjectedArrayName("WallType")

    # The default can be zero because it indicates a normal wall type (this is
    # more likely to occur on on the edges between normal and abnormal wall
    # types)
    projector.SetDefaultValue(0.0)
    projector.Update()

    return projector.GetOutput()

def updateElasticity(
        surface: vtk.vtkCommonDataModelPython.vtkPolyData, 
        elasticty: str
    )   -> vtk.vtkCommonDataModelPython.vtkPolyData:

    updateElasticity = customscripts.vmtkSurfaceAneurysmElasticity()
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
    vascularSurface = uniformSurface

# Update thickness array
print("Updating the thickness", end="\n")
updateThickness = customscripts.vmtkSurfaceVasculatureThickness()
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

# The abnormal arrays are smoothier than the original uniform arrays.
# Hence, we apply a further smoothing in the uniform-aneurysm Thickness array
# so the branches thickness is the same as the resulting abnormal-aneurysm
# Thickness array
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
