#! /usr/bin/env python3

import os
import argparse

import vtk
from vmtk import vtkvmtk
from vmtk import vmtkscripts
from vmtk4aneurysms.pypescripts import v4aScripts

import vmtk4aneurysms.lib.polydatatools as tools
import vmtk4aneurysms.wallmotion as wm


def generate_arg_parser():
    """Creates and return a parser object for this app."""

    parser = argparse.ArgumentParser(
        description="""Compute the uniform thickness and uniform aneurysm
            properties of a vascular geometry.

            The script calls the VMTK extend scripts to calculate the thickness
            and elasticity arrays on a vasculature surface with an aneurysm.
            The aneurysm will have a uniform thickness and elasticity in this
            case.

            The elasticity it computes is the classic one given by the Young's
            modulus, however, at the end of the procedure, it also computes the
            heterogeneous arrays for each material constant of the following
            constitutive models: Fung's (c1), Yeoh (c1 and c2), and
            Mooney-Rivlin (c10, c01, and c11), and stores them on the same
            surface.

            Hence, the final surface have the following arrays:

                - DistanceToNeck: measures the Euclidean distance to the neck
                  line
                - LocalWLRArray: the pointwise WLR used to build the Thickness
                - Thickness: the wall thickness array
                - E: the Young's modulus array
                - c1, (c1, c2), and (c10, c01, c11): the arrays for each
                  material constants of jthe aforementioned models."""
    )

    parser.add_argument(
        '--surfacefile',
        help="File path to the vascular geometry surface file",
        type=str,
        required=True
    )

    parser.add_argument(
        '--centerlinefile',
        help="File path to the centerline file",
        type=str,
        required=False,
        default=""
    )

    parser.add_argument(
        '--naneurysms',
        help="Number of aneurysms on the surface",
        type=int,
        required=True
    )

    parser.add_argument(
        '--status',
        help="Aneurysm's rupture status",
        type=str,
        choices=["ruptured", "unruptured"],
        required=True
    )

    # Optional
    parser.add_argument(
        '--aneurysmscale',
        help="Scale factor to control aneurysm global thickness (default: 0.75)",
        type=float,
        default=0.75,
        required=False
    )

    parser.add_argument(
        '--influencedistance',
        help="""Distance, in millimeters, from neck line influenced by the
            aneurysm (default: 0.5)""",
        type=float,
        default=0.5,
        required=False
    )

    parser.add_argument(
        '--updatethickness',
        help="Indicate to only update thickness",
        dest="updatethickness",
        action="store_true",
    )
    parser.set_defaults(updatethickness=False)

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
            (stored in the case directory, default: surfaceWithUniformAneurysmArrays.vtp)""",
        type=str,
        required=False,
        default="surfaceWithUniformAneurysmArrays.vtp"
    )

    return parser

# Mechanical properties (elasticities or stiffnesses) of arteries and aneurysms
# according to status, selected to be used, in Pascal
arteriesConst = {"E"     : 1e6,
                 "c1Fung": 0.3536e6,
                 "c1Yeoh": 0.1067e6,
                 "c2Yeoh": 5.1602e6,
                 "c10"   : 0.1966e6,
                 "c01"   : 0.0163e6,
                 "c11"   : 7.8370e6}

aneurysmConst = {"ruptured": {
                    "E"     : 0.5e6,
                    "c1Fung": 0.1768e6, # half the arteries
                    "c1Yeoh": 0.0700e6,
                    "c2Yeoh": 2.1000e6,
                    "c10"   : 0.1900e6,
                    "c01"   : 0.0260e6,
                    "c11"   : 1.3770e6},
                 "unruptured": {
                    "E"     : 2e6,
                    "c1Fung": 0.7072e6, # double the arteries
                    "c1Yeoh": 0.1200e6,
                    "c2Yeoh": 6.8000e6,
                    "c10"   : 0.1900e6,
                    "c01"   : 0.0230e6,
                    "c11"   : 11.780e6}}

def computeElasticity(
        surface: vtk.vtkCommonDataModelPython.vtkPolyData,
        constant_name: str,
        stat: str,
        n_aneurysms: int
    )   -> vtk.vtkCommonDataModelPython.vtkPolyData:

    # the default, only one aneurysm
    aneurysmProperties = (aneurysmConst[stat][constant_name],)

    if n_aneurysms == 2:
        print(
            "Informed two aneurysms, updating values of elasticities",
            end="\n"
        )

        # Update aneurysm properties
        # Aneurysm #1 == ruptured
        # Aneurysm #2 == unruptured
        # For case 4, particularly
        aneurysmProperties = (aneurysmConst["ruptured"][constant_name],
                              aneurysmConst["unruptured"][constant_name])

    calcElasticity = v4aScripts.vmtkSurfaceAneurysmElasticity()
    calcElasticity.Surface = surface
    calcElasticity.NumberOfAneurysms = n_aneurysms
    calcElasticity.ElasticityArrayName = constant_name
    calcElasticity.ArteriesElasticity = arteriesConst[constant_name]
    calcElasticity.AneurysmElasticity = aneurysmProperties
    calcElasticity.UniformElasticity = 1
    calcElasticity.Execute()

    return calcElasticity.Surface


# Parse arguments
argsParser = generate_arg_parser()
args = argsParser.parse_args()

# Get args
vascularSurface = tools.ReadSurface(args.surfacefile)
centerlineFile  = args.centerlinefile
nAneurysms      = args.naneurysms
aneurysmStatus  = args.status
aneurysmScale   = args.aneurysmscale
influenceDist   = args.influencedistance
updateThickness = args.updatethickness
casePath        = args.case

uniformAneurysmSurfaceFile = os.path.join(
                                 args.case,
                                 args.ofilename
                             )

thicknessArrayName = "Thickness"

# Default parameters

# Build centerlines to include aneurysms
# this avoid unrealistic thickness around the aneurysm
# Compute thickness with non-uniform WLR (should it be the default? yes)
if not updateThickness:
    # Create new thickness and properties

    if os.path.isfile(centerlineFile):
        centerlines = tools.ReadSurface(centerlineFile)

    else:
        centerlines = vmtkscripts.vmtkCenterlines()
        centerlines.Surface = vascularSurface
        centerlines.AppendEndPoints = True
        centerlines.Execute()

        centerlines = centerlines.Centerlines

    # Update thickness array
    print("Computing thickness", end="\n")

    computeThickness = v4aScripts.vmtkSurfaceVasculatureThickness()
    computeThickness.Surface     = vascularSurface
    computeThickness.Centerlines = centerlines
    computeThickness.Aneurysm    = True

    computeThickness.OnlyUpdateThickness     = False
    computeThickness.UniformWallToLumenRatio = False
    computeThickness.SelectAneurysmRegions   = False
    computeThickness.GenerateWallMesh        = False
    computeThickness.AbnormalHemodynamicsRegions = False

    computeThickness.NumberOfAneurysms   = nAneurysms
    computeThickness.GlobalScaleFactor   = aneurysmScale
    computeThickness.ThicknessArrayName  = thicknessArrayName
    computeThickness.SmoothingIterations = 15
    computeThickness.AneurysmInfluenceRegionDistance = influenceDist
    computeThickness.Execute()

    vascularSurface = computeThickness.Surface

    # After finishing the surface, create the nonlinear models' material
    # constants
    print(
        "Computing nonlinear models material constants",
        end="\n"
    )

    for property_ in arteriesConst.keys():
        print("Adding {}".format(property_), end="\n")

        vascularSurface = computeElasticity(
                              vascularSurface,
                              property_,
                              aneurysmStatus,
                              nAneurysms
                          )

else:
    # Only update thickness field
    if thicknessArrayName in tools.GetPointArrays(vascularSurface):
        updateThickness = v4aScripts.vmtkSurfaceVasculatureThickness()
        updateThickness.Surface = vascularSurface
        updateThickness.Aneurysm = True
        updateThickness.NumberOfAneurysms = nAneurysms
        updateThickness.GenerateWallMesh = False
        updateThickness.SmoothingIterations = 10
        updateThickness.UniformWallToLumenRatio = False
        updateThickness.OnlyUpdateThickness = True
        updateThickness.SelectAneurysmRegions = True
        updateThickness.LocalScaleFactor = 0.9
        updateThickness.AbnormalHemodynamicsRegions = False
        updateThickness.Execute()

        vascularSurface = updateThickness.Surface

    else:
        raise ValueError("{} array not in surface.".format(thicknessArrayName))

tools.WriteSurface(
    vascularSurface,
    uniformAneurysmSurfaceFile
)
