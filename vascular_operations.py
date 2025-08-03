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

"""Collection of tools to manipulate vascular models and aneurysms."""

import sys
import vtk
import numpy as np
import morphman.common as mplib
from morphman.manipulate_curvature import extract_single_line

from vmtk import vtkvmtk
from vmtk import vmtkscripts
from scipy import interpolate

from vtkmodules.numpy_interface import dataset_adapter as dsa

from .lib import names
from .lib import centerlines as cl
from .lib import constants as const
from .lib import polydatatools as tools
from .lib import polydatageometry as geo

from vmtk4aneurysms.vascular_classes import VascularCenterline
from vmtk4aneurysms.neck_extractor import ComputeGeodesicDistanceToAneurysmNeck

def ClipVasculatureWithPlane(
        vascular_surface: names.polyDataType,
        plane_center: tuple,
        plane_normal: tuple
    )   -> names.polyDataType:
    """Clip vascular tree section with a plane.

    Given a plane center and normal, clip the passed vascular surface model at
    the plane. The portion of the surface kept is on the normal direction.
    """

    # Clip vessel at inlet location
    vascular_surface = tools.SeamPlaneTubularStructureMarker(
                           vascular_surface,
                           plane_center=plane_center,
                           plane_normal=plane_normal,
                           seam_scalar_array_name=names.SeamScalarsArrayName
                       )

    # The SeamScalars are positive (1.0) in the region of the positiove
    # direction of the plane normal so use inside_out False.
    return tools.ClipWithScalar(
               vascular_surface,
               names.SeamScalarsArrayName,
               const.zero,
               inside_out=False
           )

def ClipVasculature(
        vascular_surface: names.polyDataType,
        centerlines=None
    )   -> names.polyDataType:
    """Clip a vascular surface segment, by selecting end points.

    Given a vascular surface, the user is prompted to select points
    on the surface that 1) identifies the surface's bulk and 2) where the
    vasculature should be clipped. Uses, internally, the
    'vmtksurfaceendclipper' script.
    """

    if centerlines is None:
        objCenterlines = VascularCenterline.from_vascular_surface(vascular_surface)

    else:
        objCenterlines = VascularCenterline(centerlines)

    surfaceEndClipper = vmtkscripts.vmtkSurfaceEndClipper()
    surfaceEndClipper.Surface = vascular_surface
    surfaceEndClipper.CenterlineNormals = 1
    surfaceEndClipper.Centerlines = objCenterlines.GetCenterline()
    surfaceEndClipper.FrenetTangentArrayName = names.vmtkFrenetTangentArrayName
    surfaceEndClipper.Execute()

    return surfaceEndClipper.Surface

# PUT THIS ONE INTP THE Vascular Surface or ICA Surface Class Class
def SplitICAModelIntoBends(
        vascular_model: names.polyDataType,
        centerlines: names.polyDataType,
        bif_point: tuple
    )   -> names.polyDataType:
    """Split ICA vascular surface model into bends based on curvature and
    torsion."""

    # Split centerlines into bends
    bendsCenterlines = SplitICACenterlineIntoBends(
                            centerlines,
                            bif_point
                        )

    return tools.ProjectPointArray(
                vascular_model,
                bendsCenterlines,
                names.BendsIdsFieldName
            )

def ClipVasculatureOffBifurcation(
        vascular_surface: names.polyDataType,
        centerlines: names.polyDataType,
        clip_vessel_field: str=names.vmtkAbscissasArrayName,
        inlet_vessel_clip_value: float=-40.0,
        outlet_vessel_clip_value: str=8.0,
        bif_point: tuple=None,
        aneurysm_point: tuple=None
    )   -> names.polyDataType:
    """Automatically clip a vessel structure away of a specified bifurcation
    and an aneurysm, if it exists.

    Given a vessel surface model and its centerlines, clip the vessel at points
    espefied by a distance away from a selected bifurcation and away from the
    bifurcation closer to a specified aneurysm. The function is intended to
    clip inlet and outlet boundary conditions in vascular models based a
    predefined distance from a bifurcation and an aneurysm. For example, you
    can select the ICA bifurcation and pass two values identified as the
    distance fom the ICA bifurcation where the vascular model surface will be
    clipped, if no aneurysm on the model. If an aneurysm is present and you
    want to clip after the aneurysm, you can pass the ICA bifurcation point and
    the aneurysm dome point (if not passed, the function prompts the user to
    interactively select it), so the function clips the vasculature after the
    aneurysm bifurcation.

    Both the clip values for inlet (before the bifurcation) as the clip value
    of outlets (after the bifurcation can be passed. The default field used to
    clip is the 'Abscissas' field of distance values along the centerline with
    origin its closest point to the selected bifurcation. If an aneurysm
    exists, then the clip arg. 'outlet_vessel_clip_value' is relative to the
    bifurcation closest to the aneurysm.

    The point closest to the selected bifurcation can be passed via the arg.
    'bif_point'. If None, the user is prompted to interactively select it.
    """

    if bif_point is None:

        bif_point = tools.SelectSurfacePoint(
                        vascular_surface,
                        input_text="Select point at the ICA bifurcation\n"
                    )

    # Compute centerlines abscissas and other attributes
    offsetCenterlines = cl.ComputeCenterlinePropertiesOffBifurcation(
                            centerlines,
                            bif_point
                        )

    # Computing the bifurcation reference system
    referenceSystems = cl.CenterlineReferenceSystems(offsetCenterlines)

    # Get ICA-MCA-ACA bifurcation
    # Get closest point to ICA at the bifurcation found
    bifGroupId  = int(
                      tools.GetFieldValueAtClosestPoint(
                          referenceSystems,
                          bif_point,
                          names.vmtkGroupIdsArrayName
                      )
                  )

    # Identify the bifurcation GroupId and its Abscissas closer to the aneurysm
    if aneurysm_point is None:

        aneurysm_point = tools.SelectSurfacePoint(
                             vascular_surface,
                             input_text="Select a  point on the aneurysm surface"
                         )

    # Get center of bifurcation closest to aneurysm point
    iaClosestPointToBif = tools.LocateClosestPointOnPolyData(
                                referenceSystems,
                                aneurysm_point
                            )

    onlyBifurcations = tools.ExtractPortion(
                           offsetCenterlines,
                           # only centerlines potions of bifs.
                           names.vmtkBlankingArrayName,
                           const.one
                       )

    # Get also the group id of the portion where the aneurysm is
    iaAbscissasClosestBif = tools.GetFieldValueAtClosestPoint(
                                  onlyBifurcations,
                                  iaClosestPointToBif,
                                  names.vmtkAbscissasArrayName
                            )

    # Get also the group id of the portion where the aneurysm is
    iaGroupIdClosestBif = tools.GetFieldValueAtClosestPoint(
                                onlyBifurcations,
                                aneurysm_point,
                                names.vmtkGroupIdsArrayName
                          )

    # Check whether passed clip values are within the clip field range
    offsetAbscissasRange = offsetCenterlines.GetPointData().GetArray(
                                names.vmtkAbscissasArrayName
                            ).GetRange()

    if inlet_vessel_clip_value < min(offsetAbscissasRange):

        raise ValueError(
                "{} smaller than min of {} (~ {}).\nSpecify higher value.".format(
                    inlet_vessel_clip_value,
                    clip_vessel_field,
                    round(min(offsetAbscissasRange), 3)
                  )
              )

    if outlet_vessel_clip_value + iaAbscissasClosestBif > max(offsetAbscissasRange):

        raise ValueError(
                "{} distance relative to aneurysm bifurcation is larger than max of {} (~ {}).\nSpecify smaller value.".format(
                    outlet_vessel_clip_value,
                    clip_vessel_field,
                    round(max(offsetAbscissasRange), 3)
                  )
              )

    # Cretae dict to better storing of separate centerlines
    individualCenterlines = cl.SplitCenterlineObject(offsetCenterlines)

    # Get longest centerline
    # ID of ICA clip will be identified in this portion
    idLongestCenterline = max(
                                individualCenterlines,
                                key=lambda idx: individualCenterlines[idx]["length"]
                            )

    longestCenterline = individualCenterlines[idLongestCenterline]["object"]
    npLongestCenterline = dsa.WrapDataObject(longestCenterline)

    # Clip inlet artery
    clipArray = npLongestCenterline.PointData.GetArray(clip_vessel_field)

    # Get id of the point where to clip
    icaClipPointId = (
                        np.abs(clipArray - inlet_vessel_clip_value)
                     ).argmin()

    icaClipPoint = tuple(npLongestCenterline.Points[icaClipPointId])

    icaClipNormal = tuple(
                        npLongestCenterline.PointData.GetArray(
                            names.vmtkFrenetTangentArrayName
                        )[icaClipPointId]
                    )

    # Clip vessel at inlet location
    vascular_surface = ClipVasculatureWithPlane(
                           vascular_surface,
                           plane_center=icaClipPoint,
                           plane_normal=icaClipNormal
                       )

    # Store the group id to whcih the point belong
    outletClipPoints = {}

    for cl_id, dict_ in individualCenterlines.items():

        npClPortion = dsa.WrapDataObject(dict_["object"])
        clClipField = npClPortion.PointData.GetArray(clip_vessel_field)

        clGroups = npClPortion.PointData.GetArray(
                        names.vmtkGroupIdsArrayName
                    )

        clGroupIdsList = list(set(clGroups))

        # Check whether the centerline have abscissas
        if bifGroupId in clGroupIdsList:

            # If the ica bif. and aneurysm group ids are on the centerline,
            # then use the updated clip value to clip (meaning to clip AFTER
            # the aneurysm abscissa)
            if iaGroupIdClosestBif in clGroupIdsList and \
               iaGroupIdClosestBif > bifGroupId:

                clipValue = iaAbscissasClosestBif + \
                            outlet_vessel_clip_value

            else:
                clipValue = outlet_vessel_clip_value

            # Use the GroupIds as keys to dict to avoid repetitive entries
            outletClipPointId = (
                                    np.abs(
                                        clClipField - clipValue
                                    )
                                ).argmin()

            outletClipPoint  = tuple(npClPortion.Points[outletClipPointId])
            outletClipNormal = tuple(
                                   npClPortion.PointData.GetArray(
                                       names.vmtkFrenetTangentArrayName
                                   )[outletClipPointId]
                               )

            outletClipPoints.update({
                clGroups[outletClipPointId]: {
                    "center": outletClipPoint,
                    "normal": outletClipNormal
                }
            })

        else:
            continue

    for dict_ in outletClipPoints.values():

        # Clip at outlets: invert normal at these positions
        vascular_surface = ClipVasculatureWithPlane(
                               vascular_surface,
                               plane_center=dict_["center"],
                               plane_normal=tuple(-val for val in dict_["normal"])
                           )

    return tools.CleanupArrays(vascular_surface)
