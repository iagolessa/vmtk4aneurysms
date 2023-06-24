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

"""Collection of tools to mechanically characterize an aneurysm wall motion."""

import sys
import vtk
from vtk.numpy_interface import dataset_adapter as dsa

from vmtk4aneurysms.lib import names
from vmtk4aneurysms.lib import constants as const
from vmtk4aneurysms.lib import polydatatools as tools
from vmtk4aneurysms.lib import polydatageometry as geo

from .vascular_operations import MarkAneurysmSacManually

# Dictionalry holding the IDs of each wall type
# See docstring of func WallTypeClassification
IaWallTypes = {"RegularWall": 0,
               "AtheroscleroticWall": 1,
               "RedWall": 2}


def WallTypeClassification(
        surface: names.polyDataType,
        low_wss: float=5.0,
        high_wss: float=10.0,
        low_osi: float=0.001,
        high_osi: float=0.01,
        distance_to_neck_array: str=names.DistanceToNeckArrayName,
        neck_iso_value: float=const.NeckIsoValue
    )   -> names.polyDataType:
    """Based on the WSS hemodynamics, characterize an aneurysm wall morphology.

    Based on the TAWSS and OSI fields, identifies the aneurysm regions prone to
    atherosclerotic walls (thicker walls) and red wall (thinner) by adding a
    new array on the passed surface name "WallType" with the following values:

    .. table:: Local wall type characterization
        :widths: auto

        =====   ===============
        Label   Wall Type
        =====   ===============
            0   Normal wall
            1   Atherosclerotic
            2   "Red" wall
        =====   ===============

    Classifications based on the references:

        [1] Furukawa et al. "Hemodynamic characteristics of hyperplastic
        remodeling lesions in cerebral aneurysms". PLoS ONE. 2018 Jan
        16;13:1–11.

        [2] Cebral et al. "Local hemodynamic conditions associated with focal
        changes in the intracranial aneurysm wall". American Journal of
        Neuroradiology.  2019; 40(3):510–6.
    """
    normalWall  = IaWallTypes["RegularWall"]
    thickerWall = IaWallTypes["AtheroscleroticWall"]
    thinnerWall = IaWallTypes["RedWall"]

    # Maybe put this limiting values to be passed by the user
    # for flexibility
    limitHemodynamics = {names.TAWSS: {"low": low_wss, "high": high_wss},
                         names.OSI  : {"low": low_osi, "high": high_osi}#,
                         #names.RRT  : {"low": 0.25,  "high": 0.75}
                        }

    arraysInSurface = tools.GetPointArrays(surface) + \
                      tools.GetCellArrays(surface)

    if distance_to_neck_array not in arraysInSurface:
        print("Distance to neck array name not in surface. Computing it.")

        surface = MarkAneurysmSacManually(
                      surface,
                      aneurysm_neck_array_name=distance_to_neck_array
                  )

    elif names.TAWSS not in arraysInSurface:
        raise ValueError("TAWSS array not in surface!")

    elif names.OSI not in arraysInSurface:
        raise ValueError("OSI array not in surface!")

    fieldsDf = tools.vtkPolyDataToDataFrame(surface)

    # Add int field which will indicate the thicker regions
    # zero indicates normal wall... the aneuysm portion wil be updated
    fieldsDf[names.WallTypeArrayName] = normalWall

    # Groups of conditions
    isAneurysm = fieldsDf[distance_to_neck_array] < const.NeckIsoValue

    isHighWss = fieldsDf[names.TAWSS] > limitHemodynamics[names.TAWSS]["high"]
    isLowWss  = fieldsDf[names.TAWSS] < limitHemodynamics[names.TAWSS]["low"]

    isHighOsi = fieldsDf[names.OSI] > limitHemodynamics[names.OSI]["high"]
    isLowOsi  = fieldsDf[names.OSI] < limitHemodynamics[names.OSI]["low"]

    # isHighRrt = fieldsDf[names.RRT] > limitHemodynamics[names.RRT]["high"]
    # isLowRrt = fieldsDf[names.RRT] < limitHemodynamics[names.RRT]["low"]

    thickerWallCondition = (isAneurysm) & (isLowWss)  & (isHighOsi)# & (isHighRrt)
    thinnerWallCondition = (isAneurysm) & (isHighWss) & (isLowOsi) # & (isLowRrt)

    # Update wall type array
    fieldsDf.loc[thickerWallCondition, names.WallTypeArrayName] = thickerWall
    fieldsDf.loc[thinnerWallCondition, names.WallTypeArrayName] = thinnerWall

    hemodynamicSurfaceNumpy = dsa.WrapDataObject(surface)

    # Add new field to surface
    hemodynamicSurfaceNumpy.CellData.append(
        dsa.VTKArray(fieldsDf[names.WallTypeArrayName]),
        names.WallTypeArrayName
    )

    return hemodynamicSurfaceNumpy.VTKObject

def DeformConfiguration(
        ref_config_obj,
        displ_field_name
    ):
    """Deform the undeformed configuration by a point field defined on it."""

    return geo.WarpVtkObject(
               ref_config_obj,
               displ_field_name
           )
