"""Mechanical characterization of aneurysms wall motion."""

import sys
import vtk
from vtk.numpy_interface import dataset_adapter as dsa

from .lib import names
from .lib import constants as const
from .lib import polydatatools as tools
from .lib import polydatageometry as geo

from . import aneurysms as aneu

def AneurysmPulsatility(displacement_surface: names.polyDataType,
                        ps_displ_field_name: str,
                        ld_displ_field_name: str,
                        aneurysm_neck_array_name: str = aneu.AneurysmNeckArrayName):
    """Compute aneurysm wall pulsatility.

    Aneurysm pulsatility computation, defined as (Sanchez et al. (2014)):

        Pulsatility = (V_ps/V_ld) - 1

    where "ld" indicate low diastole and "ps" indicate peak systole values. It
    uses the the lumen surface with the peak systole and low diastole
    displacement field on the surface. The input surface must alread have the
    aneurysm neck array, otherwise the function prompts the user to select the
    aneurysm neck contour.
    """

    # Warp whole surface at peak systole and low diastole
    ldLumenSurface = geo.WarpPolydata(displacement_surface,
                                      ld_displ_field_name)

    psLumenSurface = geo.WarpPolydata(displacement_surface,
                                      ps_displ_field_name)

    # With the surfaces warped by the displacement field,
    # now we just need to clip the aneurysm sac region.
    ldAneurysmSurface = tools.ClipWithScalar(ldLumenSurface,
                                             aneurysm_neck_array_name,
                                             aneu.NeckIsoValue)

    psAneurysmSurface = tools.ClipWithScalar(psLumenSurface,
                                             aneurysm_neck_array_name,
                                             aneu.NeckIsoValue)

    ldAneurysm = aneu.Aneurysm(ldAneurysmSurface)
    psAneurysm = aneu.Aneurysm(psAneurysmSurface)

    return psAneurysm.GetAneurysmVolume()/ldAneurysm.GetAneurysmVolume() - 1.0

def AneurysmPulsatility2(lumen_surface: names.polyDataType,
                         displacement_over_time: dict,
                         peak_systole_instant: float,
                         low_diastole_instant: float,
                         aneurysm_neck_array_name: str = aneu.AneurysmNeckArrayName):
    """Compute aneurysm wall pulsatility.

    Alternative version of the aneurysm pulsatility computation by using the
    lumen surface and the dictionary with the displacement field computed with
    the GetPatchFieldOverTime function. The input surface must alread have the
    aneurysm neck array, otherwise the function prompts the user to select the
    aneurysm neck contour.
    """

    ldDisplField = displacement_over_time.get(low_diastole_instant)
    psDisplField = displacement_over_time.get(peak_systole_instant)

    # Add both field to the surfaces, separately
    npLumenSurface = dsa.WrapDataObject(lumen_surface)
    npLumenSurface.GetCellData().append(ldDisplField, lowDiastoleDisplFieldName)
    npLumenSurface.GetCellData().append(psDisplField, peakSystoleDisplFieldName)

    lumenSurface = npLumenSurface.VTKObject

    # Project aneurysm neck contour to the surface
    if aneu.AneurysmNeckArrayName not in tools.GetPointArrays(lumenSurface):
        lumenSurface = aneu.SelectAneurysm(lumenSurface)
    else:
        pass

    # Warp whole surface at peak systole and low diastole
    ldLumenSurface = geo.WarpPolydata(lumenSurface, lowDiastoleDisplFieldName)
    psLumenSurface = geo.WarpPolydata(lumenSurface, peakSystoleDisplFieldName)

    # Clip aneurysm
    ldAneurysmSurface = tools.ClipWithScalar(ldLumenSurface,
                                             aneurysm_neck_array_name,
                                             aneu.NeckIsoValue)

    psAneurysmSurface = tools.ClipWithScalar(psLumenSurface,
                                             aneurysm_neck_array_name,
                                             aneu.NeckIsoValue)

    # Initiate aneurysm model
    ldAneurysm = aneu.Aneurysm(ldAneurysmSurface)
    psAneurysm = aneu.Aneurysm(psAneurysmSurface)

    # Compute pulsatility
    return psAneurysm.GetAneurysmVolume()/ldAneurysm.GetAneurysmVolume() - 1.0

def WallTypeCharacterization(surface: names.polyDataType,
                             low_wss: float = 5.0,
                             high_wss: float = 10.0,
                             low_osi: float = 0.001,
                             high_osi: float = 0.01) -> names.polyDataType:
    """Based on hemodynamics, characterize wall morphology.

    Based on the TAWSS and OSI fields, identifies the aneurysm regions prone to
    atherosclerotic walls (thicker walls) and red wall (thinner) by adding a
    new array on the passed surface name "WallType" with the following values:

        0 -> normal wall;
        1 -> atherosclerotic wall;
        2 -> thinner wall.

    Classifications based on the publications:

        Furukawa et al. "Hemodynamic characteristics of hyperplastic remodeling
        lesions in cerebral aneurysms". PLoS ONE. 2018 Jan 16;13:1–11.

    and

        Cebral et al. "Local hemodynamic conditions associated with focal
        changes in the intracranial aneurysm wall". American Journal of
        Neuroradiology.  2019; 40(3):510–6.
    """
    # Code
    normalWall  = 0
    thickerWall = 1
    thinnerWall = 2

    wallTypeArrayName = "WallType"
    aneurysmNeckArrayName = aneu.AneurysmNeckArrayName

    # Maybe put this limiting values to be passed by the user
    # for flexibility
    limitHemodynamics = {names.TAWSS: {"low": low_wss, "high": high_wss},
                         names.OSI  : {"low": low_osi, "high": high_osi}#,
                         #names.RRT  : {"low": 0.25,  "high": 0.75}
                        }

    arraysInSurface = tools.GetPointArrays(surface) + \
                      tools.GetCellArrays(surface)

    if aneurysmNeckArrayName not in arraysInSurface:
        surface = aneu.SelectAneurysm(surface)

    elif names.TAWSS not in arraysInSurface:
        sys.exit("TAWSS array not in surface!")

    elif names.OSI not in arraysInSurface:
        sys.exit("OSI array not in surface!")

    else:
        pass

    fieldsDf = tools.vtkPolyDataToDataFrame(surface)

    # Add int field which will indicate the thicker regions
    # zero indicates normal wall... the aneuysm portion wil be updated
    fieldsDf[wallTypeArrayName] = normalWall

    # Groups of conditions
    isAneurysm = fieldsDf[aneurysmNeckArrayName] < 0.5

    isHighWss = fieldsDf[names.TAWSS] > limitHemodynamics[names.TAWSS]["high"]
    isHighOsi = fieldsDf[names.OSI] > limitHemodynamics[names.OSI]["high"]
    # isHighRrt = fieldsDf[names.RRT] > limitHemodynamics[names.RRT]["high"]

    isLowWss = fieldsDf[names.TAWSS] < limitHemodynamics[names.TAWSS]["low"]
    isLowOsi = fieldsDf[names.OSI] < limitHemodynamics[names.OSI]["low"]
    # isLowRrt = fieldsDf[names.RRT] < limitHemodynamics[names.RRT]["low"]

    thickerWallCondition = (isAneurysm) & (isLowWss)  & (isHighOsi)# & (isHighRrt)
    thinnerWallCondition = (isAneurysm) & (isHighWss) & (isLowOsi) # & (isLowRrt)

    # Update wall type array
    fieldsDf.loc[thickerWallCondition, wallTypeArrayName] = thickerWall
    fieldsDf.loc[thinnerWallCondition, wallTypeArrayName] = thinnerWall

    hemodynamicSurfaceNumpy = dsa.WrapDataObject(surface)

    # Add new field to surface
    hemodynamicSurfaceNumpy.CellData.append(
        dsa.VTKArray(fieldsDf[wallTypeArrayName]),
        wallTypeArrayName
    )

    return hemodynamicSurfaceNumpy.VTKObject
