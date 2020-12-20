"""Mechanical characterization of aneurysms wall motion."""

import sys
import vtk

from .lib import constants as const
from .lib import polydatatools as tools
from .lib import polydatageometry as geo

from . import aneurysms as aneu

_polyDataType = vtk.vtkCommonDataModelPython.vtkPolyData

def AneurysmPulsatility(displacement_surface: _polyDataType,
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

def AneurysmPulsatility2(lumen_surface: _polyDataType,
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
