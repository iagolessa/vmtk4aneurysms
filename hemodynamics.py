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

"""Collection of tools to characterize the hemodynamics of vascular models.

Given the results of an OpenFOAM simulation of the flow in a vascular model,
with an aneurysm or not, provide tools to the hemodynamic characterization of
the wall shear stress (WSS) vector.
"""

import os
import sys
import warnings
import numpy as np

import vtk
from vmtk import vtkvmtk
from vtk.numpy_interface import dataset_adapter as dsa

from vmtk4aneurysms.lib import names
from vmtk4aneurysms.lib import constants as const
from vmtk4aneurysms.lib import polydatatools as tools
from vmtk4aneurysms.lib import polydatageometry as geo
from vmtk4aneurysms.lib import polydatamath as pmath
from vmtk4aneurysms.lib import foamtovtk as fvtk

from vmtk4aneurysms.aneurysms import SelectParentArtery

# Default density used for WSS
_density = 1056.0 # kg/m3

def _wss_over_time(
        foam_case: str,
        density: float=_density,
        field_name: str=names.foamWSS,
        patch: str=names.wallPatchName,
        multi_region: bool=False,
        region_name: str=''
    )   -> tuple:
    """Get surface object and the WSS vector field over time.

    Given the OpenFOAM case with the wall shear component calculated at each
    time-step, extracts the surface object (vtkPolyData) and the WSS vector
    field over time as a Python dictionary with the time-steps as keys and an
    VTKArray as the values. Returns a tuple with the surface object and the
    dictionary.

    The function also requires as optional arguments the density and the name
    of the patch where the WSS is defined.
    """

    surface, fieldsOverTime = fvtk.GetPatchFieldOverTime(
                                  foam_case,
                                  field_name,
                                  patch,
                                  multi_region=multi_region,
                                  region_name=region_name
                              )

    fieldOverTime = fieldsOverTime[field_name]

    # Compute the WSS = density * wallShearComponent
    wssVectorOverTime = {time: density*wssField
                         for time, wssField in fieldOverTime.items()}

    return surface, wssVectorOverTime

def _compute_gon(
        np_surface,
        temporal_wss,
        p_hat_array,
        q_hat_array,
        time_steps
    ):
    """Compute the Gradient Oscillatory Number (GON)."""

    setArray = np_surface.CellData.append
    delArray = np_surface.GetCellData().RemoveArray
    getArray = np_surface.CellData.GetArray

    GVecOverTime = []
    addInstantGVec = GVecOverTime.append

    for time in time_steps:
        wssVecDotQHat = pmath.HadamardDot(temporal_wss.get(time), q_hat_array)
        wssVecDotPHat = pmath.HadamardDot(temporal_wss.get(time), p_hat_array)

        setArray(wssVecDotPHat, names.WSSDotP)
        setArray(wssVecDotQHat, names.WSSDotQ)

        # Compute the surface gradient of (wss dot p) and (wss dot q)
        surfaceWithSGrad = geo.SurfaceGradient(np_surface.VTKObject, names.WSSDotP)
        surfaceWithSGrad = geo.SurfaceGradient(surfaceWithSGrad, names.WSSDotQ)

        tSurface = dsa.WrapDataObject(surfaceWithSGrad)

        # Now project each surface gradient on coordinate direction
        sGradDotPHat = pmath.HadamardDot(
                            tSurface.CellData.GetArray(names.WSSDotP+names.sgrad),
                            p_hat_array
                        )

        sGradDotQHat = pmath.HadamardDot(
                            tSurface.CellData.GetArray(names.WSSDotQ+names.sgrad),
                            q_hat_array
                        )

        tGVector = []
        append = tGVector.append

        for pComp, qComp in zip(sGradDotPHat, sGradDotQHat):
            append([pComp, qComp])

        addInstantGVec(dsa.VTKArray(tGVector))

    GVecOverTime = dsa.VTKArray(GVecOverTime)

    # Compute the norm of the averaged G vector
    period = max(time_steps) - min(time_steps)
    timeStep = period/len(time_steps)

    avgGVecArray = pmath.TimeAverage(
                       GVecOverTime,
                       np.array(time_steps)
                   )

    magAvgGVecArray = pmath.NormL2(avgGVecArray, 1)

    # Compute the average of the magnitude of G vec
    magGVecArray = pmath.NormL2(GVecOverTime, 2)

    avgMagGVecArray = pmath.TimeAverage(
                          magGVecArray,
                          np.array(time_steps)
                      )

    # TODO: divided by zero here?
    GON = 1.0 - magAvgGVecArray/avgMagGVecArray

    # Array clean-up
    delArray(names.WSSDotP)
    delArray(names.WSSDotQ)
    delArray(names.WSSDotP+names.sgrad)
    delArray(names.WSSDotQ+names.sgrad)

    setArray(GON, names.GON)
    setArray(avgGVecArray, names.WSSSG)
    setArray(avgMagGVecArray, names.WSSSGmag)

def Hemodynamics(
        foam_case: str,
        t_peak_systole: float,
        t_low_diastole: float,
        density: float=_density,  # kg/m3
        field_name: str=names.foamWSS,
        patch: str=names.wallPatchName,
        compute_gon: bool=False,
        compute_afi: bool=False,
        multi_region: bool=False,
        region_name: str=''
    )   -> names.polyDataType:
    """Compute WSS-derived hemodynamics of the flow in a vascular surface.

    Given the path to an OpenFOAM case and the wall patch name, compute
    temporal statistics of the wall shear stress (WSS) field:

        - time-averaged WSS (TAWSS);
        - peak-systole WSS (PSWSS);
        - low-diastole WSS (LDWSS).

    As also dependent parameters relevant for a complete hemodynamics
    characterization and important in the context of cerebral aneurysms and
    other vascular diseases:

        - oscillatory shear index (OSI);
        - relative residance time (RRT);
        - WSS pulsatility index (WSSPI);
        - WSS temporal gradient (WSSTG);
        - transversal WSS (transWSS);
        - time-averaged WSS gradient (TAWSSG);
        - the average WSS direction vector, :math:`\\vec{p}`;
        - the orthogonal vector to :math:`\\vec{p}` and the surface normal,
          :math:`\\vec{n}`, :math:`\\vec{q}`.

    The triad :math:`(\\vec{p},\\vec{q},\\vec{n}` forms a suitable coordinate
    system defined on the vascular surface.

    Optional parameters may also be computed by optional arguments:

        - aneurysm formation indication (AFI);
        - gradient oscillatory number (GON).

    Arguments
        - foam_case (str) -- path to the '.foam' file where the OpenFOAM case is;
        - t_peak_systole (float) -- peak systole instant;
        - t_low_diastole (float) -- low diastole instant;

    Optional
        - density (float) -- blood's assumed density (default 1056.0 kg/m3);
        - field_name (str) -- name of the WSS field name (default
          'wallShearComponent');
        - patch (str) -- name of the wall patch (default 'wall');
        - compute_gon (bool) -- to compute the GON (default False);
        - compute_afi (bool) -- to compute the AFI (default False);

    Return
        - vtkPolyData of the 'patch' surface with the variables fields.

    References
    ==========

        On the OSI, see:
        [1] He X and Ku DN. Pulsatile Flow in the Human Left Coronary Artery
        Bifurcation: Average Conditions. Journal of Biomechanical Engineering.
        1996;118:74–82.

        On the RRT, see:
        [2] Himburg et al. Spatial comparison between wall shear stress
        measures and porcine arterial endothelial permeability. American
        Journal of Physiology Heart and Circulatory Physiology.
        2004;286:1916–22.

        On the WSSTG, see:
        [3] Lee et al. Correlations among indicators of disturbed flow at the
        normal carotid bifurcation. Journal of Biomechanical Engineering.
        2009;131(6):1–7.

        On the transWSS, see:
        [4] Peiffer et al. Computation in the rabbit aorta of a new metric -
        the transverse wall shear stress - to quantify the multidirectional
        character of disturbed blood flow. Journal of Biomechanics.
        2013;46(15):2651–8.

        On the AFI, see:
        [5] Mantha et al. Hemodynamics in a Cerebral Artery before and after
        the Formation of an Aneurysm. American Journal of Neuroradiology.
        2006;27(May):1113–8.

        On the GON, see:
        [6] Shimogonya et al. Can temporal fluctuation in spatial wall shear
        stress gradient initiate a cerebral aneurysm? A proposed novel
        hemodynamic index, the gradient oscillatory number (GON). Journal of
        Biomechanics. 2009;42:550–4.
    """
    # Get WSS over time
    surface, temporalWss = _wss_over_time(
                               foam_case,
                               density=density,
                               field_name=field_name,
                               patch=patch,
                               multi_region=multi_region,
                               region_name=region_name
                           )

    # Compute WSS time statistics
    surface = fvtk.FieldTimeStats(
                  surface,
                  {names.WSS: temporalWss},
                  t_peak_systole,
                  t_low_diastole
              )

    # Compute normals and gradient of TAWSS
    surfaceWithNormals  = geo.Surface.Normals(surface)
    surfaceWithGradient = geo.SurfaceGradient(surfaceWithNormals, names.TAWSS)

    # Convert VTK polydata to numpy object
    numpySurface = dsa.WrapDataObject(surfaceWithGradient)

    # Functions to interface with the numpy vtk wrapper
    getArray = numpySurface.GetCellData().GetArray
    setArray = numpySurface.CellData.append

    # Get arrays currently on the surface
    # that will be used for the calculations
    avgVecWSSArray = getArray(names.WSS + names.avg)
    avgMagWSSArray = getArray(names.TAWSS)
    maxMagWSSArray = getArray(names.WSSmag + names.max_)
    minMagWSSArray = getArray(names.WSSmag + names.min_)
    normalsArray   = getArray(names.normals)
    sGradientArray = getArray(names.TAWSS + names.sgrad)

    # Compute the magnitude of the WSS vector time average
    magAvgVecWSSArray = pmath.NormL2(avgVecWSSArray, 1)

    # Several array will be stored at the end
    # of this procedure. So, create list to
    # store the array and its name (name, array).
    arraysToBeStored = []
    storeArray = arraysToBeStored.append

    # Compute WSS derived quantities
    # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
    storeArray(
        (names.WSSPI, (maxMagWSSArray - minMagWSSArray)/avgMagWSSArray)
    )

    OSIArray = 0.5*(1 - magAvgVecWSSArray/avgMagWSSArray)
    storeArray(
        (names.OSI, OSIArray)
    )

    storeArray(
        (names.RRT, 1.0/((1.0 - 2.0*OSIArray)*avgMagWSSArray))
    )

    # Calc surface orthogonal vectors
    # -> p: timeAvg WSS vector
    pHatArray = avgVecWSSArray/avgMagWSSArray
    storeArray(
        (names.pHat, pHatArray)
    )

    # -> q: perpendicular to p and normal
    qHatArray = np.cross(pHatArray, normalsArray)
    storeArray(
        (names.qHat, qHatArray)
    )

    # Compute the TAWSSG = surfaceGradTAWSS dot p
    storeArray(
        (names.TAWSSG, pmath.HadamardDot(pHatArray, sGradientArray))
    )

    if compute_afi:
        # AFI at peak-systole
        psWSS = getArray(names.peakSystoleWSS)
        psWSSmag = pmath.NormL2(psWSS, axis=1)

        storeArray(
            (names.AFI + '_peak_systole',
             pmath.HadamardDot(pHatArray, psWSS)/psWSSmag)
        )

    # Get time step list (ordered)
    timeSteps = list(temporalWss.keys())

    # Sort list of time steps
    timeSteps.sort()
    period = max(timeSteps) - min(timeSteps)
    timeStep = period/len(timeSteps)

    if compute_gon:
        _compute_gon(numpySurface,
                     temporalWss, pHatArray, qHatArray,
                     timeSteps)

    # TransWss = tavg(abs(wssVecOverTime dot qHat))
    # I had to compute the transWSS here because it needs
    # an averaged array (qHat) and a time-dependent array
    wssVecDotQHatProd = lambda time: abs(pmath.HadamardDot(temporalWss.get(time),
                                         qHatArray))

    # Get array with product
    wssVecDotQHat = dsa.VTKArray([wssVecDotQHatProd(time)
                                  for time in timeSteps])

    storeArray(
        (
            names.transWSS,
            pmath.TimeAverage(
                wssVecDotQHat,
                np.array(timeSteps)
            )
        )
    )

    # Compute the WSSTG = max(dWSSdt) over time
    magWssOverTime = np.array([pmath.NormL2(temporalWss.get(time), axis=1)
                               for time in timeSteps])

    dWssdt = np.gradient(magWssOverTime, timeStep, axis=0)

    storeArray(
        (names.WSSTG,
         dWssdt.max(axis=0))
    )

    for name, array in arraysToBeStored:
        setArray(array, name)

    return numpySurface.VTKObject

def NearWallTransportFeatures(
        vasc_surface: names.polyDataType,
        wss_field_name: str
    )   -> tuple:
    """Characterize the near wall transport adjacent to a vascular wall.

    The wall shear vector is a proxy of the blood flow velocity field near the
    luminal wall of the vessel or aneurysm, as explained in several works, for
    example:

        A. Arzani, A. M. Gambaruto, G. Chen, and S. C. Shadden, “Lagrangian
        wall shear stress structures and near-wall transport in
        high-Schmidt-number aneurysmal flows,” Journal of Fluid Mechanics, vol.
        790, pp. 158–172, 2016, doi: 10.1017/jfm.2016.6.

    Based on this fact, this function implements the two steps to characterize
    the near wall transport in a vascular wall, be it of a healthy wall or an
    intracranial aneurysm for example, proposed in the paper:

        V. Mazzi et al., “A Eulerian method to analyze wall shear stress fixed
        points and manifolds in cardiovascular flows,” Biomechanics and
        Modeling in Mechanobiology, vol. 19, no. 5, pp. 1403–1423, Oct. 2020,
        doi: 10.1007/s10237-019-01278-3.

    Specifically,

    (1) computation of the divergence of the normalized WSS vectorial field to
    identify the topological skeleton of that field; and
    (2) locate and characterize the fixed points of the field.

    Arguments
        - vasc_surface (polyDataType) -- the discretized vascular surface;
        - wss_field_name (str) -- the name of the WSS vector field defined on the surface;

    Return
        - A tuple with the same input surface but with the divergence field
          added to it and a polyData with the fixed points.
    """

    # Field names defined in this function
    normWssFieldName = wss_field_name + names.norm

    # Get surface as numpy data
    npSurface = dsa.WrapDataObject(vasc_surface)

    # Get WSS vector field
    vecWssField = npSurface.CellData.GetArray(wss_field_name)

    # Compute the normalized vectors
    # For the WSS particularly, use the reversed direction
    # because in OpenFOAM, the WSS was computed as it acts on the
    # fluid flow domain
    normVecWssField = -vecWssField/pmath.NormL2(vecWssField, 1)

    npSurface.CellData.append(
        normVecWssField,
        normWssFieldName
    )

    # Correct the orientation of the WSS vector for
    # the near-wall flow analysis specifically
    npSurface.CellData.append(
        -vecWssField,
        wss_field_name
    )

    # Add the divergence field
    # This surface is the one that must be returned
    divSurface = geo.Divergence(
                     npSurface.VTKObject,
                     normWssFieldName
                 )

    # Second part: identify the fixed-points with the
    # computation of the Poincaré index
    fixedPointsData = geo.ComputeSurfaceVectorFixedPoints(
                            divSurface,
                            wss_field_name
                        )

    return (divSurface, fixedPointsData)

def PressureTemporalStats(
        foam_case: str,
        t_peak_systole: float,
        t_low_diastole: float,
        density: float=_density,  # kg/m3
        p_field_name: str="p",
        patch: str=names.wallPatchName,
        multi_region: bool=False,
        region_name: str=''
    )   -> names.polyDataType:
    """Compute the pressure field temporal statistics.

    Given an OpenFOAM simulation case with a pressure field varying over time,
    compute its temporal statistics on a patch.
    """

    surface, fieldsOverTime = fvtk.GetPatchFieldOverTime(
                                  foam_case,
                                  p_field_name,
                                  patch,
                                  multi_region=multi_region,
                                  region_name=region_name
                              )

    fieldOverTime = fieldsOverTime[p_field_name]

    # Compute the WSS = density * wallShearComponent
    pOverTime = {time: density*pField
                 for time, pField in fieldOverTime.items()}

    # Compute WSS time statistics
    surface = fvtk.FieldTimeStats(
                  surface,
                  {"p": pOverTime},
                  t_peak_systole,
                  t_low_diastole
              )

    return surface

# This calculation depends on the WSS defined only on the parent artery
# surface. I think the easiest way to com- pute that is by drawing the artery
# contour in the same way as the aneurysm neck is beuild. So, I will assume in
# this function that the surface is already cut to in- clude only the parent
# artery portion and that includes
# TODO: improve the WSS stats over the parent
# artery by automatically computing a ring of the parent artery adjancent to
# the aneurysm
def WssParentVessel(
        parent_artery_surface: names.polyDataType,
        parent_artery_array: str = names.ParentArteryArrayName,
        parent_artery_iso_value: float = const.NeckIsoValue,
        wss_field: str = names.TAWSS
    )   -> float:
    """Calculates the surface averaged WSS value over the parent artery."""

    try:
        # Try to read if file name is given
        surface = tools.ReadSurface(parent_artery_surface)
    except:
        surface = parent_artery_surface

    # Check if surface has parent artery contour array
    pointArrays = tools.GetPointArrays(surface)

    if parent_artery_array not in pointArrays:
        # Compute parent artery portion
        surface = SelectParentArtery(surface)
    else:
        pass

    # Get parent artery portion
    parentArtery = tools.ClipWithScalar(surface,
                                        parent_artery_array,
                                        parent_artery_iso_value)

    # Get Array
    # npParentArtery = dsa.WrapDataObject(parentArtery)
    # wssArray = npParentArtery.GetCellData().GetArray(wss_field)
    # WSS averaged
    # maximum = np.max(np.array(wssArray))
    # minimum = np.min(np.array(wssArray))
    # percentile = np.percentile(np.array(wssArray), n_percentile)
    # average = np.average(np.array(wssArray))
    return pmath.SurfaceAverage(parentArtery, wss_field)

def WssSurfaceAverage(
        foam_case: str,
        neck_surface: names.polyDataType=None,
        neck_array_name: str=names.AneurysmNeckArrayName,
        neck_iso_value: float=const.NeckIsoValue,
        density: float=_density,
        field_name: str=names.foamWSS,
        patch: str=names.wallPatchName,
        multi_region: bool=False,
        region_name: str=''
    )   -> dict:
    """Compute the surface-averaged WSS over time.

    Given an OpenFOAM case file, compute surface integrals of WSS over an
    aneurysm or vessels surface.  Return a dictionary with the time instants as
    keys.

    .. note::
        Accepts as argument a surface derived form the simulations with a field
        named 'neck_array_name' that indicates the aneurysm portion with 0 and
        1 on the rest of the vasculature. The function uses this array to clip
        the aneurysm portion.  If this surface is not passed, the function
        computes the average over the whole surface patch.

    .. warning::
        It is essential that the 'neck_surface' with the 'neck_array_name' be
        the same as the wall surface of the OpenFOAM case, i.e.  they are the
        same mesh.
    """
    # Define condition to compute on aneurysm portion
    computeOnAneurysm = neck_surface is not None

    surface, temporalWss = _wss_over_time(
                               foam_case,
                               density=density,
                               field_name=field_name,
                               patch=patch,
                               multi_region=multi_region,
                               region_name=region_name
                           )

    # Map neck array into surface
    if computeOnAneurysm:
        # Map neck array field into current surface
        # (aneurysmExtract triangulas the surface)
        # Important: both surface must match the scaling

        surfaceProjection = vtkvmtk.vtkvmtkSurfaceProjection()
        surfaceProjection.SetInputData(surface)
        surfaceProjection.SetReferenceSurface(neck_surface)
        surfaceProjection.Update()

        surface = surfaceProjection.GetOutput()
    else:
        pass

    npSurface = dsa.WrapDataObject(surface)

    # Function to compute average of wss over surface
    def wss_average_on_surface(t):
        # Add WSSt array on surface
        wsst = temporalWss.get(t)
        npSurface.CellData.append(pmath.NormL2(wsst, 1), 'WSSt')

        if computeOnAneurysm:
            # Clip aneurysm portion
            aneurysm = tools.ClipWithScalar(npSurface.VTKObject,
                                            neck_array_name,
                                            neck_iso_value)

            npAneurysm = dsa.WrapDataObject(aneurysm)
            surfaceToComputeAvg = npAneurysm

        else:
            surfaceToComputeAvg = npSurface

        return pmath.SurfaceAverage(surfaceToComputeAvg.VTKObject, 'WSSt')

    return {time: wss_average_on_surface(time)
            for time in temporalWss.keys()}

def LsaInstant(
        foam_case: str,
        neck_surface: names.polyDataType,
        low_wss: float,
        neck_array_name: str=names.AneurysmNeckArrayName,
        neck_iso_value: float=const.NeckIsoValue,
        density: float=_density,
        field_name: str=names.foamWSS,
        patch: str=names.wallPatchName,
        multi_region: bool=False,
        region_name: str=''
    )   -> dict:
    """Compute the LSA over time.

    Given an OpenFOAM case file, compute the instantaneous LSA of an aneurysm.
    Return a dictionary with the time instants as keys.

    .. note::
        Accepts as argument a surface derived form the simulations with a field
        named 'neck_array_name' that indicates the aneurysm portion with 0 and
        1 on the rest of the vasculature. The function uses this array to clip
        the aneurysm portion.  If this surface is not passed, the function
        computes the average over the whole surface patch.

    .. warning::
        It is essential that the 'neck_surface' with the 'neck_array_name' be
        the same as the wall surface of the OpenFOAM case, i.e.  they are the
        same mesh.
    """

    # Check if neck_surface has aneurysm neck contour array
    neckSurfaceArrays = tools.GetPointArrays(neck_surface)

    if neck_array_name not in neckSurfaceArrays:
        sys.exit(neck_array_name + " not in surface!")
    else:
        pass

    # Get aneurysm surface are
    aneurysm = tools.ClipWithScalar(neck_surface,
                                    neck_array_name,
                                    neck_iso_value)

    aneurysmArea = geo.Surface.Area(aneurysm)

    # Compute WSS temporal for foam_case
    surface, temporalWss = _wss_over_time(
                               foam_case,
                               density=density,
                               field_name=field_name,
                               patch=patch,
                               multi_region=multi_region,
                               region_name=region_name
                           )

    # Project the aneurysm neck contour array to the surface
    # TODO: check if surface are equal (must match the scaling)
    surfaceProjection = vtkvmtk.vtkvmtkSurfaceProjection()
    surfaceProjection.SetInputData(surface)
    surfaceProjection.SetReferenceSurface(neck_surface)
    surfaceProjection.Update()

    surface = surfaceProjection.GetOutput()

    # Iterate over the wss fields over time
    npSurface = dsa.WrapDataObject(surface)

    def lsa_on_surface(t):
        wsst = temporalWss.get(t)
        npSurface.CellData.append(pmath.NormL2(wsst, 1), names.WSSmag)

        # Clip aneurysm portion
        aneurysm = tools.ClipWithScalar(npSurface.VTKObject,
                                        neck_array_name,
                                        neck_iso_value)

        # Get low shear area
        # Wonder: does the surface project works in only a portion of the
        # surface? If yes, I could do the mapping directly on the aneurysm
        lsaPortion = tools.ClipWithScalar(aneurysm, names.WSSmag, low_wss)
        lsaArea = geo.Surface.Area(lsaPortion)

        return lsaArea/aneurysmArea

    return {time: lsa_on_surface(time)
            for time in temporalWss.keys()}
