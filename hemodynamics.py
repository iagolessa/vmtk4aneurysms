"""Hemodynamics characterization of the flow in a vascular model.

Given an OpenFOAM\R simulation results of the flow in a vasculature, computes
the hemodynamic characterization of the wall shear stress (WSS) vector. The
main function provided by the module is the 'hemodynamics' function: given
the FOAM case with the WSS field over time, it already computes all the
parameters that are WSS dependent as arrays defined on the surface of the
model.

The module also provides the 'aneurysm_stats' function that computes the
statistics of any of the fields calculated on the 'hemodynamics' over the
aneurysm surface, if this is the case.
"""

import os
import sys
import warnings
import numpy as np

import vtk
from vmtk import vtkvmtk
from vtk.numpy_interface import dataset_adapter as dsa

from .lib import names
from .lib import constants as const
from .lib import polydatatools as tools
from .lib import polydatageometry as geo
from .lib import polydatamath as pmath
from .lib import foamtovtk as fvtk

from . import aneurysms as aneu

# Default density used for WSS
_density = 1056.0 # kg/m3

def _wss_over_time(foam_case: str,
                   density=_density,
                   field=names.foamWSS,
                   patch=names.wallPatchName) -> tuple:
    """Get surface object and the WSS vector field over time.

    Given the OpenFOAM case with the WSS calculated at each time-step, extracts
    the surface object (vtkPolyData) and the WSS vector field over time as a
    Python dictionary with the time-steps as keys and an VTKArray as the
    values. Returns a tuple with the surface object and the dictionary.

    The function also requires as optional arguments the density and the name
    of the patch where the WSS is defined.
    """

    surface, fieldsOverTime = fvtk.GetPatchFieldOverTime(
                                  foam_case,
                                  field,
                                  patch
                              )

    fieldOverTime = fieldsOverTime[field]

    # Compute the WSS = density * wallShearComponent
    wssVectorOverTime = {time: _density*wssField
                         for time, wssField in fieldOverTime.items()}

    return surface, wssVectorOverTime

def _wss_time_stats(surface: names.polyDataType,
                    temporal_wss: dict,
                    t_peak_systole: float,
                    t_low_diastole: float) -> names.polyDataType:
    """Compute WSS time statistics from OpenFOAM data.

    Get time statistics of the wall shear stress field defined on a surface S
    over time for a cardiac cycle, generated with OpenFOAM. Outputs a surface
    with: the time-averaged WSS, maximum and minimum over time, peak-systole
    and low-diastole WSS vector fields. Since this function use OpenFOAM data,
    specify the density considered.
    """

    return fvtk.FieldTimeStats(surface, names.WSS,
                               temporal_wss,
                               t_peak_systole,
                               t_low_diastole)

def _compute_gon(np_surface,
                 temporal_wss,
                 p_hat_array,
                 q_hat_array,
                 time_steps):
    """Computes the Gradient Oscillatory Number (GON)."""

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


def Hemodynamics(foam_case: str,
                 t_peak_systole: float,
                 t_low_diastole: float,
                 density: float = _density,  # kg/m3
                 field: str = names.foamWSS,
                 patch: str = names.wallPatchName,
                 compute_gon: bool = False,
                 compute_afi: bool = False) -> names.polyDataType:
    """Compute hemodynamics of WSS field.

    Based on the temporal statistics of the WSS field over a vascular and
    aneurysm surface, compute the following parameters: oscillatory shear index
    (OSI), relative residance time (RRT), WSS pulsatility index (WSSPI), the
    time-averaged WSS gradient, TAWSSG, the average WSS direction vector, p,
    and orthogonal, q, to p and the normal, n, to the surface. The triad (p, q,
    n) is a suitable coordinate system defined on the vascular surface.
    """
    # Get WSS over time
    surface, temporalWss = _wss_over_time(foam_case,
                                          density=density,
                                          field=field,
                                          patch=patch)

    # Compute WSS time statistics
    surface = _wss_time_stats(surface,
                              temporalWss,
                              t_peak_systole,
                              t_low_diastole)

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

def AneurysmStats(
        neck_surface: names.polyDataType,
        array_name: str,
        neck_array_name: str=aneu.AneurysmNeckArrayName,
        n_percentile: float=99,
        neck_iso_value: float=aneu.NeckIsoValue
    )   -> dict:
    """Compute statistics of array on aneurysm surface.

    Given a surface with the fields of hemodynamics variables defined on it,
    computes the average, maximum, minimum, percetile (value passed as optional
    by the user) and the area averaged over the aneurysm surface.  Assumes that
    the surface also contain a binary array value that indicates the aneurysm
    portion with 0 and 1 on the rest of the vasculature. The function uses this
    array to clip the aneurysm portion. If this is not present on the surface,
    the function prompts the user to delineate the aneurysm neck.
    """

    pointArrays = tools.GetPointArrays(neck_surface)
    cellArrays  = tools.GetCellArrays(neck_surface)

    neckArrayInSurface = neck_array_name in pointArrays or \
                         neck_array_name in cellArrays

    if not neckArrayInSurface:
        # Compute neck array
        neck_surface = aneu.SelectAneurysm(neck_surface)

    # Get aneurysm
    aneurysmSurface = tools.ClipWithScalar(
                          neck_surface,
                          neck_array_name,
                          neck_iso_value
                      )

    return pmath.SurfaceFieldStatistics(
               aneurysmSurface,
               array_name,
               n_percentile=n_percentile
           )

def LsaAverage(neck_surface: names.polyDataType,
               lowWSS: float,
               neck_array_name: str = aneu.AneurysmNeckArrayName,
               neck_iso_value: float = aneu.NeckIsoValue,
               avgMagWSSArray: str = names.TAWSS):
    """Computes the LSA based on the time-averaged WSS field.

    Calculates the LSA (low WSS area ratio) for aneurysms simulations performed
    in OpenFOAM. Thi input is a sur- face with the time-averaged WSS over the
    surface and an array defined on it indicating the aneurysm neck.  The
    function then calculates the aneurysm surface area and the area where the
    WSS is lower than a reference value provided by the user.
    """
    try:
        # Try to read if file name is given
        surface = tools.ReadSurface(neck_surface)
    except:
        surface = neck_surface

    # Get aneurysm
    aneurysm = tools.ClipWithScalar(surface, neck_array_name, neck_iso_value)

    # Get aneurysm area
    aneurysmArea = geo.Surface.Area(aneurysm)

    # Get low shear area
    lsaPortion = tools.ClipWithScalar(aneurysm, avgMagWSSArray, lowWSS)
    lsaArea = geo.Surface.Area(lsaPortion)

    return lsaArea/aneurysmArea


# This calculation depends on the WSS defined only on the parent artery
# surface. I think the easiest way to com- pute that is by drawing the artery
# contour in the same way as the aneurysm neck is beuild. So, I will assume in
# this function that the surface is already cut to in- clude only the parent
# artery portion and that includes
def WssParentVessel(parent_artery_surface: names.polyDataType,
                    parent_artery_array: str = aneu.ParentArteryArrayName,
                    parent_artery_iso_value: float = aneu.NeckIsoValue,
                    wss_field: str = names.TAWSS) -> float:
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
        surface = aneu.SelectParentArtery(surface)
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

def WssSurfaceAverage(foam_case: str,
                      neck_surface: names.polyDataType = None,
                      neck_array_name: str = aneu.AneurysmNeckArrayName,
                      neck_iso_value: float = aneu.NeckIsoValue,
                      density: float = _density,
                      field: str = names.foamWSS,
                      patch: str = names.wallPatchName):
    """Compute the surface-averaged WSS over time.

    Function to compute surface integrals of WSS over an aneurysm or vessels
    surface. It takes the Open- FOAM case file and an optional surface where it
    is stored a field with the aneurysm neck line loaded as a ParaView PolyData
    surface. If the surface is None, it computes the integral over the entire
    sur- face. It is essential that the surface with the ne- ck array be the
    same as the wall surface of the OpenFOAM case, i.e. they are the same mesh.
    """
    # Define condition to compute on aneurysm portion
    computeOnAneurysm = neck_surface is not None

    surface, temporalWss = _wss_over_time(foam_case,
                                          density=density,
                                          field=field,
                                          patch=patch)

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

def LsaInstant(foam_case: str,
               neck_surface: names.polyDataType,
               low_wss: float,
               neck_array_name: str = aneu.AneurysmNeckArrayName,
               neck_iso_value: float = aneu.NeckIsoValue,
               density: float = _density,
               field: str = names.foamWSS,
               patch: str = names.wallPatchName) -> list:
    """Compute the LSA over time.

    Calculates the LSA (low WSS area ratio) for aneurysm simulations performed
    in OpenFOAM. The input is a surface with the time-averaged WSS over the
    surface an OpenFOAM case with the WSS field and a surface which contains
    the array with the aneurysm neck iso line.  The function then calculates
    the aneurysm surface area and the area where the WSS is lower than a
    reference value provided by the user, for each instant in the cycles
    simulated, returning a list with the LSA values over time, for the last
    cycle.
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
    surface, temporalWss = _wss_over_time(foam_case,
                                          density=density,
                                          field=field,
                                          patch=patch)

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
