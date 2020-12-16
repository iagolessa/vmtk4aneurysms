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
import numpy as np
from scipy.integrate import simps

import vtk
from vmtk import vtkvmtk
from vmtk import vmtkscripts
from vtk.numpy_interface import dataset_adapter as dsa

from .lib import constants as const
from .lib import polydatatools as tools
from .lib import polydatageometry as geo

from . import aneurysms as aneu

# Attribute array names
_polyDataType = vtk.vtkCommonDataModelPython.vtkPolyData
_multiBlockType = vtk.vtkCommonDataModelPython.vtkMultiBlockDataSet

_density = 1056.0 # SI units

# Field suffixes
_avg = '_average'
_min = '_minimum'
_max = '_maximum'
_mag = '_magnitude'
_grad = '_gradient'
_sgrad = '_sgradient'

# Attribute array names
_WSS = 'WSS'
_OSI = 'OSI'
_RRT = 'RRT'
_AFI = 'AFI'
_GON = 'GON'
_TAWSS = 'TAWSS'
_WSSPI = 'WSSPI'
_WSSTG = 'WSSTG'
_TAWSSG = 'TAWSSG'
_transWSS = 'transWSS'

_WSSmag = _WSS + _mag
_peakSystoleWSS = 'PSWSS'
_lowDiastoleWSS = 'LDWSS'

_WSSSG = 'WSSSG' + _avg
_WSSSGmag = 'WSSSG' + _mag +_avg
_WSSDotP = 'WSSDotP'
_WSSDotQ = 'WSSDotQ'

# Possibly deprecated
_WSSGrad = 'WSSGradient'
_WSS_surf_avg = 'WSS_surf_avg'

# Local coordinate system
_pHat = 'pHat'
_qHat = 'qHat'
_normals = 'Normals'

# Other attributes
_foamWSS = 'wallShearComponent'
_wallPatch = 'wall'


def _normL2(array, axis):
    """Compute L2-norm of an array along an axis."""

    return np.linalg.norm(array, ord=2, axis=axis)

def _time_average(array, step, period):
    """Compute temporal average of a time-dependent variable."""

    return simps(array, dx=step, axis=0)/period

# TODO: improve this computattion. I thought about using vtkIntegrateAttributes
# but is not available in the version shipped with vmtk!
def _area_average(surface, array_name):
    """Compute area-averaged array over surface with first-order accuracy."""

    triangulate = vtk.vtkTriangleFilter()
    triangulate.SetInputData(surface)
    triangulate.Update()

    surface = triangulate.GetOutput()

    # Helper functions
    cellData = surface.GetCellData()
    getArea  = lambda id_: surface.GetCell(id_).ComputeArea()
    getValue = lambda id_, name: cellData.GetArray(name).GetValue(id_)

    def getCellValue(id_):
        cellArea   = getArea(id_)
        arrayValue = getValue(id_, array_name)

        return cellArea, arrayValue

    integral = 0.0
    cellIds = range(surface.GetNumberOfCells())

    # Map function to cell ids
    integral = sum(area*value for area, value in map(getCellValue, cellIds))

    surfaceArea = geo.Surface.Area(surface)

    # Compute L2-norm
    return integral/surfaceArea

def _HadamardDot(np_array1, np_array2):
    """Computes dot product in a Hadamard product way.

    Given two Numpy arrays representing arrays of vectors on a surface, compute
    the vector-wise dot product between each element.
    """
    # Seems that multiply is faster than a*b
    return np.multiply(np_array1, np_array2).sum(axis=1)

def GetPatchFieldOverTime(foam_case: str,
                          field_name: str,
                          active_patch_name: str) -> (_polyDataType, dict):
    """Gets a time-varying patch field from an OpenFOAM case.

    Given an OpenFOAM case, the field name and the patch name, return a tuple
    with the patch surface and a dictionary with the time-varying field with
    the instants as keys and the value the field given as a VTK Numpy Array.
    """

    # Read OF case reader
    ofReader = vtk.vtkPOpenFOAMReader()
    ofReader.SetFileName(foam_case)
    ofReader.AddDimensionsToArrayNamesOff()
    ofReader.DecomposePolyhedraOff()
    ofReader.SkipZeroTimeOn()
    ofReader.CreateCellToPointOff()
    ofReader.DisableAllLagrangianArrays()
    ofReader.DisableAllPointArrays()
    ofReader.EnableAllCellArrays()
    ofReader.Update()

    # Get list with time steps
    nTimeSteps = ofReader.GetTimeValues().GetNumberOfValues()
    timeSteps  = list((ofReader.GetTimeValues().GetValue(id_)
                       for id_ in range(nTimeSteps)))

    # Update OF reader with only selected patch
    patches = list((ofReader.GetPatchArrayName(index)
                    for index in range(ofReader.GetNumberOfPatchArrays())))

    if active_patch_name not in patches:
        message = "Patch {} not in geometry surface.".format(active_patch_name)
        sys.exit(message)
    else:
        pass

    patches.remove('internalMesh')

    # Set active patch
    for patchName in patches:
        if patchName == active_patch_name:
            ofReader.SetPatchArrayStatus(patchName, 1)
        else:
            ofReader.SetPatchArrayStatus(patchName, 0)

    ofReader.Update()

    # Get blocks and get surface block
    blocks  = ofReader.GetOutput()
    nBlocks = blocks.GetNumberOfBlocks()

    # Surface is the last one (0: internalMesh, 1:surface)
    surface = blocks.GetBlock(nBlocks - 1)

    # The active patch is the only one left
    activePatch = surface.GetBlock(0)

    # Check if array in surface
    cellArraysInPatch  = tools.GetCellArrays(activePatch)
    pointArraysInPatch = tools.GetPointArrays(activePatch)

    if field_name not in cellArraysInPatch:
        message = "Field {} not in surface patch {}.".format(field_name,
                                                             active_patch_name)

        sys.exit(message)
    else:
        pass

    npActivePatch = dsa.WrapDataObject(activePatch)

    def _get_field(time):
        ofReader.UpdateTimeStep(time)
        return npActivePatch.GetCellData().GetArray(field_name)

    fieldOverTime = {time: _get_field(time) for time in timeSteps}

    # Clean surface from any point or cell field
    activePatch = npActivePatch.VTKObject

    for arrayName in cellArraysInPatch:
        activePatch.GetCellData().RemoveArray(arrayName)

    for arrayName in pointArraysInPatch:
        activePatch.GetPointData().RemoveArray(arrayName)

    return npActivePatch.VTKObject, fieldOverTime

def FieldTimeStats(surface: _polyDataType,
                   field_name: str,
                   temporal_field: dict,
                   t_peak_systole: float,
                   t_low_diastole: float) -> _polyDataType:
    """Compute field time statistics from OpenFOAM data.

    Get time statistics of a field defined on a surface S
    over time for a cardiac cycle, generated with OpenFOAM. Outputs a surface
    with: the time-averaged of the field magnitude (if not a scalar), maximum
    and minimum over time, peak-systole and low-diastole fields.

    Arguments:
        surface (vtkPolyData) -- the surface where the field is defined;
        temporal_field (dict) -- a dictuionary with the field over each instant;
        t_peak_systole (float) -- instant of the peak systole;
        t_low_diastole (float) -- instant of the low diastole;
    """
    npSurface = dsa.WrapDataObject(surface)

    # Get field over time as a Numpy array in ordered manner
    timeSteps = list(temporal_field.keys())

    # Sort list of time steps
    timeSteps.sort()
    fieldOverTime = dsa.VTKArray([temporal_field.get(time)
                                  for time in timeSteps])

    # Assum that the input field is a scalar field
    # then assign the magnitude field to itself
    # it will be changed later if tensor order higher than 1
    fieldMagOverTime = fieldOverTime.copy()

    # Check if size of field equals number of cells
    if surface.GetNumberOfCells() not in fieldOverTime.shape:
        sys.exit("Size of surface and of field do not match.")
    else:
        pass

    # Check if low diastole or peak systoel not in time list
    lastTimeStep  = max(timeSteps)
    firstTimeStep = min(timeSteps)

    if t_low_diastole not in timeSteps:
        warningMsg = "Low diastole instant not in " \
                     "time-steps list. Using last time-step."
        warnings.warn(warningMsg)

        t_low_diastole = lastTimeStep

    elif t_peak_systole not in timeSteps:
        warningMsg = "Peak-systole instant not in " \
                     "time-steps list. Using first time-step."
        warnings.warn(warningMsg)

        t_peak_systole = firstTimeStep
    else:
        pass

    # List of tuples to store stats arrays and their name
    # [(array1, name1), ... (array_n, name_n)]
    arraysToBeStored = []
    storeArray = arraysToBeStored.append

    # Get peak-systole and low-diastole WSS
    storeArray(
        (temporal_field.get(t_peak_systole, None),
         _peakSystoleWSS if field_name == _WSS
                         else '_'.join(["peak_systole", field_name]))
    )

    storeArray(
        (temporal_field.get(t_low_diastole, None),
         _lowDiastoleWSS if field_name == _WSS
                         else '_'.join(["low_diastole", field_name]))
    )

    # Get period of time steps
    period   = lastTimeStep - firstTimeStep
    timeStep = period/len(timeSteps)

    # Append to the numpy surface wrap
    appendToSurface = npSurface.CellData.append

    # Compute the time-average of the WSS vector
    # assumes uniform time-step (calculated above)
    storeArray(
        (_time_average(fieldOverTime, timeStep, period),
         field_name + _avg)
    )

    # If the array is a tensor of order higher than one
    # compute its magnitude too
    if len(fieldOverTime.shape) == 3:
        # Compute the time-average of the magnitude of the WSS vector
        fieldMagOverTime = _normL2(fieldOverTime, 2)

        storeArray(
            (_time_average(fieldMagOverTime, timeStep, period),
             _TAWSS if field_name == _WSS else field_name + _mag + _avg)
        )

    else:
        pass

    storeArray(
        (fieldMagOverTime.max(axis=0),
         field_name + _mag + _max)
    )

    storeArray(
        (fieldMagOverTime.min(axis=0),
         field_name + _mag + _min)
    )

    # Finally, append all arrays to surface
    for array, name in arraysToBeStored:
        appendToSurface(array, name)

    return npSurface.VTKObject

def _wss_over_time(foam_case: str,
                   density=_density,
                   field=_foamWSS,
                   patch=_wallPatch) -> tuple:
    """Get surface object and the WSS vector field over time.

    Given the OpenFOAM case with the WSS calculated at each time-step, extracts
    the surface object (vtkPolyData) and the WSS vector field over time as a
    Python dictionary with the time-steps as keys and an VTKArray as the
    values. Returns a tuple with the surface object and the dictionary.

    The function also requires as optional arguments the density and the name
    of the patch where the WSS is defined.
    """

    surface, fieldOverTime = GetPatchFieldOverTime(foam_case,
                                                   field,
                                                   patch)

    # Compute the WSS = density * wallShearComponent
    wssVectorOverTime = {time: _density*wssField
                         for time, wssField in fieldOverTime.items()}

    return surface, wssVectorOverTime

def _wss_time_stats(surface: _polyDataType,
                    temporal_wss: dict,
                    t_peak_systole: float,
                    t_low_diastole: float) -> _polyDataType:
    """Compute WSS time statistics from OpenFOAM data.

    Get time statistics of the wall shear stress field defined on a surface S
    over time for a cardiac cycle, generated with OpenFOAM. Outputs a surface
    with: the time-averaged WSS, maximum and minimum over time, peak-systole
    and low-diastole WSS vector fields. Since this function use OpenFOAM data,
    specify the density considered.
    """

    return FieldTimeStats(surface, _WSS,
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
        wssVecDotQHat = _HadamardDot(temporal_wss.get(time), q_hat_array)
        wssVecDotPHat = _HadamardDot(temporal_wss.get(time), p_hat_array)

        setArray(wssVecDotPHat, _WSSDotP)
        setArray(wssVecDotQHat, _WSSDotQ)

        # Compute the surface gradient of (wss dot p) and (wss dot q)
        surfaceWithSGrad = geo.SurfaceGradient(np_surface.VTKObject, _WSSDotP)
        surfaceWithSGrad = geo.SurfaceGradient(surfaceWithSGrad, _WSSDotQ)

        tSurface = dsa.WrapDataObject(surfaceWithSGrad)

        # Now project each surface gradient on coordinate direction
        sGradDotPHat = _HadamardDot(
                            tSurface.CellData.GetArray(_WSSDotP+_sgrad),
                            p_hat_array
                        )

        sGradDotQHat = _HadamardDot(
                            tSurface.CellData.GetArray(_WSSDotQ+_sgrad),
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

    avgGVecArray = _time_average(GVecOverTime, timeStep, period)
    magAvgGVecArray = _normL2(avgGVecArray, 1)

    # Compute the average of the magnitude of G vec
    magGVecArray = _normL2(GVecOverTime, 2)
    avgMagGVecArray = _time_average(magGVecArray, timeStep, period)

    GON = 1.0 - magAvgGVecArray/avgMagGVecArray

    # Array clean-up
    delArray(_WSSDotP)
    delArray(_WSSDotQ)
    delArray(_WSSDotP+_sgrad)
    delArray(_WSSDotQ+_sgrad)

    setArray(GON, _GON)
    setArray(avgGVecArray, _WSSSG)
    setArray(avgMagGVecArray, _WSSSGmag)


def Hemodynamics(foam_case: str,
                 t_peak_systole: float,
                 t_low_diastole: float,
                 density: float = _density,  # kg/m3
                 field: str = _foamWSS,
                 patch: str = _wallPatch,
                 compute_gon: bool = False,
                 compute_afi: bool = False) -> _polyDataType:
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
    surfaceWithGradient = geo.SurfaceGradient(surfaceWithNormals, _TAWSS)

    # Convert VTK polydata to numpy object
    numpySurface = dsa.WrapDataObject(surfaceWithGradient)

    # Functions to interface with the numpy vtk wrapper
    getArray = numpySurface.GetCellData().GetArray
    setArray = numpySurface.CellData.append

    # Get arrays currently on the surface
    # that will be used for the calculations
    avgVecWSSArray = getArray(_WSS + _avg)
    avgMagWSSArray = getArray(_TAWSS)
    maxMagWSSArray = getArray(_WSSmag + _max)
    minMagWSSArray = getArray(_WSSmag + _min)
    normalsArray   = getArray(_normals)
    sGradientArray = getArray(_TAWSS + _sgrad)

    # Compute the magnitude of the WSS vector time average
    magAvgVecWSSArray = _normL2(avgVecWSSArray, 1)

    # Several array will be stored at the end
    # of this procedure. So, create list to
    # store the array and its name (name, array).
    arraysToBeStored = []
    storeArray = arraysToBeStored.append

    # Compute WSS derived quantities
    # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
    storeArray(
        (_WSSPI, (maxMagWSSArray - minMagWSSArray)/avgMagWSSArray)
    )


    OSIArray = 0.5*(1 - magAvgVecWSSArray/avgMagWSSArray)
    storeArray(
        (_OSI, OSIArray)
    )

    storeArray(
        (_RRT, 1.0/((1.0 - 2.0*OSIArray)*avgMagWSSArray))
    )

    # Calc surface orthogonal vectors
    # -> p: timeAvg WSS vector
    pHatArray = avgVecWSSArray/avgMagWSSArray
    storeArray(
        (_pHat, pHatArray)
    )

    # -> q: perpendicular to p and normal
    qHatArray = np.cross(pHatArray, normalsArray)
    storeArray(
        (_qHat, qHatArray)
    )

    # Compute the TAWSSG = surfaceGradTAWSS dot p
    storeArray(
        (_TAWSSG, _HadamardDot(pHatArray, sGradientArray))
    )

    if compute_afi:
        # AFI at peak-systole
        psWSS = getArray(_peakSystoleWSS)
        psWSSmag = _normL2(psWSS, axis=1)

        storeArray(
            (_AFI + '_peak_systole',
             _HadamardDot(pHatArray, psWSS)/psWSSmag)
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
    wssVecDotQHatProd = lambda time: abs(_HadamardDot(temporalWss.get(time),
                                         qHatArray))

    # Get array with product
    wssVecDotQHat = dsa.VTKArray([wssVecDotQHatProd(time)
                                  for time in timeSteps])

    storeArray(
        (_transWSS,
         _time_average(wssVecDotQHat, timeStep, period))
    )

    # Compute the WSSTG = max(dWSSdt) over time
    magWssOverTime = np.array([_normL2(temporalWss.get(time), axis=1)
                               for time in timeSteps])

    dWssdt = np.gradient(magWssOverTime, timeStep, axis=0)

    storeArray(
        (_WSSTG,
         dWssdt.max(axis=0))
    )

    for name, array in arraysToBeStored:
        setArray(array, name)

    return numpySurface.VTKObject

def AneurysmStats(neck_surface: _polyDataType,
                  array_name: str,
                  neck_array_name: str = aneu.AneurysmNeckArrayName,
                  n_percentile: float = 95,
                  neck_iso_value: float = aneu.NeckIsoValue) -> dict:
    """Compute statistics of array on aneurysm surface.

    Given a surface with the fields of hemodynamics variables defined on it,
    computes the average, maximum, minimum, percetile (value passed as optional
    by the user) and the area averaged over the aneurysm surface.  Assumes that
    the surface also contain a binary array value that indicates the aneurysm
    portion with 0 and 1 the rest of the vasculature. The function uses this
    array to clip the aneurysm portion. If this is not present on the surface,
    the function prompts the user to delineate the aneurysm neck.
    """

    pointArrays = tools.GetPointArrays(neck_surface)
    cellArrays  = tools.GetCellArrays(neck_surface)

    arrayInSurface = array_name in pointArrays or \
                     array_name in cellArrays

    neckArrayInSurface = neck_array_name in pointArrays or \
                         neck_array_name in cellArrays

    # Check if arrays are on surface
    if not arrayInSurface:
        sys.exit(array_name + " not found in the surface.")
    else:
        pass

    if not neckArrayInSurface:
        # Compute neck array
        neck_surface = aneu.SelectAneurysm(neck_surface)
    else:
        pass

    # Get aneurysm
    aneurysm = tools.ClipWithScalar(
                    neck_surface,
                    neck_array_name,
                    neck_iso_value
                )

    # Get Array
    npAneurysm = dsa.WrapDataObject(aneurysm)

    arrayOnAneurysm = npAneurysm.GetCellData().GetArray(array_name)

    # Check type of field: vector or scalar
    nComponents = arrayOnAneurysm.shape[-1]

    if nComponents == 3:
        arrayOnAneurysm = _normL2(arrayOnAneurysm, 1)
    else:
        pass

    # WSS averaged
    maximum = np.max(arrayOnAneurysm)
    minimum = np.min(np.array(arrayOnAneurysm))
    average = np.average(np.array(arrayOnAneurysm))
    percentile  = np.percentile(np.array(arrayOnAneurysm), n_percentile)
    areaAverage = _area_average(aneurysm, array_name)

    return {'surf_avg': _area_average(aneurysm, array_name),
            'average': np.average(arrayOnAneurysm),
            'maximum': np.max(arrayOnAneurysm),
            'minimum': np.min(arrayOnAneurysm),
            str(n_percentile)+'percentil': np.percentile(arrayOnAneurysm,
                                                         n_percentile)}

def LsaAverage(neck_surface: _polyDataType,
                  lowWSS: float,
                  neck_array_name: str = aneu.AneurysmNeckArrayName,
                  neck_iso_value: float = aneu.NeckIsoValue,
                  avgMagWSSArray: str = _TAWSS):
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
def WssParentVessel(parent_artery_surface: _polyDataType,
                    parent_artery_array: str = aneu.ParentArteryArrayName,
                    parent_artery_iso_value: float = aneu.NeckIsoValue,
                    wss_field: str = _TAWSS) -> float:
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
    return _area_average(parentArtery, wss_field)

def WssSurfaceAverage(foam_case: str,
                      neck_surface: _polyDataType = None,
                      neck_array_name: str = aneu.AneurysmNeckArrayName,
                      neck_iso_value: float = aneu.NeckIsoValue,
                      density: float = _density,
                      field: str = _foamWSS,
                      patch: str = _wallPatch):
    """Compute the surface-averaged WSS over time.

    Function to compute surface integrals of WSS over an aneurysm or vessels
    surface. It takes the Open- FOAM case file and an optional surface where it
    is stored a field with the aneurysm neck line loaded as a ParaView PolyData
    surface. If the surface is None, it computes the integral over the entire
    sur- face. It is essential that the surface with the ne- ck array be the
    same as the wall surface of the OpenFOAM case, i.e. they are the same mesh.
    """
    # Define condition to compute on aneurysm portion
    computeOnAneurysm = neck_surface is not None and neck_array_name is not None

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
        npSurface.CellData.append(_normL2(wsst, 1), 'WSSt')

        if computeOnAneurysm:
            # Clip aneurysm portion
            aneurysm = tools.ClipWithScalar(npSurface.VTKObject,
                                            neck_array_name,
                                            neck_iso_value)

            npAneurysm = dsa.WrapDataObject(aneurysm)
            surfaceToComputeAvg = npAneurysm

        else:
            surfaceToComputeAvg = npSurface

        return _area_average(surfaceToComputeAvg.VTKObject, 'WSSt')

    return {time: wss_average_on_surface(time)
            for time in temporalWss.keys()}

def LsaInstant(foam_case: str,
               neck_surface: _polyDataType,
               low_wss: float,
               neck_array_name: str = aneu.AneurysmNeckArrayName,
               neck_iso_value: float = aneu.NeckIsoValue,
               density: float = _density,
               field: str = _foamWSS,
               patch: str = _wallPatch) -> list:
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
        npSurface.CellData.append(_normL2(wsst, 1), _WSSmag)

        # Clip aneurysm portion
        aneurysm = tools.ClipWithScalar(npSurface.VTKObject,
                                        neck_array_name,
                                        neck_iso_value)

        # Get low shear area
        # Wonder: does the surface project works in only a portion of the
        # surface? If yes, I could do the mapping directly on the aneurysm
        lsaPortion = tools.ClipWithScalar(aneurysm, _WSSmag, low_wss)
        lsaArea = geo.Surface.Area(lsaPortion)

        return lsaArea/aneurysmArea

    return {time: lsa_on_surface(time)
            for time in temporalWss.keys()}
