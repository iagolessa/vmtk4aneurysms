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

# Attribute array names
_polyDataType = vtk.vtkCommonDataModelPython.vtkPolyData
_multiBlockType = vtk.vtkCommonDataModelPython.vtkMultiBlockDataSet

_density = 1056.0 # SI units
_neck_value = 0.5

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
_aneurysmNeckArray = 'AneurysmNeckContourArray'
_parentArteryArray = 'ParentArteryContourArray'


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
    getArea = lambda id_: surface.GetCell(id_).ComputeArea()
    getValue = lambda id_, name: cellData.GetArray(name).GetValue(id_)

    def getCellValue(id_):
        cellArea = getArea(id_)
        arrayValue = getValue(id_, array_name)

        return cellArea, arrayValue

    integral = 0.0
    cellIds = range(surface.GetNumberOfCells())

    # Map function to cell ids
    for area, value in map(getCellValue, cellIds):
        integral += area*value

    surfaceArea = geo.SurfaceArea(surface)

    # Compute L2-norm 
    return integral/surfaceArea

def _HadamardDot(np_array1, np_array2):
    """Computes dot product in a Hadamard product way.
    
    Given two Numpy arrays representing arrays of vectors on a surface, compute
    the vector-wise dot product between each element.
    """
    # Seems that multiply is faster than a*b
    return np.multiply(np_array1, np_array2).sum(axis=1)

def _get_wall_with_wss(multi_block: _multiBlockType,
                       patch: str,
                       field: str) -> _polyDataType:

    # Get blocks and surface patch
    # TODO: I need to fetch oatch by name. How?
    nBlocks = multi_block.GetNumberOfBlocks()

    # Get outer surface (include inlet and outlets)
    surface = multi_block.GetBlock(nBlocks - 1)

    # Find wall patch (largest number of cells)
    SMALL = -1e10
    wallPatch = None
    nMaxCells = SMALL

    for block_id in range(surface.GetNumberOfBlocks()):
        patch = surface.GetBlock(block_id)

        # Get patch with largest number of cells
        # (forcely, the wall)
        nCells = patch.GetNumberOfCells()
        if nCells > nMaxCells:
            nMaxCells = nCells
            wallPatch = patch

    # Remove U and p arrays
    arrays = tools.GetCellArrays(wallPatch)

    # Remove arrays except field (WSS)
    arrays.remove(field)

    for array in arrays:
        wallPatch.GetCellData().RemoveArray(array)

    return wallPatch

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

    # Check if file or folder
    extension = os.path.splitext(foam_case)[-1]

    if not os.path.isfile(foam_case) and extension != '.foam':
        sys.exit("Unrecognized file format (must be .foam).")
    else:
        pass

    ofReader = vtk.vtkPOpenFOAMReader()

    # Apparently, it needs to be ran 2 times to load the data
    for _ in range(2):
        ofReader.SetFileName(foam_case)
        ofReader.AddDimensionsToArrayNamesOff()
        ofReader.DecomposePolyhedraOff()
        ofReader.SkipZeroTimeOn()
        ofReader.CreateCellToPointOff()

        # Apparently, this is important to load the
        # external surface
        ofReader.EnableAllPatchArrays()
        ofReader.Update()

    # Get list with time steps
    nTimeSteps = ofReader.GetTimeValues().GetNumberOfValues()
    timeSteps = list((ofReader.GetTimeValues().GetValue(id_) 
                      for id_ in range(nTimeSteps)))

    # Get WSS data
    wssVecOverTime = {}
    addInstantWss = wssVecOverTime.update

    # I will store one surface for reference here
    # to hold the final data
    baseSurface = None

    # for surface in map(readSurface, glob.glob(folder + '*.vtk')):
    for time in timeSteps:
        ofReader.UpdateTimeStep(time)

        surface = _get_wall_with_wss(ofReader.GetOutput(),
                                     patch=_wallPatch,
                                     field=_foamWSS)

        # Convert to Numpy object for efficiency
        npSurface = dsa.WrapDataObject(surface)
        tWssArray = density*npSurface.GetCellData().GetArray(_foamWSS)
        addInstantWss({time: tWssArray})

        # Store last surface
        # (for moving domain, should I use the first?)
        baseNpSurface = npSurface

    return baseNpSurface.VTKObject, wssVecOverTime

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

    Arguments:
        OpenFOAM case file (str) -- name of OpenFOAM .foam case;
        wssFieldName (str, optional) -- string containing the name of the wall
            shear stress field (default="wallShearComponent");
        patchName (str, optional) -- patch name where to calculate the OSI
            (default="wall");
        blood density (float, optional) -- default 1056.0 kg/m3.
    """
    npSurface = dsa.WrapDataObject(surface)
    
    # Get WSS over time in ordered manner
    timeSteps = list(temporal_wss.keys())
    
    # Sort list of time steps
    timeSteps.sort()
    wssVecOverTime = dsa.VTKArray([temporal_wss.get(time) 
                                   for time in timeSteps])

    # Compute the time-average of the magnitude of the WSS vector
    wssMagOverTime = _normL2(wssVecOverTime, 2)

    # Check if low diastole or peak systoel not in time list    
    lastTimeStep = max(timeSteps)
    firstTimeStep = min(timeSteps)
    
    if t_low_diastole not in timeSteps:
        warningMsg = "Low diastole instant not in " \
                     "time-steps list. Using last time-step."
        print(warningMsg, end='\n')

        t_low_diastole = lastTimeStep
    
    elif t_peak_systole not in timeSteps:
        warningMsg = "Peak-systole instant not in " \
                     "time-steps list. Using first time-step."
        print(warningMsg, end='\n')

        t_peak_systole = firstTimeStep

    # List of tuples to store stats arrays and their name
    # [(array1, name1), ... (array_n, name_n)]
    arraysToBeStored = []
    storeArray = arraysToBeStored.append
    
    # Get peak-systole and low-diastole WSS
    storeArray((temporal_wss[t_peak_systole], _peakSystoleWSS))
    storeArray((temporal_wss[t_low_diastole], _lowDiastoleWSS))
    
    # Get period of time steps
    period = lastTimeStep - firstTimeStep
    timeStep = period/len(timeSteps)
    
    # Append to the numpy surface wrap
    appendToSurface = npSurface.CellData.append

    # Compute the time-average of the WSS vector
    # assumes uniform time-step (calculated above)
    storeArray(
        (_time_average(wssVecOverTime, timeStep, period),
         _WSS + _avg)
    )
               
    storeArray(
        (_time_average(wssMagOverTime, timeStep, period),
         _TAWSS)
    )

    storeArray(
        (wssMagOverTime.max(axis=0),
         _WSSmag + _max)
    )

    storeArray(
        (wssMagOverTime.min(axis=0),
         _WSSmag + _min)
    )
    
    # Finally, append all arrays to surface
    for array, name in arraysToBeStored:
        appendToSurface(array, name)
    
    return npSurface.VTKObject


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

def _select_aneurysm(surface: _polyDataType) -> _polyDataType:
    """Compute array marking the aneurysm."""
    aneurysmSelection = vmtkscripts.vmtkSurfaceRegionDrawing()
    aneurysmSelection.Surface = surface
    aneurysmSelection.InsideValue = 0.0 # the aneurysm portion
    aneurysmSelection.OutsideValue = 1.0
    aneurysmSelection.ContourScalarsArrayName = _aneurysmNeckArray
    aneurysmSelection.Execute()

    smoother = vmtkscripts.vmtkSurfaceArraySmoothing()
    smoother.Surface = aneurysmSelection.Surface
    smoother.Connexity = 1
    smoother.Iterations = 10
    smoother.SurfaceArrayName = aneurysmSelection.ContourScalarsArrayName
    smoother.Execute()

    return smoother.Surface

def _select_parent_artery(surface: _polyDataType) -> _polyDataType:
    """Compute array masking the parent vessel region."""
    parentArteryDrawer = vmtkscripts.vmtkSurfaceRegionDrawing()
    parentArteryDrawer.Surface = surface
    parentArteryDrawer.InsideValue = 0.0
    parentArteryDrawer.OutsideValue = 1.0
    parentArteryDrawer.ContourScalarsArrayName = _parentArteryArray
    parentArteryDrawer.Execute()

    smoother = vmtkscripts.vmtkSurfaceArraySmoothing()
    smoother.Surface = parentArteryDrawer.Surface
    smoother.Connexity = 1
    smoother.Iterations = 10
    smoother.SurfaceArrayName = parentArteryDrawer.ContourScalarsArrayName
    smoother.Execute()

    return smoother.Surface

def hemodynamics(foam_case: str,
                 t_peak_systole: float,
                 t_low_diastole: float,
                 density=_density,  # kg/m3
                 field=_foamWSS,
                 patch=_wallPatch,
                 compute_gon=False,
                 compute_afi=False) -> _polyDataType:
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
    surfaceWithNormals  = geo.SurfaceNormals(surface)
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
    wssVecDotQHatProd = lambda time: abs(_HadamardDot(temporalWss.get(time), qHatArray))
    
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

def aneurysm_stats(neck_surface: _polyDataType,
                   neck_array_name: str,
                   array_name: str,
                   n_percentile: float = 95,
                   neck_iso_value: float = 0.5) -> dict:
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
    cellArrays = tools.GetCellArrays(neck_surface)

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
        neck_surface = _select_aneurysm(neck_surface) 
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

def lsa_wss_avg(neck_surface,
                neck_array_name,
                lowWSS,
                neck_iso_value=0.5,
                avgMagWSSArray=_TAWSS):
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
    aneurysmArea = geo.SurfaceArea(aneurysm)

    # Get low shear area
    lsaPortion = tools.ClipWithScalar(aneurysm, avgMagWSSArray, lowWSS)
    lsaArea = geo.SurfaceArea(lsaPortion)

    return lsaArea/aneurysmArea


# This calculation depends on the WSS defined only on the parent artery
# surface. I think the easiest way to com- pute that is by drawing the artery
# contour in the same way as the aneurysm neck is beuild. So, I will assume in
# this function that the surface is already cut to in- clude only the parent
# artery portion and that includes
def wss_parent_vessel(parent_artery_surface: _polyDataType,
                      parent_artery_array: str,
                      parent_artery_iso_value=0.5,
                      wss_field=_TAWSS) -> float:
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
        surface = _select_parent_artery(surface)
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

def wss_surf_avg(foam_case: str,
                 neck_surface: _polyDataType = None,
                 neck_array_name: str = _aneurysmNeckArray,
                 neck_iso_value: float = 0.5,
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

def lsa_instant(foam_case: str,
                neck_surface: _polyDataType,
                neck_array_name: str,
                low_wss: float,
                neck_iso_value=_neck_value,
                density=_density,
                field=_foamWSS,
                patch=_wallPatch) -> list:
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

    aneurysmArea = geo.SurfaceArea(aneurysm)

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
        lsaArea = geo.SurfaceArea(lsaPortion)

        return lsaArea/aneurysmArea

    return {time: lsa_on_surface(time) 
            for time in temporalWss.keys()}
