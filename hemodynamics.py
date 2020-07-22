"""
Python library of functions to calculate morphological and hemodynamic 
parameters related to aneurysms geometry and hemodynamics using ParaView 
filters.

The library works with the paraview.simple module. 
"""

# import paraview.simple as pv

import vtk
import numpy as np

from constants import *
from vtk.numpy_interface import dataset_adapter as dsa
from scipy.integrate import simps

import polydatageometry as geo
import polydatatools as tools

# Attribute array names
_polyDataType = vtk.vtkCommonDataModelPython.vtkPolyData
_multiBlockType = vtk.vtkCommonDataModelPython.vtkMultiBlockDataSet

_density = 1056.0 # SI units

# Attribute array names
_Area = 'Area'
_WSS = 'WSS'
_TAWSS = 'TAWSS'
_OSI = 'OSI'
_RRT = 'RRT'
_AFI = 'AFI'
_WSSPI = 'WSSPI'
_TAWSSG = 'TAWSSG'
_PSWSS = 'PSWSS'
_LDWSS = 'LDWSS'
_WSSmag = 'WSS_magnitude'
_WSSGrad = 'WSSGradient'
_transWSS = 'transWSS'
_WSS_surf_avg = 'WSS_surf_avg'
_GON = 'GON'
_SWSSG = 'SWSSG'+_avg
_SWSSGmag = 'SWSSG_mag'+_avg
_WSSDotP = 'WSSDotP'
_WSSDotQ = 'WSSDotQ'

# Local coordinate system
_pHat = 'pHat'
_qHat = 'qHat'

_normals = 'Normals'
_xAxis = '_X'
_yAxis = '_Y'
_zAxis = '_Z'
_avg = '_average'
_min = '_minimum'
_max = '_maximum'
_grad = '_gradient'
_sgrad = '_sgradient'

# Other attributes
_foamWSS = 'wallShearComponent'
_wallPatch = 'wall'
_aneurysmArray = 'AneurysmNeckArray'
_parentArteryArray = 'ParentArteryArray'

# _cellDataMode = 'Cell Data'
# _pointDataMode = 'Point Data'

def _normL2(array, axis):
    """Compute L2-norm of an array along an axis."""

    return np.linalg.norm(array, ord=2, axis=axis)

def _time_average(array, step, period):
    """Compute temporal average of a time-dependent variable."""
    
    return simps(array, dx=step, axis=0)/period

def _HadamardDot(np_array1, np_array2):
    """Computes dot product in a Hadamard product way.
    
    Given two Numpy arrays representing arrays of vectors
    on a surface, compute the vector-wise dot product 
    between each element.
    """
    # Seems that multiply is faster than a*b
    return np.multiply(np_array1, np_array2).sum(axis=1)

def _get_wall_surface(multi_block: _multiBlockType) -> _polyDataType:

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
    wallPatch.GetCellData().RemoveArray("U")
    wallPatch.GetCellData().RemoveArray("p")

    return wallPatch

def _wss_over_time(foam_case: str,
                   density=_density,
                   field=_foamWSS,
                   patch=_wallPatch) -> tuple:

    ofReader = vtk.vtkPOpenFOAMReader()

    # Apparently, it needs to be ran 2 times
    # to load the data
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
    timeSteps = [ofReader.GetTimeValues().GetValue(id_)
                 for id_ in range(nTimeSteps)]

    # Get WSS data
    wssVecOverTime = {}
    addInstantWss = wssVecOverTime.update

    # I will store one surface for reference here
    # to hold the final data
    baseSurface = None

    # for surface in map(readSurface, glob.glob(folder + '*.vtk')):
    for time in timeSteps:
        ofReader.UpdateTimeStep(time)
        surface = _get_wall_surface(ofReader.GetOutput())

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
    """Compute WSS time statistocs from OpenFOAM data.
    
    Get time statistics of wall shear stress field defined on 
    a surface S over time for a cardiac cycle, generated with
    OpenFOAM. Outputs a surface with: time-averaged WSS, 
    maximum and minimum over time, peak-systole and low-diastole
    WSS vector fields. Since this function use OpenFOAM data, 
    specify the density considered.

    Input args:
    - OpenFOAM case file (str): name of OpenFOAM .foam case;
    - wssFieldName (str, optional): string containing the name 
        of the wall shear stress field (default="wallShearComp-
        onent");
    - patchName (str, optional): patch name where to calculate 
        the OSI (default="wall");
    - blood density (float, optional): default 1056.0 kg/m3
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
    storeArray((temporal_wss[t_peak_systole], _PSWSS))
    storeArray((temporal_wss[t_low_diastole], _LDWSS))
    
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


def _GON(np_surface,
         temporal_wss, 
         p_hat_array,
         q_hat_array,
         time_steps):

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
        surfaceWithSGrad = geo.surfaceGradient(np_surface.VTKObject, _WSSDotP)
        surfaceWithSGrad = geo.surfaceGradient(surfaceWithSGrad, _WSSDotQ)

        tSurface = dsa.WrapDataObject(surfaceWithSGrad)

        # Now project each surface gradient on coordinate direction
        sGradDotPHat = _HadamardDot(tSurface.CellData.GetArray(_WSSDotP+_sgrad), p_hat_array)
        sGradDotQHat = _HadamardDot(tSurface.CellData.GetArray(_WSSDotQ+_sgrad), q_hat_array)

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
    setArray(avgGVecArray, _SWSSG)
    setArray(avgMagGVecArray, _SWSSGmag)

def hemodynamics(foam_case: str,
                 t_peak_systole: float,
                 t_low_diastole: float,
                 density=_density,  # kg/m3
                 field=_foamWSS,
                 patch=_wallPatch,
                 compute_gon=False,
                 compute_afi=False) -> _polyDataType:
    """Compute hemodynamics of WSS field.
    
    Based on the temporal statistics of the WSS field
    over a vascular and aneurysm surface, compute the 
    following parameters: oscillatory shear index (OSI),
    relative residance time (RRT), WSS pulsatility index
    (WSSPI), the time-averaged WSS gradient, TAWSSG,
    the average WSS direction vector, p, and 
    orthogonal, q, to p and the normal, n, to the 
    surface. The triad (p, q, n) is a suitable coordi-
    nate system defined on the vascular surface.
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
    surfaceWithNormals  = geo.surfaceNormals(surface)
    surfaceWithGradient = geo.surfaceGradient(surfaceWithNormals, _TAWSS)

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
        psWSS = getArray(_PSWSS)
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
        _GON(numpySurface, 
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
    
    for name, array in arraysToBeStored:
        setArray(array, name)
    
    return numpySurface.VTKObject

def wss_stats_aneurysm(neckSurface,
                       neckArrayName,
                       n_percentile,
                       neckIsoValue=0.5,
                       avgMagWSSArray=_TAWSS):
    """
        Computes surface-averaged and maximum value 
        of time-averaged WSS for an aneurysm surface.
        Input is a PolyData surface with the averaged
        fields and the aneurysm neck contour field. 
        Return list with aneurysm area, WSSav and 
        WSSmax.
    """

    try:
        # Try to read if file name is given
        surface = tools.readSurface(neckSurface)
    except:
        surface = neckSurface

    # Get aneurysm 
    aneurysm = tools.clipWithScalar(surface, neckArrayName, neckIsoValue)

    # Get Array
    array = dsa.WrapDataObject(aneurysm)

    wssArray = array.GetCellData().GetArray(avgMagWSSArray)

    # WSS averaged
    maximum = np.max(np.array(wssArray))
    minimum = np.min(np.array(wssArray))
    percentile = np.percentile(np.array(wssArray), n_percentile)
    average = np.average(np.array(wssArray))

    return [average, maximum, minimum, percentile]


def osi_stats_aneurysm(neckSurface,
                       neckArrayName,
                       n_percentile,
                       neckIsoValue=0.5,
                       osiArrayName=_OSI):
    """
        Computes surface-averaged and maximum value 
        of OSI for an aneurysm surface.
        Input is a PolyData surface with the averaged
        fields and the aneurysm neck contour field. 
        Return list with aneurysm area, OSIav, OSImax, and 
        OSImin.
    """
    try:
        # Try to read if file name is given
        surface = tools.readSurface(neckSurface)

    except:
        surface = neckSurface

    # Get aneurysm 
    aneurysm = tools.clipWithScalar(surface, neckArrayName, neckIsoValue)

    # Get Array
    array = dsa.WrapDataObject(aneurysm)

    osiArray = array.GetCellData().GetArray(osiArrayName)

    # WSS averaged
    maximum = np.max(np.array(osiArray))
    minimum = np.min(np.array(osiArray))
    percentile = np.percentile(np.array(osiArray), n_percentile)
    average = np.average(np.array(osiArray))

    return [average, maximum, minimum, percentile]

def lsa_wss_avg(neckSurface,
                neckArrayName,
                lowWSS,
                neckIsoValue=0.5,
                avgMagWSSArray=_TAWSS):
    """ 
    Calculates the LSA (low WSS area ratio) for aneurysms
    simulations performed in OpenFOAM. Thi input is a sur-
    face with the time-averaged WSS over the surface and 
    an array defined on it indicating the aneurysm neck.
    The function then calculates the aneurysm surface area
    and the area where the WSS is lower than a reference 
    value provided by the user.
    """
    try:
        # Try to read if file name is given
        surface = tools.readSurface(neckSurface)
    except:
        surface = neckSurface

    # Get aneurysm 
    aneurysm = tools.clipWithScalar(surface, neckArrayName, neckIsoValue)

    # Get aneurysm area
    aneurysmArea = geo.surfaceArea(aneurysm)

    # Get low shear area
    lsaPortion = tools.clipWithScalar(aneurysm, avgMagWSSArray, lowWSS)
    lsaArea = geo.surfaceArea(lsaPortion)

    return lsaArea/aneurysmArea


# TODO: the following functions still depends on ParaView
# Eliminate that!

# This calculation depends on the WSS defined only on the
# parent artery surface. I think the easiest way to com-
# pute that is by drawing the artery contour in the same
# way as the aneurysm neck is beuild. So, I will assume
# in this function that the surface is already cut to in-
# clude only the parent artery portion and that includes
def wss_parent_vessel(parentArterySurface,
                      parentArteryArrayName,
                      parentArteryIsoValue=0.5):
    """
        Calculates the surface averaged WSS value
        over the parent artery surface.
    """

    try:
        # Try to read if file name is given
        surface = pv.XMLPolyDataReader(FileName=parentArterySurface)
    except:
        surface = parentArterySurface

    clipParentArtery = pv.Clip()
    clipParentArtery.Input = surface
    clipParentArtery.ClipType = 'Scalar'
    clipParentArtery.Scalars = ['POINTS', parentArteryArrayName]
    clipParentArtery.Invert = 1                     # gets smaller portion
    # based on the definition of field ContourScalars
    clipParentArtery.Value = parentArteryIsoValue
    clipParentArtery.UpdatePipeline()

    # Finaly we integrate over Sa
    integrateOverArtery = pv.IntegrateVariables()
    integrateOverArtery.Input = clipParentArtery
    integrateOverArtery.UpdatePipeline()

    parentArteryArea = integrateOverArtery.CellData.GetArray(_Area).GetRange()[
        0]
    parentArteryWSS = integrateOverArtery.CellData.GetArray(
        _WSSmag+'_average').GetRange()[0]

    # Delete pv objects
    pv.Delete(clipParentArtery)
    del clipParentArtery

    pv.Delete(integrateOverArtery)
    del integrateOverArtery

    return parentArteryWSS/parentArteryArea


def wss_surf_avg(foamCase,
                 neckSurface=None,
                 neckArrayName=None,
                 neckIsoValue=0.5,
                 density=1056.0,
                 field=_foamWSS,
                 patch=_wallPatch):
    """
        Function to compute surface integrals of WSS over 
        an aneurysm or vessels surface. It takes the Open-
        FOAM case file and an optional surface where it is 
        stored a field with the aneurysm neck line loaded 
        as a ParaView PolyData surface. If the surface is
        None, it computes the integral over the entire sur-
        face. It is essential that the surface with the ne-
        ck array be the same as the wall surface of the 
        OpenFOAM case, i.e. they are the same mesh.
    """

    try:
        # Try to read if file name is given
        ofData = pv.OpenFOAMReader(FileName=foamCase)
    except:
        ofData = foamCase

    # Read OpenFOAM data and process the WSS
    # to get its magnitude
    ofData.CellArrays = [field]
    ofData.MeshRegions = [patch]
    ofData.SkipZeroTime = 1
    ofData.Createcelltopointfiltereddata = 0

    # Get time-steps values
    timeSteps = np.array(ofData.TimestepValues)
    ofData.UpdatePipeline()

    # Triangulate data to coincide with
    triangulate = pv.Triangulate()
    triangulate.Input = ofData
    triangulate.UpdatePipeline()

    # Compute magnitude of WSS in each cell of the aneurysm surface
    magWSS = pv.Calculator()
    magWSS.Input = triangulate
    magWSS.Function = str(density) + '*mag(' + field + ')'
    magWSS.AttributeType = 'Cell Data'
    magWSS.ResultArrayName = _WSSmag
    magWSS.UpdatePipeline()

    extractSurface = pv.ExtractSurface()
    extractSurface.Input = magWSS
    extractSurface.UpdatePipeline()

    # Delete pv objects
    pv.Delete(triangulate)
    del triangulate

    pv.Delete(ofData)
    del ofData

    if neckSurface and neckArrayName is not None:
        try:
            # Try to read if file name is given
            surface = pv.XMLPolyDataReader(FileName=neckSurface)
        except:
            surface = neckSurface

        # Clip original aneurysm surface in the neck line
        clipAneurysm = pv.Clip()
        clipAneurysm.Input = surface
        clipAneurysm.ClipType = 'Scalar'
        clipAneurysm.Scalars = ['POINTS', neckArrayName]
        clipAneurysm.Invert = 1
        # based on the definition of field ContourScalars
        clipAneurysm.Value = neckIsoValue
        clipAneurysm.UpdatePipeline()

        # Resample OpenFOAM data to clipped aneeurysm surface
        resample = pv.ResampleWithDataset()
        # resample.Input = magWSS
        resample.SourceDataArrays = magWSS
        # resample.Source = clipAneurysm
        resample.DestinationMesh = clipAneurysm
        resample.PassCellArrays = 1
        resample.UpdatePipeline()

        # Since all fields in ResampleWithDataSet filter
        # are interpolated to points, therefore
        # apply point data to cell data filter
        pointToCell = pv.PointDatatoCellData()
        pointToCell.Input = resample
        pointToCell.UpdatePipeline()

        extractSurface = pv.ExtractSurface()
        extractSurface.Input = pointToCell
        extractSurface.UpdatePipeline()

        pv.Delete(clipAneurysm)
        del clipAneurysm

        pv.Delete(resample)
        del resample

    # Delete pv objects
    pv.Delete(magWSS)
    del magWSS

    surfAvgWSSList = []

    for timeStep in timeSteps:
        # Integrate WSS on the wall
        integrate = pv.IntegrateVariables()
        integrate.Input = extractSurface
        integrate.UpdatePipeline(time=timeStep)

        # Get area of surface, in m2
        area = integrate.CellData.GetArray(_Area).GetRange()[0]
        integralWSS = integrate.CellData.GetArray(_WSSmag).Name

        # Instantiate calculater filter
        surfAvgWSS = pv.Calculator()
        surfAvgWSS.Input = integrate
        surfAvgWSS.Function = integralWSS + '/' + str(area)
        surfAvgWSS.ResultArrayName = _WSS_surf_avg
        surfAvgWSS.AttributeType = 'Cell Data'
        surfAvgWSS.UpdatePipeline(time=timeStep)

        averagedWSS = surfAvgWSS.CellData.GetArray(_WSS_surf_avg).GetRange()[0]
        surfAvgWSSList.append(averagedWSS)

    return surfAvgWSSList

def lsa_instant(foamCase,
                neckSurface,
                neckArrayName,
                lowWSS,
                neckIsoValue=0.5,
                density=1056.0,
                field=_foamWSS,
                patch=_wallPatch):
    """ 
    Calculates the LSA (low WSS area ratio) for aneurysms
    simulations performed in OpenFOAM. The input is a sur-
    face with the time-averaged WSS over the surface an
    OpenFOAM case with the WSS field and a surface which
    contains the array with the aneurysm neck iso line.
    The function then calculates the aneurysm surface area
    and the area where the WSS is lower than a reference 
    value provided by the user, for each instant in the 
    cycles simulated, returning a list with the LSA values
    over time, for the last cycle.
    """

    try:
        # Try to read if file name is given
        surface = pv.XMLPolyDataReader(FileName=neckSurface)
    except:
        surface = neckSurface

    # Clip original aneurysm surface in the neck line
    clipAneurysm = pv.Clip()
    clipAneurysm.Input = surface
    clipAneurysm.ClipType = 'Scalar'
    clipAneurysm.Scalars = ['POINTS', neckArrayName]
    clipAneurysm.Invert = 1
    # based on the definition of field ContourScalars
    clipAneurysm.Value = neckIsoValue
    clipAneurysm.UpdatePipeline()

    integrateWSS = pv.IntegrateVariables()
    integrateWSS.Input = clipAneurysm
    integrateWSS.UpdatePipeline()

    # Get area of surface, in m2
    aneurysmArea = integrateWSS.CellData.GetArray(_Area).GetRange()[0]

    # Read openfoam data
    try:
        # Try to read if file name is given
        ofData = pv.OpenFOAMReader(FileName=foamCase)
    except:
        ofData = foamCase

    ofData.CellArrays = [field]
    ofData.MeshRegions = [patch]
    ofData.SkipZeroTime = 1
    ofData.Createcelltopointfiltereddata = 0
    ofData.UpdatePipeline()

    # Get time-steps
    timeSteps = np.array(ofData.TimestepValues)

    # Triangulate data to coincide with time averaged surface
    # Error prone, the triangulation must be the same
    triangulate = pv.Triangulate()
    triangulate.Input = ofData
    triangulate.UpdatePipeline()

    # Compute magnitude of WSS in each cell of the aneurysm surface
    magWSS = pv.Calculator()
    magWSS.Input = triangulate
    magWSS.Function = str(density)+'*mag('+field+')'
    magWSS.ResultArrayName = _WSSmag
    magWSS.AttributeType = 'Cell Data'
    magWSS.UpdatePipeline()

    # Resample OpenFOAM data to clipped aneeurysm surface
    resample = pv.ResampleWithDataset()
    # resample.Input = magWSS
    resample.SourceDataArrays = magWSS
    # resample.Source = clipAneurysm
    resample.DestinationMesh = clipAneurysm
    resample.PassCellArrays = 1
    resample.UpdatePipeline()

    # Clip the aneurysm surface in the lowWSS
    # anD gets portion smaller than it
    clipLSA = pv.Clip()
    clipLSA.Input = resample
    clipLSA.Value = lowWSS
    clipLSA.ClipType = 'Scalar'
    clipLSA.Scalars = ['POINTS', _WSSmag]
    clipLSA.Invert = 1   # gets portion smaller than the value
    clipLSA.UpdatePipeline()

    # Delete objects
    pv.Delete(ofData)
    del ofData

    pv.Delete(triangulate)
    del triangulate

    pv.Delete(magWSS)
    del magWSS

    pv.Delete(resample)
    del resample

    LSAt = []
    for instant in timeSteps:

        # Integrate to get area of lowWSSValue
        integrateOverLSA = pv.IntegrateVariables()
        integrateOverLSA.Input = clipLSA
        integrateOverLSA.UpdatePipeline(time=instant)

        area = integrateOverLSA.CellData.GetArray(_Area)
        if area == None:
            lsaArea = 0.0
        else:
            lsaArea = integrateOverLSA.CellData.GetArray(_Area).GetRange()[0]

        LSAt.append(lsaArea/aneurysmArea)

    # Delete objects
    pv.Delete(clipLSA)
    del clipLSA

    pv.Delete(integrateOverLSA)
    del integrateOverLSA

    return LSAt

def afi(foamCase,
        timeIndexRange,
        instant,
        outputFile,
        density=1056.0,  # kg/m3
        field=_foamWSS,
        patch=_wallPatch):

    try:
        # Try to read if file name is given
        ofData = pv.OpenFOAMReader(FileName=foamCase)
    except:
        ofData = foamCase

    # First we define only the field that are going
    # to be used: the WSS on the aneurysm wall
    ofData.CellArrays = [field]
    ofData.MeshRegions = [patch]
    ofData.Createcelltopointfiltereddata = 0
    ofData.SkipZeroTime = 1
    ofData.UpdatePipeline()

    mergeBlocks = pv.MergeBlocks()
    mergeBlocks.Input = ofData
    mergeBlocks.UpdatePipeline()

    extractSurface = pv.ExtractSurface()
    extractSurface.Input = mergeBlocks
    extractSurface.UpdatePipeline()

    triangulate = pv.Triangulate()
    triangulate.Input = extractSurface
    triangulate.UpdatePipeline()

    # Multiplying WSS per density
    calcWSS = pv.Calculator()
    calcWSS.Input = triangulate
    calcWSS.AttributeType = 'Cell Data'
    calcWSS.ResultArrayName = _WSS
    calcWSS.Function = str(density) + '*' + field
    calcWSS.UpdatePipeline()

    # Calculating the magnitude of the wss vector
    calcMagWSS = pv.Calculator()
    calcMagWSS.Input = calcWSS
    calcMagWSS.AttributeType = 'Cell Data'
    calcMagWSS.ResultArrayName = _WSSmag
    calcMagWSS.Function = 'mag(' + _WSS + ')'
    calcMagWSS.UpdatePipeline()

    timeData = pv.ExtractSurface()
    timeData.Input = calcMagWSS
    timeData.UpdatePipeline()

    # Delete objects
    pv.Delete(ofData)
    del ofData

    pv.Delete(mergeBlocks)
    del mergeBlocks

    pv.Delete(extractSurface)
    del extractSurface

    pv.Delete(calcWSS)
    del calcWSS

    # Extract desired time range
    timeInterval = pv.ExtractTimeSteps()
    timeInterval.Input = calcMagWSS
    timeInterval.SelectionMode = 'Select Time Range'
    timeInterval.TimeStepRange = timeIndexRange
    timeInterval.UpdatePipeline()

    # Now compute the temporal statistics
    # filter computes the average values of all fields
    calcTimeStats = pv.TemporalStatistics()
    calcTimeStats.Input = timeInterval
    calcTimeStats.ComputeAverage = 1
    calcTimeStats.ComputeMinimum = 0
    calcTimeStats.ComputeMaximum = 0
    calcTimeStats.ComputeStandardDeviation = 0
    calcTimeStats.UpdatePipeline()

    timeStats = pv.ExtractSurface()
    timeStats.Input = calcTimeStats
    timeStats.UpdatePipeline()

    # Resample OpenFOAM data to clipped aneeurysm surface
    resample = pv.ResampleWithDataset()
    # resample.Input = timeStats
    resample.SourceDataArrays = timeStats
    resample.DestinationMesh = timeData
    resample.PassCellArrays = 1
    resample.PassCellArrays = 1
    resample.UpdatePipeline()

    pointToCell = pv.PointDatatoCellData()
    pointToCell.Input = resample
    pointToCell.UpdatePipeline()

    # Calculates OSI
    calcAFI = pv.Calculator()
    calcAFI.Input = pointToCell
    calcAFI.AttributeType = 'Cell Data'
    calcAFI.ResultArrayName = _AFI

    # AFI calculation
    # AFI = (vecWSSt dot meanVecWSS)/(magWSSt dot magMeanVecWSS)
    calcAFI.Function = '(' + _WSS+'_X * '+_WSS+'_average_X +' + \
        _WSS+'_Y * '+_WSS+'_average_Y +' + \
        _WSS+'_Z * '+_WSS+'_average_Z)/' + \
        '(mag('+_WSS+') * mag('+_WSS+'_average))'

    calcAFI.UpdatePipeline(time=instant)

    pv.SaveData(outputFile, proxy=calcAFI)

    # Delete other fields
    # This is some sort of redudancy, but it was
    # the only way I found to isolate the resulting field
    surface = pv.XMLPolyDataReader(FileName=outputFile)
    surface.CellArrayStatus = [_WSS,
                               _WSSmag,
                               _WSS + '_average',
                               _AFI]

    pv.SaveData(outputFile, proxy=surface)

    # Delete objects
    pv.Delete(calcAFI)
    del calcAFI

    pv.Delete(timeInterval)
    del timeInterval

    pv.Delete(resample)
    del resample


if __name__ == '__main__':

    # Testing: computes hemodynamcis given an aneurysm OF case
    import sys
    import polydatatools as tools

    from vmtk import vmtkscripts
    from vmtkextend import customscripts

    foamCase = sys.argv[1]
    outFile = sys.argv[2]

    density = 1056.0
    peakSystoleTime = 2.09
    lowDiastoleTime = 2.80

    print("Computing hemodynamics", end='\n')
    hemodynamicsSurface = hemodynamics(foamCase, 
                                       peakSystoleTime, 
                                       lowDiastoleTime, 
                                       compute_gon=False, 
                                       compute_afi=False)

    scaling = vmtkscripts.vmtkSurfaceScaling()
    scaling.Surface = hemodynamicsSurface
    scaling.ScaleFactor = 1000.0
    scaling.Execute()

    extractAneurysm = customscripts.vmtkExtractAneurysm()
    extractAneurysm.Surface = scaling.Surface
    extractAneurysm.Execute()

    surface = extractAneurysm.Surface

    # project = vmtkscripts.vmtkSurfaceProjection()
    # project.Surface = scaling.Surface
    # project.ReferenceSurfaceInputFileName = '/home/iagolessa/surface_with_aneurysm_array.vtp'
    # project.IORead()
    # project.Execute()

    # surface = project.Surface

    # Computes WSS and OSI statistics
    neckArrayName = 'AneurysmNeckContourArray'
    print(wss_stats_aneurysm(surface, neckArrayName, 95), end='\n')
    print(osi_stats_aneurysm(surface, neckArrayName, 95), end='\n')
    print(lsa_wss_avg(surface, neckArrayName, 1.5), end='\n')

    # tools.writeSurface(scaling.Surface, outFile)
