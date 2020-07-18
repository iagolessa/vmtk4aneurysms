"""
Python library of functions to calculate morphological and hemodynamic 
parameters related to aneurysms geometry and hemodynamics using ParaView 
filters.

The library works with the paraview.simple module. 
"""

import paraview.simple as pv

import vtk
import numpy as np
from vtk.numpy_interface import dataset_adapter as dsa

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

# Local coordinate system
_pHat = 'pHat'
_qHat = 'qHat'
_normals = 'Normals'

# Other attributes
_foamWSS = 'wallShearComponent'
_wallPatch = 'wall'
_aneurysmArray = 'AneurysmNeckContourArray'
_parentArteryArray = 'ParentArteryArray'

# ParaView auxiliary variables
_xAxis = '_X'
_yAxis = '_Y'
_zAxis = '_Z'
_avg = '_average'
_min = '_minimum'
_max = '_maximum'
_grad = '_gradient'

_cellDataMode = 'Cell Data'
_pointDataMode = 'Point Data'


def wss_time_stats(foamCase,
                   timeIndexRange,
                   peakSystole,
                   lowDiastole,
                   timeStep=0.01,
                   density=1056.0,  # kg/m3
                   field=_foamWSS,
                   patch=_wallPatch):
    """Compute WSS time statistics: averaged, peak systole and low diastole.

    Get time statistics of wall shear stress field defined on 
    a surface S over time for a cardiac cycle, generated with
    OpenFOAM. Outputs a surface with: time-averaged and peak 
    time (peak systole) WSS magnitude.
    Since this function use OpenFOAM data, 
    please specify the density considered.

    Input args:
    - OpenFOAM case file (str): name of OpenFOAM .foam case;
    - wssFieldName (str, optional): string containing the name 
        of the wall shear stress field (default="wallShearComp-
        onent");
    - patchName (str, optional): patch name where to calculate 
        the OSI (default="wall");
    - timeIndexRange (list): list of initial and final time-
        steps indices limits of the integral [Ti, Tf];
    - outputFileName (str): file name for the output file with 
        osi field (must be a .vtp file).
    - blood density (float, optional): default 1056.0 kg/m3
    """
    ofData = pv.OpenFOAMReader(FileName=foamCase)

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

    # triangulate = pv.Triangulate()
    # triangulate.Input = extractSurface
    # triangulate.UpdatePipeline()

    # Multiplying WSS per density
    calcWSS = pv.Calculator()
    calcWSS.Input = extractSurface
    calcWSS.AttributeType = _cellDataMode
    calcWSS.ResultArrayName = _WSS
    calcWSS.Function = str(density) + '*' + field
    calcWSS.UpdatePipeline()

    # Calculating the magnitude of the wss vector
    calcMagWSS = pv.Calculator()
    calcMagWSS.Input = calcWSS
    calcMagWSS.AttributeType = _cellDataMode
    calcMagWSS.ResultArrayName = _WSSmag
    calcMagWSS.Function = 'mag(' + _WSS + ')'
    calcMagWSS.UpdatePipeline()

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
    calcAvgWSS = pv.TemporalStatistics()
    calcAvgWSS.Input = timeInterval
    calcAvgWSS.ComputeAverage = 1
    calcAvgWSS.ComputeMinimum = 1
    calcAvgWSS.ComputeMaximum = 1
    calcAvgWSS.ComputeStandardDeviation = 0
    calcAvgWSS.UpdatePipeline()

    # Change name of WSS_magnitude_average to TAWSS
    includeTAWSS = pv.Calculator()
    includeTAWSS.Input = calcAvgWSS
    includeTAWSS.Function = _WSSmag + _avg
    includeTAWSS.AttributeType = _cellDataMode
    includeTAWSS.ResultArrayName = _TAWSS
    includeTAWSS.UpdatePipeline()
    
    # Get peak systole WSS
    peakSystoleWSS = pv.Calculator()
    peakSystoleWSS.Input = calcMagWSS
    peakSystoleWSS.Function = _WSS
    peakSystoleWSS.AttributeType = _cellDataMode
    peakSystoleWSS.ResultArrayName = _PSWSS
    peakSystoleWSS.UpdatePipeline(time=peakSystole)

    # Get low diastole WSS
    lowDiastoleWSS = pv.Calculator()
    lowDiastoleWSS.Input = calcMagWSS
    lowDiastoleWSS.Function = _WSS
    lowDiastoleWSS.AttributeType = _cellDataMode
    lowDiastoleWSS.ResultArrayName = _LDWSS
    lowDiastoleWSS.UpdatePipeline(time=lowDiastole)

    merge = pv.AppendAttributes()
    merge.Input = [includeTAWSS, peakSystoleWSS, lowDiastoleWSS]
    merge.UpdatePipeline()

    extractSurface = pv.ExtractSurface()
    extractSurface.Input = merge
    extractSurface.UpdatePipeline()

    # Field clean up
    # TODO: find a better way to extract the vtkPolyData 
    # from paraview (in the mean time use this write and read)
    # Horrible
    outputFile = "/tmp/surface_wss_stats.vtp"
    pv.SaveData(outputFile, extractSurface)
    
    # The only I found so far: to write and read the surface!
    # Very ineficient
    reader = vtk.vtkXMLPolyDataReader()
    reader.SetFileName(outputFile)
    reader.Update()
    
    surface = reader.GetOutput()
    
    # Clean-up unnecessary arrays
    arraysForDeletion = [field, field+_avg, field+_max, field+_min, field+'_input_2', 
                         _WSSmag, _WSSmag+_avg, _WSSmag+'_input_2',
                         _WSS, _WSS+_min, _WSS+_max, _WSS+'_input_2']
    
    for array in arraysForDeletion:
        surface.GetCellData().RemoveArray(array)

    # Delete objects
    pv.Delete(calcAvgWSS)
    del calcAvgWSS
    
    pv.Delete(includeTAWSS)
    del includeTAWSS
    
    pv.Delete(timeInterval)
    del timeInterval

    pv.Delete(peakSystoleWSS)
    del peakSystoleWSS

    pv.Delete(lowDiastoleWSS)
    del lowDiastoleWSS
    
    return surface

def surfaceNormals(surface):
    """Compute outward surface normals."""
    
    normals = vtk.vtkPolyDataNormals()
    
    normals.ComputeCellNormalsOn()
    normals.ComputePointNormalsOff()
    # normals.AutoOrientNormalsOff()
    # normals.FlipNormalsOn()
    normals.SetInputData(surface)
    normals.Update()
    
    return normals.GetOutput()

def spatialGradient(surface, field_name):
    """Compute gradient of field on a surface."""
    
    gradient = vtk.vtkGradientFilter()
    gradient.SetInputData(surface)

    # TODO: Make check of type of field
    # scalar or vector
    # 1 is the field type: means vector
    gradient.SetInputScalars(1, field_name)
    gradient.SetResultArrayName(field_name+_grad)
    gradient.ComputeDivergenceOff()
    gradient.ComputeGradientOn()
    gradient.ComputeQCriterionOff()
    gradient.ComputeVorticityOff()
    gradient.Update()

    return gradient.GetOutput()

def HadamardDot(np_array1, np_array2):
    """Computes dot product in a Hadamard product way.
    
    Given two Numpy arrays representing arrays of vectors
    on a surface, compute the vector-wise dot product 
    between each element.
    """
    # Seems that multiply is faster than a*b
    return np.multiply(np_array1, np_array2).sum(axis=1)

def hemodynamics(surface):
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
    
    # Compute normals and gradient of TAWSS
    surfaceWithNormals  = surfaceNormals(surface)
    surfaceWithGradient = spatialGradient(surfaceWithNormals, _TAWSS)

    # Convert VTK polydata to numpy object
    numpySurface = dsa.WrapDataObject(surfaceWithGradient)
    
    # Functions to interface with the numpy vtk wrapper
    getArray = numpySurface.GetCellData().GetArray
    appendToSurface = numpySurface.CellData.append
    
    # Get arrays currently on the surface
    # that will be used for the calculations
    avgVecWSSArray = getArray(_WSS + _avg)
    avgMagWSSArray = getArray(_TAWSS)
    maxMagWSSArray = getArray(_WSSmag + _max)
    minMagWSSArray = getArray(_WSSmag + _min)
    normalsArray   = getArray(_normals)
    gradientArray  = getArray(_TAWSS + _grad)

    # Compute the magnitude of the WSS vector time average
    magAvgVecWSSArray = np.linalg.norm(avgVecWSSArray, ord=2, axis=1)
    
    # Compute WSS derived quantities
    # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
    
    WSSPI = (maxMagWSSArray - minMagWSSArray)/avgMagWSSArray

    OSI = 0.5*(1 - magAvgVecWSSArray/avgMagWSSArray)
    
    RRT = 1.0/((1.0 - 2.0*OSI)*avgMagWSSArray)
    
    # Calc surface orthogonal vectors
    # -> p: timeAvg WSS vector
    pHat = avgVecWSSArray/avgMagWSSArray

    # -> q: perpendicular to p and normal
    qHat = np.cross(pHat, normalsArray)

    # Compute the normal gradient = vec(n) dot grad(TAWSS)
    normalGradient = HadamardDot(normalsArray, gradientArray)

    # Compute the surface gradient
    surfaceGradTAWSS = gradientArray - normalGradient*normalsArray

    # Compute the TAWSSG = surfaceGradTAWSS dot p
    TAWSSG = HadamardDot(pHat, surfaceGradTAWSS)

    # AFI at peak-systole
    psWSS = getArray(_PSWSS)
    psWSSmag = np.linalg.norm(psWSS, axis=1)
    
    # Compute AFI at peak systole
    psAFI = HadamardDot(pHat, psWSS)/psWSSmag

    # Store new fields in surface
    appendToSurface(OSI, _OSI)
    appendToSurface(RRT, _RRT)
    appendToSurface(pHat, _pHat)
    appendToSurface(qHat, _qHat)
    appendToSurface(WSSPI, _WSSPI)
    appendToSurface(TAWSSG, _TAWSSG)
    appendToSurface(psAFI, _AFI + '_peak_systole')
    
    return numpySurface.VTKObject

def wss_stats_aneurysm(neckSurface,
                       neckArrayName,
                       n_percentile,
                       neckIsoValue=0.5,
                       avgMagWSSArray=_WSSmag+'_average'):
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
        reader = vtk.vtkXMLPolyDataReader()
        reader.SetFileName(neckSurface)
        reader.Update()

        surface = reader.GetOutput()
    except:
        surface = neckSurface

    surface.GetPointData().SetActiveScalars(neckArrayName)

    # Clip aneurysm surface
    clipper = vtk.vtkClipPolyData()
    clipper.SetInputData(surface)
    clipper.SetValue(neckIsoValue)
    clipper.SetInsideOut(1)
    clipper.Update()

    aneurysm = clipper.GetOutput()

    # Get aneurysm area
    aneurysmProperties = vtk.vtkMassProperties()
    aneurysmProperties.SetInputData(aneurysm)
    aneurysmProperties.Update()

    surfaceArea = aneurysmProperties.GetSurfaceArea()

    # Get Array
    array = dsa.WrapDataObject(aneurysm)

    wssArray = array.GetCellData().GetArray(avgMagWSSArray)

    # WSS averaged
    maximum = np.max(np.array(wssArray))
    minimum = np.min(np.array(wssArray))
    percentile = np.percentile(np.array(wssArray), n_percentile)
    average = np.average(np.array(wssArray))

    return [surfaceArea, average, maximum, minimum, percentile]


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
        reader = vtk.vtkXMLPolyDataReader()
        reader.SetFileName(neckSurface)
        reader.Update()

        surface = reader.GetOutput()
    except:
        surface = neckSurface

    surface.GetPointData().SetActiveScalars(neckArrayName)

    # Clip aneurysm surface
    clipper = vtk.vtkClipPolyData()
    clipper.SetInputData(surface)
    clipper.SetValue(neckIsoValue)
    clipper.SetInsideOut(1)
    clipper.Update()

    aneurysm = clipper.GetOutput()

    # Get aneurysm area
    aneurysmProperties = vtk.vtkMassProperties()
    aneurysmProperties.SetInputData(aneurysm)
    aneurysmProperties.Update()

    surfaceArea = aneurysmProperties.GetSurfaceArea()

    # Get Array
    array = dsa.WrapDataObject(aneurysm)

    osiArray = array.GetCellData().GetArray(osiArrayName)

    # WSS averaged
    maximum = np.max(np.array(osiArray))
    minimum = np.min(np.array(osiArray))
    percentile = np.percentile(np.array(osiArray), n_percentile)
    average = np.average(np.array(osiArray))

    return [surfaceArea, average, maximum, minimum, percentile]

def lsa_wss_avg(neckSurface,
                neckArrayName,
                lowWSS,
                neckIsoValue=0.5,
                avgMagWSSArray=_WSSmag+'_average'):
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
        surface = pv.XMLPolyDataReader(FileName=neckSurface)
    except:
        surface = neckSurface

    # Clip aneurysm surface
    clipAneurysm = pv.Clip()
    clipAneurysm.Input = surface
    clipAneurysm.ClipType = 'Scalar'
    clipAneurysm.Scalars = ['POINTS', neckArrayName]
    clipAneurysm.Invert = 1             # gets smaller portion
    # based on the definition of field ContourScalars
    clipAneurysm.Value = neckIsoValue
    clipAneurysm.UpdatePipeline()

    # Integrate to get aneurysm surface area
    integrateOverAneurysm = pv.IntegrateVariables()
    integrateOverAneurysm.Input = clipAneurysm
    integrateOverAneurysm.UpdatePipeline()

    aneurysmArea = integrateOverAneurysm.CellData.GetArray(_Area).GetRange()[
        0]  # m2

    # Clip the aneurysm surface in the lowWSSValue
    # ang gets portion smaller than it
    clipLSA = pv.Clip()
    clipLSA.Input = clipAneurysm
    clipLSA.ClipType = 'Scalar'
    clipLSA.Scalars = ['CELLS', avgMagWSSArray]
    clipLSA.Invert = 1   # gets portion smaller than the value
    clipLSA.Value = lowWSS
    clipLSA.UpdatePipeline()

    # Integrate to get area of lowWSSValue
    integrateOverLSA = pv.IntegrateVariables()
    integrateOverLSA.Input = clipLSA
    integrateOverLSA.UpdatePipeline()

    area = integrateOverLSA.CellData.GetArray(_Area)
    if area == None:
        lsaArea = 0.0
    else:
        lsaArea = integrateOverLSA.CellData.GetArray(_Area).GetRange()[0]

    # Delete pv objects
    pv.Delete(clipAneurysm)
    del clipAneurysm

    pv.Delete(integrateOverAneurysm)
    del integrateOverAneurysm

    pv.Delete(clipLSA)
    del clipLSA

    pv.Delete(integrateOverLSA)
    del integrateOverLSA

    return lsaArea/aneurysmArea


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

    foamCase = sys.argv[1]
    outFile = sys.argv[2]

    density = 1056.0
    timeIndexRange = [0,100]
    timeStep = 0.01
    peakSystoleTime = 2.09
    lowDiastoleTime = 2.90

    # Get temporal statistics of WSS field
    print("Computing WSS stats", end='\n')
    timeStatsSurface = wss_time_stats(
                           foamCase, 
                           timeIndexRange, 
                           peakSystoleTime, 
                           lowDiastoleTime
                       )

    print("Computing hemodynamics", end='\n')
    hemodynamicsSurface = hemodynamics(timeStatsSurface)

    #     extractAneurysm = customscripts.vmtkExtractAneurysm()
                # extractAneurysm.Surface = self._surface
                # extractAneurysm.Execute()

                # aneurysm_surface = extractAneurysm.AneurysmSurface

    # # Computes WSS and OSI statistics
    # wss_stats_aneurysm(hemodynamicsSurface, neckArrayName, 95)
    # osi_stats_aneurysm(fileName, neckArrayName, 95)

    writer = vtk.vtkXMLPolyDataWriter()
    writer.SetInputData(hemodynamicsSurface)
    writer.SetFileName(outFile)
    writer.Update()
