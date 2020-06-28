"""
Python library of functions to calculate morphological and hemodynamic 
parameters related to aneurysms geometry and hemodynamics using ParaView 
filters.

The library works with the paraview.simple module. 
"""

import sys
import numpy as np
import paraview.simple as pv

# Attribute array names
_Area = 'Area'
_WSS = 'WSS'
_OSI = 'OSI'
_RRT = 'RRT'
_AFI = 'AFI'
_WSSPI = 'WSSPI'
_TAWSSG = 'TAWSSG'
_WSSpst = 'WSS_peak_systole'
_WSSldt = 'WSS_low_diastole'
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
_aneurysmArray = 'AneurysmNeckArray'
_parentArteryArray = 'ParentArteryArray'

# ParaView auxiliary variables
_xAxis = '_X'
_yAxis = '_Y'
_zAxis = '_Z'
_avg = '_average'
_min = '_minimum'
_max = '_maximum'

_cellDataMode = 'Cell Data'
_pointDataMode = 'Point Data'


def aneurysm_area(neckSurface,
                  neckArrayName,
                  neckIsoValue=0.5):
    """ Compute aneurysm surface area """

    try:
        # Try to read if file name is given
        surface = pv.XMLPolyDataReader(FileName=neckSurface)
    except:
        surface = neckSurface

    clipAneurysm = pv.Clip()
    clipAneurysm.Input = surface
    clipAneurysm.ClipType = 'Scalar'
    clipAneurysm.Scalars = ['POINTS', neckArrayName]
    clipAneurysm.Invert = 1
    # based on the definition of field ContourScalars
    clipAneurysm.Value = neckIsoValue
    clipAneurysm.UpdatePipeline()

    # Finaly we integrate over Sa
    integrate = pv.IntegrateVariables()
    integrate.Input = clipAneurysmNeck
    integrate.UpdatePipeline()

    return integrate.CellData.GetArray(_Area).GetRange()[0]  # aneurysm area


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


def wss_time_stats(foamCase,
                   timeIndexRange,
                   peakSystole,
                   lowDiastole,
                   outputFile,
                   timeStep=0.01,
                   density=1056.0,  # kg/m3
                   field=_foamWSS,
                   patch=_wallPatch):
    """ 
        Get time statistics of wall shear stress field defined on 
        a surface S over time for a cardiac cycle, generated with
        OpenFOAM. Outputs a surface with: time-averaged and peak 
        time (peak systole) WSS magnitude, with also the surface
        field parameters that depends on this (such as OSI and RRT,
        if desired). The OSI field is defined as:

            OSI = 0.5*( 1 - norm2(int WSS dt) / int norm2(WSS) dt)

        where "int" implies the integral over time between two
        instants t1 and t2 (usually for a cardiac cycle, therefore 
        [t1, t2] = [Ti, Tf]) and norm2 is the L2 norm of an Eucli-
        dean vector field; WSS is the wall shear stress defined on 
        the input surface. Since this function use OpenFOAM data, 
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

    # Period given by time-steps
    period = (timeInterval.TimeStepRange[1] -
              timeInterval.TimeStepRange[0])*timeStep

    # Now compute the temporal statistics
    # filter computes the average values of all fields
    calcAvgWSS = pv.TemporalStatistics()
    calcAvgWSS.Input = timeInterval
    calcAvgWSS.ComputeAverage = 1
    calcAvgWSS.ComputeMinimum = 1
    calcAvgWSS.ComputeMaximum = 1
    calcAvgWSS.ComputeStandardDeviation = 0
    calcAvgWSS.UpdatePipeline()

    # Getting fields:
    # - Get the average of the vector WSS field
    avgVecWSS = calcAvgWSS.CellData.GetArray(_WSS + _avg).GetName()

    # - Get the stats of the magnitude of the WSS field
    avgMagWSS = calcAvgWSS.CellData.GetArray(
        calcMagWSS.ResultArrayName + _avg).GetName()
    minMagWSS = calcAvgWSS.CellData.GetArray(
        calcMagWSS.ResultArrayName + _min).GetName()
    maxMagWSS = calcAvgWSS.CellData.GetArray(
        calcMagWSS.ResultArrayName + _max).GetName()

    # Calculates WSSPI
    calcWSSPI = pv.Calculator()
    calcWSSPI.Input = calcAvgWSS
    calcWSSPI.AttributeType = _cellDataMode
    calcWSSPI.ResultArrayName = _WSSPI
    calcWSSPI.Function = '(' + maxMagWSS + '-' + minMagWSS + ')/' + avgMagWSS
    calcWSSPI.UpdatePipeline()

    # Calculates OSI
    calcOSI = pv.Calculator()
    calcOSI.Input = calcWSSPI
    calcOSI.AttributeType = _cellDataMode
    calcOSI.ResultArrayName = _OSI
    calcOSI.Function = '0.5*( 1 - ( mag( ' + \
        avgVecWSS + ') )/' + avgMagWSS + ')'
    calcOSI.UpdatePipeline()

    # Compute Relative Residance Time
    calcRRT = pv.Calculator()
    calcRRT.Input = calcOSI
    calcRRT.Function = str(period) + '/mag(' + avgVecWSS + ')'
    calcRRT.AttributeType = _cellDataMode
    calcRRT.ResultArrayName = _RRT
    calcRRT.UpdatePipeline()

    # Compute surface normals
    normals = pv.GenerateSurfaceNormals()
    normals.Input = calcRRT
    normals.ComputeCellNormals = True
    normals.UpdatePipeline()

    # Calculate local flow surface coordinate system
    calcPHat = pv.Calculator()
    calcPHat.Input = normals
    calcPHat.AttributeType = _cellDataMode
    calcPHat.ResultArrayName = _pHat
    calcPHat.Function = avgVecWSS+'/mag(' + avgVecWSS+')'
    calcPHat.UpdatePipeline()

    calcQHat = pv.Calculator()
    calcQHat.Input = calcPHat
    calcQHat.AttributeType = _cellDataMode
    calcQHat.ResultArrayName = _qHat
    calcQHat.Function = '(' + _pHat+_yAxis + '*' + _normals+_zAxis + ' - ' + _pHat+_zAxis + '*' + _normals+_yAxis + ')*iHat +' + \
                        '(' + _pHat+_zAxis + '*' + _normals+_xAxis + ' - ' + _pHat+_xAxis + '*' + _normals+_zAxis + ')*jHat +' + \
                        '(' + _pHat+_xAxis + '*' + _normals+_yAxis + ' - ' + \
        _pHat+_yAxis + '*' + _normals+_xAxis + ')*kHat'

    calcQHat.UpdatePipeline()

    # Computing spatial gradient
    gradient = pv.GradientOfUnstructuredDataSet()
    gradient.Input = calcQHat
    gradient.ScalarArray = _WSSmag + _avg
    gradient.ResultArrayName = _WSSGrad
    gradient.UpdatePipeline()

    # Project gradient to normal
    projGradToNormalName = 'nDotWSSGradient'
    surfaceGradientName = 'surfaceWSSGradient'

    nDotWSSGradient = pv.Calculator()
    nDotWSSGradient.Input = gradient
    nDotWSSGradient.AttributeType = _cellDataMode
    nDotWSSGradient.ResultArrayName = projGradToNormalName
    nDotWSSGradient.Function = _normals+_xAxis + '*' + _WSSGrad+_xAxis + '+' + \
        _normals+_yAxis + '*' + _WSSGrad+_yAxis + '+' + \
        _normals+_zAxis + '*' + _WSSGrad+_zAxis

    nDotWSSGradient.UpdatePipeline()

    # Compute surface gradient
    surfaceGrad = pv.Calculator()
    surfaceGrad.Input = nDotWSSGradient
    surfaceGrad.AttributeType = _cellDataMode
    surfaceGrad.ResultArrayName = surfaceGradientName
    surfaceGrad.Function = _WSSGrad + \
        '-(' + projGradToNormalName + '*' + _normals + ')'
    surfaceGrad.UpdatePipeline()

    calcTAWSSG = pv.Calculator()
    calcTAWSSG.Input = surfaceGrad
    calcTAWSSG.AttributeType = _cellDataMode
    calcTAWSSG.ResultArrayName = _TAWSSG
    calcTAWSSG.Function = surfaceGradientName+_xAxis + '*' + _pHat+_xAxis + '+' + \
        surfaceGradientName+_yAxis + '*' + _pHat+_yAxis + '+' + \
        surfaceGradientName+_zAxis + '*' + _pHat+_zAxis
    calcTAWSSG.UpdatePipeline()

    # Get peak systole WSS
    peakSystoleWSS = pv.Calculator()
    peakSystoleWSS.Input = calcMagWSS
    peakSystoleWSS.Function = _WSS
    peakSystoleWSS.AttributeType = _cellDataMode
    peakSystoleWSS.ResultArrayName = _WSSpst
    peakSystoleWSS.UpdatePipeline(time=peakSystole)

    # Get low diastole WSS
    lowDiastoleWSS = pv.Calculator()
    lowDiastoleWSS.Input = calcMagWSS
    lowDiastoleWSS.Function = _WSS
    lowDiastoleWSS.AttributeType = _cellDataMode
    lowDiastoleWSS.ResultArrayName = _WSSldt
    lowDiastoleWSS.UpdatePipeline(time=lowDiastole)

    merge = pv.AppendAttributes()
    merge.Input = [calcTAWSSG, peakSystoleWSS, lowDiastoleWSS]
    merge.UpdatePipeline()

    pv.SaveData(outputFile, merge)

    # Delete other fields
    # This is some sort of redudancy, but it was
    # the only way I found to isolate the resulting field
    surface = pv.XMLPolyDataReader(FileName=outputFile)
    surface.CellArrayStatus = [_OSI,
                               _RRT,
                               _WSSPI,
                               _TAWSSG,
                               _pHat,
                               _qHat,
                               _normals,
                               _WSSpst,
                               _WSSldt,
                               _WSS+_avg,
                               _WSSmag+_avg]

    pv.SaveData(outputFile, proxy=surface)

    # Delete objects
    pv.Delete(calcAvgWSS)
    del calcAvgWSS

    pv.Delete(timeInterval)
    del timeInterval

    pv.Delete(calcOSI)
    del calcOSI

    pv.Delete(calcRRT)
    del calcRRT

    pv.Delete(peakSystoleWSS)
    del peakSystoleWSS

    pv.Delete(lowDiastoleWSS)
    del lowDiastoleWSS


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


def wss_stats_aneurysm(neckSurface,
                       neckArrayName,
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
        surface = pv.XMLPolyDataReader(FileName=neckSurface)
    except:
        surface = neckSurface

    clipAneurysm = pv.Clip()
    clipAneurysm.Input = surface
    clipAneurysm.ClipType = 'Scalar'
    clipAneurysm.Scalars = ['POINTS', neckArrayName]
    # gets portion outside the clip function (values smaller than the clip value)
    clipAneurysm.Invert = 1
    clipAneurysm.Value = neckIsoValue
    clipAneurysm.UpdatePipeline()

    # Finaly we integrate over Sa
    integrate = pv.IntegrateVariables()
    integrate.Input = clipAneurysm
    integrate.UpdatePipeline()

    aneurysmArea = integrate.CellData.GetArray(_Area).GetRange()[0]

    WSSav = integrate.CellData.GetArray(avgMagWSSArray).GetRange()[
        0]/aneurysmArea  # averaged
    WSSmax = clipAneurysm.CellData.GetArray(avgMagWSSArray).GetRange()[
        1]  # maximum value
    WSSmin = clipAneurysm.CellData.GetArray(avgMagWSSArray).GetRange()[
        0]  # minimum value

    # Delete pv objects
    pv.Delete(clipAneurysm)
    del clipAneurysm

    pv.Delete(integrate)
    del integrate

    return [aneurysmArea, WSSav, WSSmax, WSSmin]


def osi_stats_aneurysm(neckSurface,
                       neckArrayName,
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
        surface = pv.XMLPolyDataReader(FileName=neckSurface)
    except:
        surface = neckSurface

    clipAneurysm = pv.Clip()
    clipAneurysm.Input = surface
    clipAneurysm.ClipType = 'Scalar'
    clipAneurysm.Scalars = ['POINTS', neckArrayName]
    clipAneurysm.Invert = 1
    # based on the definition of field ContourScalars
    clipAneurysm.Value = neckIsoValue
    clipAneurysm.UpdatePipeline()

    # Finaly we integrate over Sa
    integrate = pv.IntegrateVariables()
    integrate.Input = clipAneurysm
    integrate.UpdatePipeline()

    aneurysmArea = integrate.CellData.GetArray(_Area).GetRange()[0]

    OSIav = integrate.CellData.GetArray(osiArrayName).GetRange()[
        0]/aneurysmArea  # averaged
    OSImax = clipAneurysm.CellData.GetArray(osiArrayName).GetRange()[
        1]  # maximum value
    OSImin = clipAneurysm.CellData.GetArray(osiArrayName).GetRange()[
        0]  # minimum value

    # Delete pv objects
    pv.Delete(clipAneurysm)
    del clipAneurysm

    pv.Delete(integrate)
    del integrate

    return [aneurysmArea, OSIav, OSImax, OSImin]


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
# parent artery surface. I thimk the easiest way to com-
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
    if sys.argv[1] == 'osi':
        foamFile = sys.argv[3]
        timeRange = [sys.argv[4], sys.argv[5]]
        outputFile = sys.argv[6]
        foamCasePath = sys.argv[7]

        osi(foamCasePath+foamFile, timeRange, foamCasePath+outputFile)
