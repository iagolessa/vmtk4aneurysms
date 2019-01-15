
import paraview.simple as pv
import numpy as np

def area_averaged_wss(case):
    """ Function that calculates the area-averaged WSS
        for a surface where the wall shear stress field
        is defined. The function takes its input an 
        OpenFOAM case reader with the surface and fields.
        It automatically selects the wallShearComponent field
        and the surface wall. It returns an array with time
        in one column and the result in the other. 
    """
    # Update arrays to be used:
    # - wallShearComponent
    

    case.CellArrays  = ['wallShearComponent']

    # And select the surface where integration will be carried out
    case.MeshRegions = ['wall']

    # Get time-steps values
    timeSteps = np.array(case.TimestepValues)
    # Update time-step
    case.UpdatePipeline()

    # Integrate WSS on the wall
    computeWallArea = pv.IntegrateVariables()
    computeWallArea.Input = case
    computeWallArea.UpdatePipeline()
    
    # Get area of surface, in m2
    wallArea = computeWallArea.CellData.GetArray('Area').GetRange()[0]

    areaAveragedWSSList = []
    for timeStep in timeSteps:
        # Calculate WSS magnitude
        # Instantiate calculater filter
        calcMagWSS = pv.Calculator()
        calcMagWSS.Input = case
        calcMagWSS.ResultArrayName = 'magWSS'
        calcMagWSS.Function = '1056*mag(wallShearComponent)'
        calcMagWSS.AttributeType = 'Cell Data'
        calcMagWSS.UpdatePipeline(time=timeStep)

        # Integrate WSS on the wall
        integrateWSS = pv.IntegrateVariables()
        integrateWSS.Input = calcMagWSS
        integrateWSS.UpdatePipeline(time=timeStep)

        # Instantiate calculater filter
        areaAveragedWSS = pv.Calculator()
        areaAveragedWSS.Input = integrateWSS
        areaAveragedWSS.ResultArrayName = 'areaAveragedWSS'
        areaAveragedWSS.Function = 'magWSS/'+str(wallArea)
        areaAveragedWSS.AttributeType = 'Cell Data'
        areaAveragedWSS.UpdatePipeline(time=timeStep)
        areaAveragedWSSList.append([timeStep, 
                                    areaAveragedWSS.CellData.GetArray('areaAveragedWSS').GetRange()[0]])

    return np.asarray(areaAveragedWSSList)


def osi(ofCaseFile, 
        timeIndexRange, 
        outputFileName,
        timeStep=0.01,
        density=1056.0, # kg/m3
        wssFieldName='wallShearComponent', 
        patchName='wall'):
    """ 
        Function to calculate the oscillatory shear index and
        other time integrals variables of WSS over a time inter-
        val [Ti,Tf] indentified by time-step indices. The method
        (based o VTK), ignores the time-step size and 
        consider uniform time stepping (so, if the time-step is 
        large, the resulting fields may be very different if a va-
        riable time-step would be considered). The OSI field is 
        defined as:
        
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
    case = pv.OpenFOAMReader(FileName=ofCaseFile)

    # First we define only the field that are going to be used: the WSS on the aneurysm wall
    case.CellArrays = [wssFieldName]
    case.MeshRegions = [patchName]
    case.Createcelltopointfiltereddata = 0

    # Multiplying WSS per density
    densityTimesWSS = pv.Calculator()
    densityTimesWSS.Input = case
    densityTimesWSS.AttributeType   = 'Cell Data'
    densityTimesWSS.ResultArrayName = "WSS"
    densityTimesWSS.Function = str(density)+"*"+wssFieldName
    densityTimesWSS.UpdatePipeline()

    # Calculating the magnitude of the wss vector
    calcMagWSS = pv.Calculator()
    calcMagWSS.Input = densityTimesWSS
    calcMagWSS.AttributeType   = 'Cell Data'
    calcMagWSS.ResultArrayName = densityTimesWSS.ResultArrayName+"_magnitude"

    ## Get WSS field name
    wss = densityTimesWSS.ResultArrayName
    calcMagWSS.Function = "mag("+wss+")"
    calcMagWSS.UpdatePipeline()

    # Extract desired time range
    timeInterval = pv.ExtractTimeSteps()
    timeInterval.Input = calcMagWSS
    timeInterval.SelectionMode = "Select Time Range"
    timeInterval.TimeStepRange = timeIndexRange #[99, 199] # range in index
    timeInterval.UpdatePipeline()

    # Period given by time-steps
    period = (timeInterval.TimeStepRange[1] - timeInterval.TimeStepRange[0])*timeStep

    # Now compute the temporal statistics
    # filter computes the average values of all fields
    calcAvgWSS = pv.TemporalStatistics()

    calcAvgWSS.Input = timeInterval
    calcAvgWSS.ComputeAverage = 1
    calcAvgWSS.ComputeMinimum = 0
    calcAvgWSS.ComputeMaximum = 0
    calcAvgWSS.ComputeStandardDeviation = 0
    calcAvgWSS.UpdatePipeline()

    # Calculates OSI
    calcOSI = pv.Calculator()
    calcOSI.Input = calcAvgWSS
    calcOSI.ResultArrayName = 'OSI'
    calcOSI.AttributeType = 'Cell Data'

    # Getting fields:
    # - Get the average of the vector WSS field
    avgVecWSS = calcAvgWSS.CellData.GetArray(wss+"_average").GetName()
    # - Get the average of the magnitude of the WSS field 
    avgMagWSS = calcAvgWSS.CellData.GetArray(calcMagWSS.ResultArrayName+"_average").GetName()

    calcOSI.Function = "0.5*( 1 - ( mag( "+avgVecWSS+" ) )/"+avgMagWSS+" )"
    calcOSI.UpdatePipeline()

    # Compute Relative Residance Time
    calcRRT = pv.Calculator()
    calcRRT.Input = calcOSI
    calcRRT.ResultArrayName = 'RRT'
    calcRRT.AttributeType   = 'Cell Data'
    calcRRT.Function        = str(period)+"/mag("+avgVecWSS+")"
    calcRRT.UpdatePipeline()

    # Final processing of surface: merge blocks
    # and get surface for triangulation
    mergeBlocks = pv.MergeBlocks()
    mergeBlocks.Input = calcRRT
    mergeBlocks.UpdatePipeline()

    extractSurface = pv.ExtractSurface()
    extractSurface.Input = mergeBlocks
    extractSurface.UpdatePipeline()

    triangulate = pv.Triangulate()
    triangulate.Input = extractSurface
    triangulate.UpdatePipeline()

    pv.SaveData(outputFileName,triangulate)

    
def wss_statistics(timeAveragedSurface, aneurysmNeckArrayName, neckIsoValue=0.5):
    """
        Computes surface-averaged and maximum value 
        of time-averaged WSS for an aneurysm surface.
        Input is a PolyData surface with the averaged
        fields and the aneurysm neck contour field. 
        Return list with aneurysm area, WSSav and 
        WSSmax.
    """
    clipAneurysmNeck = pv.Clip()
    clipAneurysmNeck.Input = timeAveragedSurface
    clipAneurysmNeck.ClipType = 'Scalar'
    clipAneurysmNeck.Scalars  = [timeAveragedSurface.PointData.GetArray(aneurysmNeckArrayName).Name]
    clipAneurysmNeck.Invert   = 1   # gets smaller portion
    clipAneurysmNeck.Value    = neckIsoValue  # based on the definition of field ContourScalars
    clipAneurysmNeck.UpdatePipeline()

    # Finaly we integrate over Sa 
    integrateOverAneurysm = pv.IntegrateVariables()
    integrateOverAneurysm.Input = clipAneurysmNeck
    integrateOverAneurysm.UpdatePipeline()

    aneurysmArea = integrateOverAneurysm.CellData.GetArray("Area").GetRange()[0] # aneurysm area

    WSSav = integrateOverAneurysm.CellData.GetArray("WSS_magnitude_average").GetRange()[0]/aneurysmArea # averaged
    WSSmax = clipAneurysmNeck.CellData.GetArray("WSS_magnitude_average").GetRange()[1] # maximum value

    return [aneurysmArea, WSSav, WSSmax]


def area_averaged_wss_aneurysm(ofCaseFile,aneurysmClipSurface,aneurysmNeckArrayName,density=1056.0):
    """
        Function to compute surface integrals of 
        WSS over an aneurysm surface. It takes the 
        OpenFOAM case file and an extra surface where 
        it is stored a field with the aneurysm neck 
        line loaded as a ParaView PolyData surface.
        To my knowledge, it is important that the sur-
        face with thye neck line array be the same as 
        the wall surface of the OpenFOAM case, i.e.
        they are the same mesh.
    """

    # Clip original aneurysm surface in the neck line
    clipAneurysmNeck = pv.Clip()
    clipAneurysmNeck.Input = aneurysmClipSurface
    clipAneurysmNeck.ClipType = 'Scalar'
    clipAneurysmNeck.Scalars  = [aneurysmNeckArrayName]
    clipAneurysmNeck.Invert   = 1   # gets smaller portion
    clipAneurysmNeck.Value    = 0.5  # based on the definition of field ContourScalars
    clipAneurysmNeck.UpdatePipeline()
    
    integrateWSS = pv.IntegrateVariables()
    integrateWSS.Input = clipAneurysmNeck
    integrateWSS.UpdatePipeline()

    # Get area of surface, in m2
    aneurysmArea = integrateWSS.CellData.GetArray("Area").GetRange()[0]
    
    # Read OpenFOAM data and process the WSS
    # to get its magnitude
    ofData = pv.OpenFOAMReader(FileName=ofCaseFile)

    # Update arrays to be used:
    ofData.CellArrays = ['wallShearComponent']
    ofData.SkipZeroTime = 1
    # And select the surface where integration will be carried out
    ofData.MeshRegions = ['wall']

    # Get time-steps values
    timeSteps = np.array(ofData.TimestepValues)

    # Update time-step
    ofData.UpdatePipeline()

    # Triangulate data to coincide with
    triangulate = pv.Triangulate()
    triangulate.Input = ofData
    triangulate.UpdatePipeline()

    # Compute magnitude of WSS in each cell of the aneurysm surface
    magWSS = pv.Calculator()
    magWSS.Input = triangulate
    magWSS.ResultArrayName = 'magWSS'
    magWSS.Function = str(density)+'*mag('+triangulate.CellData.GetArray('wallShearComponent').Name+')'
    magWSS.AttributeType = 'Cell Data'
    magWSS.UpdatePipeline()

    # Resample OpenFOAM data to clipped aneeurysm surface
    resampleDataset = pv.ResampleWithDataset()
    resampleDataset.Input  = magWSS
    resampleDataset.Source = clipAneurysmNeck
    resampleDataset.PassCellArrays  = 1
    resampleDataset.PassPointArrays = 1
    resampleDataset.UpdatePipeline()

    # Since all fields in ResampleWithDataSet filter 
    # are interpolated to points, therefore
    # apply point data to cell data fielte
    pointToCellData = pv.PointDatatoCellData()
    pointToCellData.Input = resampleDataset
    pointToCellData.UpdatePipeline()

    areaAveragedWSSList = []

    # # Iterate over time-steps to compute time dependent variables
    # # only on the aneurysm surface: mag of WSS over time 
    for timeStep in timeSteps[-100:-1]: # get last cycle only
        # Integrate WSS on the wall
        integrateWSS = pv.IntegrateVariables()
        integrateWSS.Input = pointToCellData
        integrateWSS.UpdatePipeline(time=timeStep)

        # Instantiate calculater filter
        areaAveragedWSS = pv.Calculator()
        areaAveragedWSS.Input = integrateWSS
        areaAveragedWSS.ResultArrayName = 'areaAveragedWSS'
        areaAveragedWSS.Function = integrateWSS.CellData.GetArray('magWSS').Name+'/'+str(aneurysmArea)
        areaAveragedWSS.AttributeType = 'Cell Data'
        areaAveragedWSS.UpdatePipeline(time=timeStep)

        areaAveragedWSSList.append([timeStep, 
                                    areaAveragedWSS.CellData.GetArray('areaAveragedWSS').GetRange()[0]])

    return np.asarray(areaAveragedWSSList)
