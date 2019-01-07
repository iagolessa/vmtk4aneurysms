#! /usr/bin/env python

import paraview.simple as pv
import numpy as np

def osi(ofCaseFile, 
        timeIndexRange, 
        outputFileName, 
        wssFieldName='wallShearComponent', 
        patchName='wall'):
    """ Function to calculate the oscillatory shear index
        for a time interval [0,T] indentified by time step indices.
        The method (based o VTK), ignores the time-step size
        and consider uniform time stepping (so, if the time-step
        is large, the resulting field may be large). The OSI 
        field is defined as:
        
            OSI = 0.5*( 1 - norm2(int WSS dt) / int norm2(WSS) dt )
            
        where int implies the integral over time between instants
        t1 and t2 (usually for a cardiac cycle, therefore 
        [t1, t2] = [0, T]) and norm2 is the L2 norm an Euclidean
        vector field; WSS is the wall shear stress defined on the
        input surface.
        
        
        Input args:
        - OpenFOAM case file (str): name of OpenFOAM .foam case;
        - wssFieldName (str): string containing the name of the
            wall shear stress field (default="wallShearComponent");
        - patchName (str): patch name where to calculate OSI
            (default="wall");
        - timeIndexRange (list): list of initial and final time-
            steps indices limits of the integral [0, T];
        - outputFileName (str): file name for the output file with 
            osi field (must be a .vtp file).
    """
    case = pv.OpenFOAMReader(FileName=ofCaseFile)
    
    # First we define only the field that are going to be used: the WSS on the aneurysm wall
    case.CellArrays = [wssFieldName]
    case.MeshRegions = [patchName]
    case.Createcelltopointfiltereddata = 0
    
    # Calculating the magnitude of the wss vector
    calcMagWSS = pv.Calculator()

    calcMagWSS.Input = case
    calcMagWSS.AttributeType = 'Cell Data'
    calcMagWSS.ResultArrayName = wssFieldName+"_magnitude"
    
    ## Get WSS field name
    wss = case.CellData.GetArray(wssFieldName).GetName()
    calcMagWSS.Function = "mag("+wss+")"
    calcMagWSS.UpdatePipeline()

    # Extract desired time range
    timeInterval = pv.ExtractTimeSteps()
    timeInterval.Input = calcMagWSS
    timeInterval.SelectionMode = "Select Time Range"
    timeInterval.TimeStepRange = timeIndexRange #[99, 199] # range in index
    timeInterval.UpdatePipeline()
    
    # Now compute the temporal statistics
    calcAvgWSS = pv.TemporalStatistics()

    calcAvgWSS.Input = timeInterval
    calcAvgWSS.ComputeAverage = 1
    calcAvgWSS.ComputeMinimum = 0
    calcAvgWSS.ComputeMaximum = 0
    calcAvgWSS.ComputeStandardDeviation = 0
    calcAvgWSS.UpdatePipeline()

    calcOSI = pv.Calculator()
    calcOSI.Input = calcAvgWSS
    calcOSI.ResultArrayName = 'OSI'
    calcOSI.AttributeType = 'Cell Data'

    # Getting fields:
    # - Get the average of the vector WSS field
    avgVecWSS = calcAvgWSS.CellData.GetArray(wssFieldName+"_average").GetName()
    # - Get the average of the magnitude of the WSS field 
    avgMagWSS = calcAvgWSS.CellData.GetArray(calcMagWSS.ResultArrayName+"_average").GetName()

    # Compute OSI
    calcOSI.Function = "0.5*( 1 - ( mag( "+avgVecWSS+" ) )/"+avgMagWSS+" )"
    calcOSI.UpdatePipeline()

    mergeBlocks = pv.MergeBlocks()
    mergeBlocks.Input = calcOSI
    mergeBlocks.UpdatePipeline()

    extractSurface = pv.ExtractSurface()
    extractSurface.Input = mergeBlocks
    extractSurface.UpdatePipeline()
    
    triangulate = pv.Triangulate()
    triangulate.Input = extractSurface
    triangulate.UpdatePipeline()
    
    pv.SaveData(outputFileName,triangulate)
    
    
def area_averaged_wss(case):
    """ Function that calculates the area-averaged WSS
        for a surface where the wall shear stress field
        is defined. The function takes its input an 
        OpenFOAM case reader with the surface and fields.
        It automatically selects the wallShearComponent field
        and the surface wall.
    """
    # Update arrays to be used:
    # - wallShearComponent
    case.CellArrays = ['wallShearComponent']
    
    # And select the surface where integration will be carried out
    case.MeshRegions = ['wall']
    
    # Get time-steps values
    timeSteps = np.array(case.TimestepValues)
    # Update time-step
    case.UpdatePipeline()

    # Integrate WSS on the wall
    integrateWSS = pv.IntegrateVariables()
    integrateWSS.Input = case
    integrateWSS.UpdatePipeline()

    # Get area of surface, in m2
    wallArea = integrateWSS.CellData.GetArray('Area').GetRange()[0]

    # Instantiate calculater filter
    calcWSS = pv.Calculator()

    areaAveragedWSS = []
    for timeStep in timeSteps:  
        # Calculate WSS magnitude
        calcWSS.Input = integrateWSS
        calcWSS.ResultArrayName = 'areaAveragedWSS'
        calcWSS.Function = '(1/'+str(wallArea)+')*1056*mag('+integrateWSS.CellData.GetArray('wallShearComponent').Name+')'
        calcWSS.AttributeType = 'Cell Data'
        calcWSS.UpdatePipeline(time=timeStep)
        areaAveragedWSS.append([timeStep, 
                                calcWSS.CellData.GetArray('areaAveragedWSS').GetRange()[0]])
    
    return np.asarray(areaAveragedWSS)


# # Inform the path to the .foam files
# # THis calculation considers three meshes used for the study
# # meshCoarse = pv.OpenFOAMReader(FileName='/home/iagolessa/foam/iagolessa-4.0/run/aneurysms/ruptured/fluidFlow/Newtonian/case1/mesh700k/mesh700k.foam')
# # meshIntermediate = pv.OpenFOAMReader(FileName='/home/iagolessa/foam/iagolessa-4.0/run/aneurysms/ruptured/fluidFlow/Newtonian/case1/mesh1500k/mesh1500k.foam')
# # meshFine = pv.OpenFOAMReader(FileName='/home/iagolessa/foam/iagolessa-4.0/run/aneurysms/ruptured/fluidFlow/Newtonian/case1/mesh3000k/mesh3000k.foam')

# pathToFoamCase = "/home/iagolessa/foam/iagolessa-4.0/run/aneurysms/unruptured/fluidFlow/Newtonian/case17/mesh1100k/" 
# foamFileName = "case17Newtonian.foam"
# ofData = pv.OpenFOAMReader(FileName=pathToFoamCase+foamFileName)

# # meshCoarse.SkipZeroTime = 0
# # meshCoarse.Adddimensionalunitstoarraynames = 1
# # ofData.MeshRegions.Available, ofDataFile.CellArrays


# meshIntermediateWSS = area_averaged_wss(meshIntermediate)
# meshFineWSS = area_averaged_wss(meshFine)

# import matplotlib.pyplot as plt
# import seaborn as sb

# plt.style.use('classic')

# %matplotlib widget

# fig = plt.figure()

# plt.plot(meshIntermediateWSS[:,0], meshIntermediateWSS[:,1], 'b')
# plt.plot(meshFineWSS[:,0], meshFineWSS[:,1], 'r')

# plt.xlabel('Time')
# plt.ylabel('Area-averaged WSS (Pa)')

# plt.grid()
# plt.show()