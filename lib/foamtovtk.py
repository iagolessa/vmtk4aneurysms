"""Module with functions bridging FOAM to VTK."""

import sys
import warnings
from typing import Union
from itertools import compress, repeat

import vtk
from vtk.numpy_interface import dataset_adapter as dsa
from vmtk import vtkvmtk

from . import names
from . import polydatatools as tools
from . import polydatamath as pmath

def GetPatchFieldOverTime(
        foam_case: str,
        field_names: Union[str,list],
        active_patch_name: str,
        multi_region: bool = False,
        region_name: str = ''
    )   -> (names.polyDataType, dict):
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

    if multi_region:
        active_patch_name = '/'.join([region_name, active_patch_name])

    if active_patch_name not in patches:
        message = "Patch {} not in geometry surface.".format(active_patch_name)
        sys.exit(message)
    else:
        pass

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

    # With the selection of the patch above, here I have to find the
    # non-empty block. If there is only one block left (THE patch)
    # the loop will work regardless
    idNonEmptyBlock = nBlocks - 1 # default non empty block to last one
    nNonEmptyBlocks = 0

    for iBlock in range(nBlocks):
        block = blocks.GetBlock(iBlock)

        if block.GetNumberOfBlocks() != 0:
            idNonEmptyBlock = iBlock
            nNonEmptyBlocks += 1

        else:
            continue

    if nNonEmptyBlocks != 1:
        message = "There is more than one non-empty block when extracting {}.".format(
                    active_patch_name
                )
        sys.exit(message)
    else:
        pass

    # Get block
    block = blocks.GetBlock(idNonEmptyBlock)

    # The active patch is the only one left
    if multi_region:
    # (?) maybe this is a less error-prone alternative
    #if type(activePatch) == multiBlockType:

        # Multi region requires a multilevel block extraction
        activePatch = block.GetBlock(0).GetBlock(0)

    else:
        activePatch = block.GetBlock(0)

    # Check if array in surface
    cellArraysInPatch  = tools.GetCellArrays(activePatch)
    pointArraysInPatch = tools.GetPointArrays(activePatch)

    # Assuming that the user passes a list of fields to collect
    # get only the ones that are on the passed patch
    # Check whether field_names is a string (a single field),
    # if yes convert to list
    passed_fields = [field_names] if type(field_names) is str else field_names

    boolFieldsOnPatch = [field in cellArraysInPatch
                         for field in passed_fields]

    if all(boolFieldsOnPatch):
        print("Found all fields on the selected patch.")

        fieldsOnThePatch = passed_fields

    elif any(boolFieldsOnPatch):

        fieldsOnThePatch = list(compress(passed_fields, boolFieldsOnPatch))

        print(
            "Found only the following fields on the surface: {}". format(
                fieldsOnThePatch
            )
        )

    else:
        message = "None of the fields {} found on surface patch {}.".format(
                      passed_fields,
                      active_patch_name
                  )

        sys.exit(message)

    npActivePatch = dsa.WrapDataObject(activePatch)

    selectedFields = {}

    def _get_field(time, field_name):
        ofReader.UpdateTimeStep(time)
        return npActivePatch.GetCellData().GetArray(field_name)

    # get only the fields passed by the user
    selectedFields = {field_name: {time: _get_field(time, field_name)
                                   for time in timeSteps}
                      for field_name in fieldsOnThePatch}

    # Clean surface from any point or cell field
    activePatch = npActivePatch.VTKObject

    for arrayName in cellArraysInPatch:
        activePatch.GetCellData().RemoveArray(arrayName)

    for arrayName in pointArraysInPatch:
        activePatch.GetPointData().RemoveArray(arrayName)

    return activePatch, selectedFields

def FieldTimeStats(
        surface: names.polyDataType,
        field_name: str,
        temporal_field: dict,
        t_peak_systole: float,
        t_low_diastole: float
    )   -> names.polyDataType:
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
         names.peakSystoleWSS if field_name == names.WSS
                         else '_'.join([field_name, "peak_systole"]))
    )

    storeArray(
        (temporal_field.get(t_low_diastole, None),
         names.lowDiastoleWSS if field_name == names.WSS
                         else '_'.join([field_name, "low_diastole"]))
    )

    # Get period of time steps
    period   = lastTimeStep - firstTimeStep
    timeStep = period/len(timeSteps)

    # Append to the numpy surface wrap
    appendToSurface = npSurface.CellData.append

    # Compute the time-average of the WSS vector
    # assumes uniform time-step (calculated above)
    storeArray(
        (pmath.TimeAverage(fieldOverTime, timeStep, period),
         field_name + names.avg)
    )

    # If the array is a tensor of order higher than one
    # compute its magnitude too
    if len(fieldOverTime.shape) == 3:
        # Compute the time-average of the magnitude of the WSS vector
        fieldMagOverTime = pmath.NormL2(fieldOverTime, 2)

        storeArray(
            (pmath.TimeAverage(fieldMagOverTime, timeStep, period),
             names.TAWSS if field_name == names.WSS
                         else field_name + names.mag + names.avg)
        )

    else:
        pass

    storeArray(
        (fieldMagOverTime.max(axis=0),
         field_name + names.mag + names.max_)
    )

    storeArray(
        (fieldMagOverTime.min(axis=0),
         field_name + names.mag + names.min_)
    )

    # Finally, append all arrays to surface
    for array, name in arraysToBeStored:
        appendToSurface(array, name)

    return npSurface.VTKObject

def FieldSurfaceAverage(
        foam_case: str,
        field_name: str,
        patch_name: str,
        multi_region: bool = False,
        region_name: str = ''
    )   -> dict:
    """Compute the surface-averaged field over time.

    Function to compute surface integrals of a field over an aneurysm or
    vessels surface. It takes the OpenFOAM case file. If the field os a vector
    or tensor, first it computes its L2-norm.
    """
    surface, fieldsOverTime = GetPatchFieldOverTime(
                                  foam_case,
                                  field_name,
                                  patch_name,
                                  multi_region=multi_region,
                                  region_name=region_name
                              )


    fieldOverTime = fieldsOverTime[field_name]

    # Check type of field: scalar, vector, tensor
    # better here then inside the for
    # TODO: maybe work in a better field type function identifier
    nComponents = list(fieldOverTime.values())[0].shape[-1]

    if nComponents == 3 or nComponents == 6:
        fieldOverTime = {time: pmath.NormL2(fieldOverTime.get(time), 1)
                         for time in fieldOverTime.keys()}
    else:
        pass

    npSurface = dsa.WrapDataObject(surface)

    # Function to compute average of wss over surface
    def field_average_on_surface(t):

        npSurface.CellData.append(fieldOverTime.get(t), field_name)

        return pmath.SurfaceAverage(npSurface.VTKObject, field_name)

    return {time: field_average_on_surface(time)
            for time in fieldOverTime.keys()}

def FieldSurfaceAverageOnPatch(
        foam_case: str,
        field_name: str,
        boundary_patch: str,
        patch_surface_id: names.polyDataType = None,
        patch_array_name: str = '',
        patch_boundary_value: float = 0.0,
        multi_region: bool = False,
        region_name: str = ''
    ):
    """Compute the surface-averaged of a field over time over a patch only.

    Given an OpenFOAM simulation result, a field and the boundary patch where
    the field is defined, compute the surface-average of the field over that
    surface along time.

    If a surface, equal to the boundary is passed with an array specifying a
    patch of the surface with the values 0 in it, compute the surface-average
    over this patch, instead of the whole surface.

    Function to compute surface integrals of sigma field over an aneurysm or vessels
    surface. It takes the OpenFOAM case file and an optional surface where it
    is stored a field with the aneurysm neck line loaded as a ParaView PolyData
    surface. If the surface is None, it computes the integral over the entire
    sur- face. It is essential that the surface with the ne- ck array be the
    same as the wall surface of the OpenFOAM case, i.e. they are the same mesh.
    """
    # Define condition to compute on aneurysm portion
    computeOnPatch = patch_surface_id is not None

    surface, fieldsOverTime = GetPatchFieldOverTime(
                                  foam_case,
                                  field_name,
                                  boundary_patch,
                                  multi_region=multi_region,
                                  region_name=region_name
                              )

    fieldOverTime = fieldsOverTime[field_name]

    # It returns a dict, get only the important field

    # Map neck array into surface
    if computeOnPatch:
        # Map  array field into current surface
        # (aneurysmExtract triangulas the surface)
        # Important: both surface must match the scaling

        surfaceProjection = vtkvmtk.vtkvmtkSurfaceProjection()
        surfaceProjection.SetInputData(surface)
        surfaceProjection.SetReferenceSurface(patch_surface_id)
        surfaceProjection.Update()

        surface = surfaceProjection.GetOutput()
    else:
        pass

    npSurface = dsa.WrapDataObject(surface)

    # Function to compute average of field over surface/patch
    def average_on_patch(t):
        # Add instant field array on surface
        field_t = fieldOverTime.get(t)

        npSurface.CellData.append(
            field_t,
            field_name + "_t"
        )

        if computeOnPatch:
            # Clip patch portion
            patch = tools.ClipWithScalar(
                        npSurface.VTKObject,
                        patch_array_name,
                        patch_boundary_value
                    )

            npPatch = dsa.WrapDataObject(patch)
            surfaceToComputeAvg = npPatch

        else:
            surfaceToComputeAvg = npSurface

        return pmath.SurfaceAverage(surfaceToComputeAvg.VTKObject, field_name+'_t')

    return {time: average_on_patch(time)
            for time in fieldOverTime.keys()}
