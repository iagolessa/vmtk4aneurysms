"""Collection of conversion tools between OpenFOAM and VTK."""

import sys
import warnings
import numpy as np
from typing import Union
from itertools import compress, repeat

import vtk
from vtk.numpy_interface import dataset_adapter as dsa
from vmtk import vtkvmtk

from . import names
from . import polydatatools as tools
from . import polydatamath as pmath

def _read_foam_data(
    foam_case: str,
    patch_name: str="",
    multi_region: bool=False,
    region_name: str="",
    enable_point_fields: bool=False,
    set_cell_to_point: bool=False
)   -> (names.foamReaderType,
        Union[names.polyDataType, names.unstructuredGridType]):
    """Read and process OpenFOAM data.

    Given a FOAM case file and a patch name, this function reads in the data
    and returns the OpenFOAM VTK reader and the patch structure (be it a patch
    or the internal mesh if an empty string is passed to patch_name). By
    default, it does not assume a multi-region dataset, which should be passed
    as a bool together with the region name. It loads all the point and cell
    fields that exists in the meshes.

    By default, it does not convert cell to point fields, although this is
    possible through the option 'set_cell_to_point'.

    If the dataset has temporal data, it can be accessed through the FOAM
    reader object.
    """

    if multi_region and region_name == "":
        raise NameError("Please, pass a region name if multiregion is on.")

    # Get the internal mesh if patch passed is found in of reader
    if patch_name == "":
        print("Empty patch name: computing volumetric mesh fields.")

    active_patch_name = "internalMesh" if patch_name == "" else patch_name

    if multi_region:
        active_patch_name = '/'.join([region_name, active_patch_name])

    # Read OF case reader
    ofReader = vtk.vtkPOpenFOAMReader()
    ofReader.SetFileName(foam_case)
    ofReader.AddDimensionsToArrayNamesOff()
    ofReader.DecomposePolyhedraOff()
    ofReader.SkipZeroTimeOn()
    ofReader.SetCreateCellToPoint(set_cell_to_point) # important for resampling
    ofReader.DisableAllLagrangianArrays()
    ofReader.EnableAllCellArrays()

    if enable_point_fields:
        ofReader.EnableAllPointArrays()
    else:
        ofReader.DisableAllPointArrays()

    ofReader.Update()

    # Update OF reader with only selected patch
    patches = list((ofReader.GetPatchArrayName(index)
                    for index in range(ofReader.GetNumberOfPatchArrays())))

    if active_patch_name not in patches:
        raise ValueError(
                  "Patch {} not in geometry surface.".format(active_patch_name)
              )

    # Set active patch
    for patchName in patches:
        if patchName == active_patch_name:
            ofReader.SetPatchArrayStatus(patchName, 1)
        else:
            ofReader.SetPatchArrayStatus(patchName, 0)

    ofReader.Update()

    # Get blocks where the path or inrnalMesh is (not empty one)
    blocks  = ofReader.GetOutput()
    nBlocks = blocks.GetNumberOfBlocks()

    # If the case is not multiregion and the internalMesh is sought then, the
    # block will result in the internal mesh directly
    if not multi_region and active_patch_name == "internalMesh":
        activePatch = blocks.GetBlock(0)

    # All the other cases (no multiregion but a patch, any case
    # of the multiregion situation) then there will be a multiblock dataset
    else:
        # Hence I have to find the
        # non-empty block. If there is only one block left (the patch)
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
            raise ValueError(
                      "There is more than one non-empty block when extracting {}.".format(
                          active_patch_name
                      )
                  )

        # Get block
        block = blocks.GetBlock(idNonEmptyBlock)

        # The active patch is the only one left
        if multi_region and patch_name != "":
        # (?) maybe this is a less error-prone alternative
        #if type(activePatch) == multiBlockType:
            # Multi region requires a multilevel block extraction
            activePatch = block.GetBlock(0).GetBlock(0)

        else:
            activePatch = block.GetBlock(0)


    return (ofReader, activePatch)

def GetPatchFieldOverTime(
        foam_case: str,
        field_names: Union[str,list],
        active_patch_name: str,
        multi_region: bool = False,
        region_name: str = ''
    )   -> (names.polyDataType, dict):
    """Return a time-varying patch field from an OpenFOAM case.

    Given an OpenFOAM case file (foam_case: path to the .foam file), the field
    name (or field names as a list) and the patch name (active_patch_name),
    return a tuple with the patch surface and a dictionary with the
    time-varying field with the instants as keys and the value the field given
    as a VTK Numpy Array.

    It may also return the volumetric mesh data. In this case, the
    'active_patch_name' passed must be the empty string ("").
    """

    ofReader, activePatch = _read_foam_data(
                                 foam_case,
                                 patch_name=active_patch_name,
                                 multi_region=multi_region,
                                 region_name=region_name,
                                 enable_point_fields=False,
                                 set_cell_to_point=False
                             )

    # Get list with time steps
    nTimeSteps = ofReader.GetTimeValues().GetNumberOfValues()
    timeSteps  = list((ofReader.GetTimeValues().GetValue(id_)
                       for id_ in range(nTimeSteps)))

    # Check if array in surface
    cellArraysInPatch  = tools.GetCellArrays(activePatch)
    pointArraysInPatch = tools.GetPointArrays(activePatch)

    # Assuming that the user passes a list of fields to collect
    # get only the ones that are on the passed patch
    # Check whether field_names is a string (a single field),
    # if yes convert to list
    # TODO: account for point fields too in this framework?
    passed_fields = [field_names] if type(field_names) is str else field_names

    boolFieldsOnPatch = [field in cellArraysInPatch
                         for field in passed_fields]

    if all(boolFieldsOnPatch):
        print("Found all fields on the selected region/patch.")

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

    # Instant function to get field values
    # (needed to call method UpdateTimeStep)
    def _get_fields(time):
        ofReader.UpdateTimeStep(time)

        return {field_name: npActivePatch.GetCellData().GetArray(field_name)
                for field_name in fieldsOnThePatch}

    # Get dict with values per time of all fields
    fieldValuesPerTime = {time: _get_fields(time) for time in timeSteps}

    # 'Reshape' dict to et each field over time
    fieldsOverTime = {field_name: {time: fields[field_name]
                                   for time, fields in fieldValuesPerTime.items()}
                      for field_name in fieldsOnThePatch}

    # Clean surface from any point or cell field
    activePatch = npActivePatch.VTKObject

    for arrayName in cellArraysInPatch:
        activePatch.GetCellData().RemoveArray(arrayName)

    for arrayName in pointArraysInPatch:
        activePatch.GetPointData().RemoveArray(arrayName)

    return activePatch, fieldsOverTime

def FieldTimeStats(
        vtk_object: Union[names.polyDataType, names.unstructuredGridType],
        temporal_fields: dict,
        t_peak_systole: float,
        t_low_diastole: float,
        field_names: Union[str, list]=None,
    )   -> names.polyDataType:
    """Compute field time statistics from OpenFOAM data.

    Get time statistics of a field (or fields) defined on a surface S (or
    volume V) over time for a cardiac cycle, generated with OpenFOAM. The
    fields are passed through a dictionary where keys are the field names and
    values are also a dictionary with time values as keys and arguments as the
    temporal series of each field as a VTK numpy array. Outputs a surface (or a
    volume) with: the time-averaged of the field magnitude (if not a scalar),
    maximum and minimum over time, peak-systole and low-diastole fields.

    Arguments:
    vtk_object (vtkPolyData or vtkUnstructuredGrid) -- the surface or volume
    where the field is defined;

    temporal_fields (dict) -- a dictionary with the field over each instant
    defined over vtk_object. If field_names is a list, this must be a dict
    which keys are the field_names and value a dict with the respective field
    over time, as follows:

    .. code::
        temporal_fields = {field_name1: {t1: V1, ..., tn: Vn},
                           field_name2: {t1: U1, ..., tn: Un},
                           ...,
                           field_nameN: {t1: W1, ..., tn: Wn}}

    t_peak_systole (float) -- instant of the peak systole;

    t_low_diastole (float) -- instant of the low diastole;

    field_names (str or list of strs, optional) -- either a string with the
    field name or a list of string with the all the fields that must be
    included in the computation. if None, will compute all fields passed
    through temporal_fields.keys().
    """

    # Operate on copy of the vtk_object (transform applies the identity
    # transformation if none is passed)
    vtk_object  = tools.CopyVtkObject(vtk_object)
    nCells      = vtk_object.GetNumberOfCells()
    npVtkObject = dsa.WrapDataObject(vtk_object)

    # Append to the numpy surface wrap
    appendToVtkObject = npVtkObject.CellData.append

    # Check if first level of temporal_fields dict are strings
    if not all(type(key) == str for key in temporal_fields.keys()):
        raise ValueError(
                  "Expecting only strings in temporal_fields first-level keys."
              )

    if field_names is not None:

        # Convert to list if string
        fieldsToUse = [field_names] \
                      if type(field_names) == str \
                      else field_names

    else:
        fieldsToUse = list(temporal_fields.keys())


    # List of tuples to store stats arrays and their name
    # [(array1, name1), ... (array_n, name_n)]
    arraysToBeStored = []
    storeArray = arraysToBeStored.append

    for field_name in fieldsToUse:
        print("Computing stats for field {}".format(field_name))

        temporal_field = temporal_fields[field_name]

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
        if nCells not in fieldOverTime.shape:
            raise ValueError(
                      "VTK Object cells number and of field do not match."
                  )

        # Check if low diastole or peak systoel not in time list
        lastTimeStep  = max(timeSteps)
        firstTimeStep = min(timeSteps)

        if t_low_diastole not in timeSteps:
            warningMsg = "Low diastole instant not in " \
                         "time-steps list for field {}. " \
                         "Using last time-step.".format(field_name)

            warnings.warn(warningMsg)

            t_low_diastole = lastTimeStep

        elif t_peak_systole not in timeSteps:
            warningMsg = "Peak-systole instant not in " \
                         "time-steps list for field {}. " \
                         "Using first time-step.".format(field_name)

            warnings.warn(warningMsg)

            t_peak_systole = firstTimeStep

        else:
            pass

        # Get peak-systole and low-diastole WSS
        storeArray(
            (temporal_field.get(t_peak_systole, None),
             names.peakSystoleWSS \
             if field_name == names.WSS \
             else '_'.join([field_name, "peak_systole"]))
        )

        storeArray(
            (temporal_field.get(t_low_diastole, None),
             names.lowDiastoleWSS if field_name == names.WSS
                             else '_'.join([field_name, "low_diastole"]))
        )

        # Compute the time-average of the WSS vector
        # assumes uniform time-step (calculated above)
        storeArray(
            (
                pmath.TimeAverage(
                    fieldOverTime,
                    np.array(timeSteps)
                ),
                field_name + names.avg
            )
        )

        # If the array is a tensor of order higher than one
        # compute its magnitude too
        if len(fieldOverTime.shape) == 3:
            # Compute the time-average of the magnitude of the WSS vector
            fieldMagOverTime = pmath.NormL2(fieldOverTime, 2)

            storeArray(
                (
                    pmath.TimeAverage(
                        fieldMagOverTime,
                        np.array(timeSteps)
                    ),
                    names.TAWSS \
                        if field_name == names.WSS \
                        else field_name + names.mag + names.avg
                )
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
        appendToVtkObject(array, name)

    return npVtkObject.VTKObject

def FieldSurfaceAverage(
        foam_case: str,
        field_name: str,
        patch_name: str,
        multi_region: bool = False,
        region_name: str = ''
    )   -> dict:
    """Compute the surface-averaged metric of a temporal field.

    Given the foam_case file, compute surface integrals of a field over a
    specfied patch name.

    .. warning ::
        If the field os a vector or tensor, first it computes its L2-norm.
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
        vtk_object: Union[names.polyDataType, names.unstructuredGridType],
        temporal_fields: dict,
        field_names: Union[str, list]=None,
        patch_surface_id: names.polyDataType=None,
        patch_array_name: str='',
        patch_boundary_value: float=0.0
    ):
    """Compute the surface-averaged of a temporal field over a portion.

    Given a surface patch and the fields defined on it, as obtained by the
    function GetPatchFieldOverTime, compute the surface-average of the field(s)
    over that surface along time.

    To compute the surface-average over a portion only, instead of the whole
    surface, an extra surface (patch_surface_id), equal to the "vtk_object"
    must be passed with an array (patch_array_name) specifying a portion of the
    surface by the value 0 on it.
    """

    # Operate on copy of the vtk_object (transform applies the identity
    # transformation if none is passed)
    vtk_object  = tools.CopyVtkObject(vtk_object)

    # Define condition to compute on aneurysm portion
    computeOnPatch = patch_surface_id is not None

    # Fields to compute
    if field_names is not None:

        # Convert to list if string
        fieldsToUse = [field_names] \
                      if type(field_names) == str \
                      else field_names

    else:
        fieldsToUse = list(temporal_fields.keys())

    npVtkObject = dsa.WrapDataObject(vtk_object)

    # Add all fields to the surface (all times)
    for fname in fieldsToUse:
        for instant, field_t in temporal_fields[fname].items():
            npVtkObject.CellData.append(
                field_t,
                fname + "_" + str(instant)
            )

    # Map neck array into surface
    if computeOnPatch:
        # Map  array field into current surface
        # (aneurysmExtract triangulas the surface)
        # Important: both surface must match the scaling
        vtk_object = tools.ProjectPointArray(
                         npVtkObject.VTKObject,
                         patch_surface_id,
                         patch_array_name
                     )

        vtk_object = tools.ClipWithScalar(
                         vtk_object,
                         patch_array_name,
                         patch_boundary_value
                     )

    else:
        vtk_object = npVtkObject.VTKObject

    return {field_name: {time: pmath.SurfaceAverage(
                                    vtk_object,
                                    field_name + '_' + str(time)
                                )
                         for time, field in temporal_fields[field_name].items()}
            for field_name in fieldsToUse}
