# Copyright (C) 2022, Iago L. de Oliveira

# vmtk4aneurysms is free software: you can redistribute it and/or modify
# it under the terms of the GNU General Public License as published by
# the Free Software Foundation, either version 3 of the License, or
# (at your option) any later version.

# This program is distributed in the hope that it will be useful,
# but WITHOUT ANY WARRANTY; without even the implied warranty of
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
# GNU General Public License for more details.

# You should have received a copy of the GNU General Public License
# along with this program.  If not, see <http://www.gnu.org/licenses/>.

"""Collection of tools to manipulate vascular models and aneurysms."""

import sys
import vtk
import numpy as np
import morphman.common as mplib
from morphman.manipulate_curvature import extract_single_line

from vmtk import vtkvmtk
from vmtk import vmtkscripts
from scipy import interpolate

from vtkmodules.numpy_interface import dataset_adapter as dsa

from .lib import names
from .lib import centerlines as cl
from .lib import constants as const
from .lib import polydatatools as tools
from .lib import polydatageometry as geo

from vmtk4aneurysms.vascular_classes import VascularCenterline
from vmtk4aneurysms.neck_extractor import ComputeGeodesicDistanceToAneurysmNeck

def ClipVasculatureWithPlane(
        vascular_surface: names.polyDataType,
        plane_center: tuple,
        plane_normal: tuple
    )   -> names.polyDataType:
    """Clip vascular tree section with a plane.

    Given a plane center and normal, clip the passed vascular surface model at
    the plane. The portion of the surface kept is on the normal direction.
    """

    # Clip vessel at inlet location
    vascular_surface = tools.SeamPlaneTubularStructureMarker(
                           vascular_surface,
                           plane_center=plane_center,
                           plane_normal=plane_normal,
                           seam_scalar_array_name=names.SeamScalarsArrayName
                       )

    # The SeamScalars are positive (1.0) in the region of the positiove
    # direction of the plane normal so use inside_out False.
    return tools.ClipWithScalar(
               vascular_surface,
               names.SeamScalarsArrayName,
               const.zero,
               inside_out=False
           )

def ClipVasculature(
        vascular_surface: names.polyDataType,
        centerlines=None
    )   -> names.polyDataType:
    """Clip a vascular surface segment, by selecting end points.

    Given a vascular surface, the user is prompted to select points
    on the surface that 1) identifies the surface's bulk and 2) where the
    vasculature should be clipped. Uses, internally, the
    'vmtksurfaceendclipper' script.
    """

    if centerlines is None:
        centerlines = VascularCenterline.GenerateCenterlines(vascular_surface)

    geoCenterlines = cl.ComputeCenterlineGeometry(centerlines)

    FrenetTangentArrayName = "FrenetTangent"

    surfaceEndClipper = vmtkscripts.vmtkSurfaceEndClipper()
    surfaceEndClipper.Surface = vascular_surface
    surfaceEndClipper.CenterlineNormals = 1
    surfaceEndClipper.Centerlines = geoCenterlines
    surfaceEndClipper.FrenetTangentArrayName = FrenetTangentArrayName
    surfaceEndClipper.Execute()

    return surfaceEndClipper.Surface

def _compute_local_wlr(diameter):
    if diameter > const.VesselLargeDiameter:
        return const.WlrLarge

    elif diameter < const.VesselMediumDiameter:
        return const.WlrMedium

    else:
        # Linear threshold
        deltaWlr = const.WlrLarge - const.WlrMedium
        deltaDiameter = const.VesselLargeDiameter - const.VesselMediumDiameter
        angCoeff = deltaWlr/deltaDiameter

        return const.WlrMedium + angCoeff*(diameter - const.VesselMediumDiameter)

def ComputeVasculatureThickness(
        vascular_surface: names.polyDataType,
        centerlines: names.polyDataType=None,
        thickness_field_name: str=names.ThicknessArrayName,
        set_uniform_wlr: bool=False,
        uniform_wlr_value: float=const.WlrMedium
    )   -> names.polyDataType:
    """Compute thickness of a vasculature based on its diameter and WLR.

    Given input surface with the radius array, computes the thickness by
    multiplying by the wall-to-lumen ration. The aneurysm portion is also
    multiplyed.
    """

    # Compute centerlines
    if not centerlines:
        centerlines = VascularCenterline.GenerateCenterlines(vascular_surface)

    # Compute distance to centerlines
    # It will hold the thickness field at the end
    distanceToCenterlines = vtkvmtk.vtkvmtkPolyDataDistanceToCenterlines()
    distanceToCenterlines.SetInputData(vascular_surface)
    distanceToCenterlines.SetCenterlines(centerlines)

    distanceToCenterlines.SetUseRadiusInformation(True)
    distanceToCenterlines.SetEvaluateCenterlineRadius(True)
    distanceToCenterlines.SetEvaluateTubeFunction(False)
    distanceToCenterlines.SetProjectPointArrays(False)

    distanceToCenterlines.SetDistanceToCenterlinesArrayName(
        thickness_field_name
    )

    distanceToCenterlines.SetCenterlineRadiusArrayName(names.VascularRadiusArrayName)
    distanceToCenterlines.Update()

    # use numpy interface with VTK
    npSurface = dsa.WrapDataObject(distanceToCenterlines.GetOutput())

    distanceArray = npSurface.GetPointData().GetArray(thickness_field_name)
    radiusArray   = npSurface.GetPointData().GetArray(names.VascularRadiusArrayName)

    # This portion evaluates if distance is much higher
    # than the actual radius array
    # This necessarily will need some smoothing

    # Set high and low threshold factors
    # Are they arbitrary?
    highRadiusThresholdFactor = 1.4
    lowRadiusThresholdFactor  = 0.9

    npMaxRadiusLim = highRadiusThresholdFactor*radiusArray
    npMinRadiusLim = lowRadiusThresholdFactor*radiusArray

    distanceArray = np.where(
                        distanceArray > npMaxRadiusLim,
                        npMaxRadiusLim,
                        distanceArray
                    )

    distanceArray = np.where(
                        distanceArray < npMinRadiusLim,
                        radiusArray,
                        distanceArray
                    )

    # Smooth the distance to centerline array to avoid sudden changes of
    # thickness in certain regions
    surface = tools.SmoothSurfacePointField(
                  npSurface.VTKObject,
                  thickness_field_name,
                  niterations=5
              )

    npSurface = dsa.WrapDataObject(surface)

    # Multiply by WLR to have a prelimimar thickness array
    # I assume that the WLR is the same for medium sized arteries
    # but I can change this in a point-wise manner based on
    # the local radius array by using the algorithm contained
    # in the vmtksurfacearrayoperation script
    distanceArray = npSurface.GetPointData().GetArray(thickness_field_name)
    radiusArray   = npSurface.GetPointData().GetArray(names.VascularRadiusArrayName)

    if set_uniform_wlr:

        npSurface.PointData.append(
            dsa.VTKArray([
                uniform_wlr_value*(2.0*r)
                for r in distanceArray
            ]),
            thickness_field_name
        )

    else:
        print("Using non uniform WLR", end="\n")

        # Compute are store local WLR for debug
        localWLRArray = dsa.VTKArray([
                            _compute_local_wlr(2.0*r)
                            for r in distanceArray
                        ])

        npSurface.PointData.append(
            localWLRArray,
            "LocalWLR"
        )

        # Compute thickness array and replace the thickness array
        # (originally stored as the distance to neck array with a new one)
        npSurface.PointData.append(
            localWLRArray*(2.0*distanceArray),
            thickness_field_name
        )

    vascular_surface = npSurface.VTKObject
    vascular_surface.GetPointData().RemoveArray(names.VascularRadiusArrayName)

    return vascular_surface

def WallTypeClassification(
        surface: names.polyDataType,
        low_wss: float=5.0,
        high_wss: float=10.0,
        low_osi: float=0.001,
        high_osi: float=0.01,
        distance_to_neck_array: str=names.DistanceToNeckArrayName,
        neck_iso_value: float=const.NeckIsoValue
    )   -> names.polyDataType:
    """Based on the WSS hemodynamics, characterize an aneurysm wall morphology.

    Based on the TAWSS and OSI fields, identifies the aneurysm regions prone to
    atherosclerotic walls (thicker walls) and red wall (thinner) by adding a
    new array on the passed surface name "WallType" with the following values:

    .. table:: Local wall type characterization
        :widths: auto

        =====   ===============
        Label   Wall Type
        =====   ===============
            0   Normal wall
            1   Atherosclerotic
            2   "Red" wall
        =====   ===============

    Classifications based on the references:

        [1] Furukawa et al. "Hemodynamic characteristics of hyperplastic
        remodeling lesions in cerebral aneurysms". PLoS ONE. 2018 Jan
        16;13:1–11.

        [2] Cebral et al. "Local hemodynamic conditions associated with focal
        changes in the intracranial aneurysm wall". American Journal of
        Neuroradiology.  2019; 40(3):510–6.
    """
    normalWall  = const.IaWallTypes["RegularWall"]
    thickerWall = const.IaWallTypes["AtheroscleroticWall"]
    thinnerWall = const.IaWallTypes["RedWall"]

    # Maybe put this limiting values to be passed by the user
    # for flexibility
    limitHemodynamics = {names.TAWSS: {"low": low_wss, "high": high_wss},
                         names.OSI  : {"low": low_osi, "high": high_osi}#,
                         #names.RRT  : {"low": 0.25,  "high": 0.75}
                        }

    arraysInSurface = tools.GetPointArrays(surface) + \
                      tools.GetCellArrays(surface)

    if distance_to_neck_array not in arraysInSurface:
        print("Distance to neck array name not in surface. Computing it.")

        surface = ComputeGeodesicDistanceToAneurysmNeck(
                    surface,
                    mode="interactive"
                )

    elif names.TAWSS not in arraysInSurface:
        raise ValueError("TAWSS array not in surface!")

    elif names.OSI not in arraysInSurface:
        raise ValueError("OSI array not in surface!")

    fieldsDf = tools.vtkPolyDataToDataFrame(surface)

    # Add int field which will indicate the thicker regions
    # zero indicates normal wall... the aneuysm portion wil be updated
    fieldsDf[names.WallTypeArrayName] = normalWall

    # Groups of conditions
    isAneurysm = fieldsDf[distance_to_neck_array] < const.NeckIsoValue

    isHighWss = fieldsDf[names.TAWSS] > limitHemodynamics[names.TAWSS]["high"]
    isLowWss  = fieldsDf[names.TAWSS] < limitHemodynamics[names.TAWSS]["low"]

    isHighOsi = fieldsDf[names.OSI] > limitHemodynamics[names.OSI]["high"]
    isLowOsi  = fieldsDf[names.OSI] < limitHemodynamics[names.OSI]["low"]

    # isHighRrt = fieldsDf[names.RRT] > limitHemodynamics[names.RRT]["high"]
    # isLowRrt = fieldsDf[names.RRT] < limitHemodynamics[names.RRT]["low"]

    thickerWallCondition = (isAneurysm) & (isLowWss)  & (isHighOsi)# & (isHighRrt)
    thinnerWallCondition = (isAneurysm) & (isHighWss) & (isLowOsi) # & (isLowRrt)

    # Update wall type array
    fieldsDf.loc[thickerWallCondition, names.WallTypeArrayName] = thickerWall
    fieldsDf.loc[thinnerWallCondition, names.WallTypeArrayName] = thinnerWall

    hemodynamicSurfaceNumpy = dsa.WrapDataObject(surface)

    # Add new field to surface
    hemodynamicSurfaceNumpy.CellData.append(
        dsa.VTKArray(fieldsDf[names.WallTypeArrayName]),
        names.WallTypeArrayName
    )

    return hemodynamicSurfaceNumpy.VTKObject

def UpdateAbnormalHemodynamicsRegions(
        vascular_surface: names.polyDataType,
        field_name: str,
        atherosclerotic_factor: float=1.20,
        red_regions_factor: float=0.95
    )   -> names.polyDataType:
    """Update fields on an aneurysm surface based on adjacent hemodynamics."""

    # Factor array: compute WallTypeArrayName if not yet on the surface
    if names.WallTypeArrayName not in tools.GetCellArrays(vascular_surface):
        vascular_surface = WallTypeClassification(vascular_surface)

    npSurface = dsa.WrapDataObject(vascular_surface)

    wallTypeArray = npSurface.GetCellData().GetArray(
                        names.WallTypeArrayName
                    )

    # Add abnormal factor array
    # This is important to have a smooth field to multiply with the
    # thickness array (scale factor can be viewed as a continous
    # distribution in contrast to the WallType array that is discrete)
    abnormalFactorArray = dsa.VTKArray(
                              np.ones(shape=wallTypeArray.shape)
                          )

    # update with scale factors
    abnormalFactorArray[
        wallTypeArray == const.IaWallTypes["AtheroscleroticWall"]
    ] = atherosclerotic_factor

    abnormalFactorArray[
        wallTypeArray == const.IaWallTypes["RedWall"]
    ] = red_regions_factor

    npSurface.CellData.append(
        abnormalFactorArray,
        names.AbnormalFactorArrayName
    )

    vascular_surface = npSurface.VTKObject

    # Interpolate AbnormalFactorArray cell data to point data
    vascular_surface = tools.CellFieldToPointField(
                           vascular_surface,
                           names.AbnormalFactorArrayName
                       )

    npSurface = dsa.WrapDataObject(vascular_surface)

    abnormalFactorArray = npSurface.GetPointData().GetArray(
                              names.AbnormalFactorArrayName
                          )

    fieldToBeUpdated = npSurface.GetPointData().GetArray(
                           field_name
                       )

    npSurface.PointData.append(
        abnormalFactorArray*fieldToBeUpdated,
        field_name
    )

    vascular_surface = npSurface.VTKObject
    vascular_surface.GetCellData().RemoveArray(names.AbnormalFactorArrayName)

    return vascular_surface

def ComputeVasculatureThicknessWithAneurysm(
        vascular_surface: names.polyDataType,
        centerlines: names.polyDataType=None,
        thickness_field_name: str=names.ThicknessArrayName,
        set_uniform_wlr: bool=False,
        uniform_wlr_value: float=const.WlrMedium,
        neck_comp_mode: str="interactive",
        gdistance_to_neck_array_name: str=names.DistanceToNeckArrayName,
        aneurysm_type: str="",
        aneurysm_influence_dist: float=0.5,
        scale_factor: float=0.75,
        parent_vascular_surface: names.polyDataType=None,
        dome_point: tuple=None,
        abnormal_thickness: bool=False,
        atherosclerotic_factor: float=1.20,
        red_regions_factor: float=0.95,
        nsmooth_iterations: float=5
    )   -> names.polyDataType:
    """Calculate and set aneurysm thickness.

    Based on the vasculature thickness distribution, defined as the outside
    portion of the complete geometry from the neck selected by the user,
    estimates an aneurysm thickness by averaging the vasculature thickness
    using as weight function the inverse distance to the
    "aneurysm-influenced" region line. The estimated aneurysm thickness is,
    then, set on the aneurysm surface in the thickness array.

    The aneurysm-influenced neck line is defined as the region between the
    neck line (provided by the user or computed automatically) and the path
    that is at a distance of 'AneurysmInfluencedRegionDistance' value (in
    mm; default 0.5 mm) from the neck line.  This strip around the aneurysm
    is imagined as a region of the original vasculature that had its
    thickness changed by the aneurysm growth.

    If the surface does not already have the 'DistanceToNeckArray' scalar, then
    it will prompt the user to select the neck line, which will be stored on
    the surface. Alternatively, the user may select the option "neck_comp_mode"
    as 'automatic', which estimates a neck line.

    The aneurysm sac thickness may be estimated as 'uniform', the default
    behavior, or using the abnormal wall thickness based on the adjacent
    hemodynamics to the aneurysm wall: the TAWSS and OSI fields (controlled by
    setting the option 'abnormal_thickness' to True). In this last case, the
    passed suface must have these two field from a CFD simulation.

    The aneurysm abnormal thickness is computed based on a 'wall type array',
    and hence increase or deacrease the sac thickness. The procedure is as
    follows: With a global thickness array already defined on the surface,
    update the thickness based on the wall type array created based on the
    hemodynamics variables, by multiplying it by a factor defined below. As
    explained in the function WallTypeCharacterization of wallmotion.py, the
    three types of wall and the operation performed here for each are:

    .. table:: Local wall type characterization
        :widths: auto

        =====   =============== =========
        Label   Wall Type       Operation
        =====   =============== =========
            0   Normal wall     Nothing (default = 1)
            1   Atherosclerotic Increase thickness (default factor = 1.20)
            2   "Red" wall      Decrease thickness (default factor = 0.95)
        =====   =============== =========

    The multiplying factors for the atherosclerotic and red wall must be
    provided, with default values given above. The function will look for the
    array named "WallType" for defining its operation or compute it on the fly.
    """

    # Compute thickness of the vascular tree portion
    vascular_surface = ComputeVasculatureThickness(
                            vascular_surface,
                            centerlines,
                            thickness_field_name=thickness_field_name,
                            set_uniform_wlr=set_uniform_wlr,
                            uniform_wlr_value=uniform_wlr_value
                        )

    # Compute the distance to neck array
    if gdistance_to_neck_array_name not in tools.GetPointArrays(vascular_surface):
        vascular_surface = ComputeGeodesicDistanceToAneurysmNeck(
                               vascular_surface,
                               mode=neck_comp_mode,
                               aneurysm_type=aneurysm_type,
                               parent_vascular_surface=parent_vascular_surface
                           )

    # Surface with thickness and distnce to neck
    npDistanceSurface = dsa.WrapDataObject(vascular_surface)

    # Update both fields with selection
    thicknessArray = npDistanceSurface.GetPointData().GetArray(
                         thickness_field_name
                     )

    distanceToNeckArray = npDistanceSurface.GetPointData().GetArray(
                              gdistance_to_neck_array_name
                          )

    # First compute aneurysm thickness based on vasculature thickness
    # the vasculature is selection value > 0
    onVasculature = distanceToNeckArray > aneurysm_influence_dist

    # Filter thickness and neckScalars
    vasculatureThicknesses = onVasculature*thicknessArray
    vasculatureDistances   = onVasculature*distanceToNeckArray

    # Aneurysm thickness as weighted average
    aneurysmThickness = scale_factor*np.average(
                            vasculatureThicknesses,
                            weights=np.array([
                                1.0/x if x != 0.0 else 0.0
                                for x in vasculatureDistances
                            ])
                        )

    print(
        "Aneurysm thickness computed: {}".format(
            aneurysmThickness
        ),
        end="\n"
    )

    # Then, substitute thickness array by aneurysmThickness
    thicknessArray[vasculatureThicknesses == 0.0] = aneurysmThickness

    vascular_surface = npDistanceSurface.VTKObject

    if abnormal_thickness:
        vascular_surface = UpdateAbnormalHemodynamicsRegions(
                               vascular_surface,
                               field_name=thickness_field_name,
                               atherosclerotic_factor=atherosclerotic_factor,
                               red_regions_factor=red_regions_factor
                           )

    # After array created, smooth it hard
    vascular_surface = tools.SmoothSurfacePointField(
                           vascular_surface,
                           thickness_field_name,
                           niterations=nsmooth_iterations
                       )

    return vascular_surface


# def ComputeVasculatureThicknessWithNAneurysms(
#         vascular_surface: names.polyDataType,
#         centerlines: names.polyDataType=None,
#         thickness_field_name: str=names.ThicknessArrayName,
#         set_uniform_wlr: bool=False,
#         uniform_wlr_value: float=const.WlrMedium,
#         naneurysms: int=1,
#         aneurysm_type: str="",
#         gdistance_to_neck_array_name: str=names.DistanceToNeckArrayName,
#         neck_comp_mode: str="interactive",
#         parent_vascular_surface: names.polyDataType=None,
#         dome_point: tuple=None
#     )   -> names.polyDataType:
#     #this version will account for more than one aneurysm case
#     raise NotImplementedError("Not yet implemented.")

#     # import re
#     # # Check if there is any 'DistanceToNeck<i>' array in points arrays
#     # # where 'i' indicates that more than one aneurysm are present on
#     # # the surface.
#     # r = re.compile(gdistance_to_neck_array_name + ".*")

#     # distanceToNeckArrayNames = list(filter(r.match, pointArrays))

#     # for id_ in range(naneurysms):

#     #     # Update neck array name if more than one aneurysm
#     #     arrayName = gdistance_to_neck_array_name + str(id_ + 1) \
#     #                 if naneurysms > 1 \
#     #                 else gdistance_to_neck_array_name

def ComputeVasculatureElasticityWithAneurysm(
        vascular_surface: names.polyDataType,
        elasticity_field_name: str=names.ElasticityArrayName,
        aneurysm_elasticity_mode: str="uniform",
        arteries_elasticity: float=5e6,
        aneurysm_elasticity: float=2e6,
        neck_comp_mode: str="interactive",
        gdistance_to_neck_array_name: str=names.DistanceToNeckArrayName,
        aneurysm_type: str="",
        parent_vascular_surface: names.polyDataType=None,
        dome_point: tuple=None,
        abnormal_elasticity: bool=False,
        atherosclerotic_factor: float=1.20,
        red_regions_factor: float=0.95,
        nsmooth_iterations: float=5
    )   -> names.polyDataType:
    """Calculate and set aneurysm and vascular elasticity.

    Based on a value for the aneurysm elasticity and the arterial elasticity,
    set them on the vascular surface. The arterial elasticity is considered to
    be uniform, whereas the aneurysm elasticity accepts two modes:

        * 'uniform': uniform elasticity;
        * 'linear': elasticity linearly varying from the arterial value to a
            value set by the user too.

    The neck contour that devides the aneurysm sac is either provided by the
    user or computed automatically, through the array 'DistanceToNeck' that
    marks the neck contour with zero values. If the surface does not already
    have the 'DistanceToNeck' scalar array, then it will prompt the user to
    select the neck line, which will be stored on the surface. Alternatively,
    the user may select the option "neck_comp_mode" as 'automatic', which
    estimates a neck line.

    The option 'abnormal_elasticity' allows for the automatic update of the
    aneurysm elasticity based on the adjacent hemodynamics to the aneurysm
    wall: the TAWSS and OSI fields. In this last case, the passed surface must
    have these two field from a CFD simulation.

    The aneurysm abnormal elasticity is computed based on a 'wall type array',
    and hence increase or deacrease the sac elasticity. The procedure is as
    follows: With a global elasticity array already defined on the surface,
    update the elasticity based on the wall type array created based on the
    hemodynamics variables, by multiplying it by a factor defined below. As
    explained in the function WallTypeCharacterization of wallmotion.py, the
    three types of wall and the operation performed here for each are:

    .. table:: Local wall type characterization
        :widths: auto

        =====   =============== =========
        Label   Wall Type       Operation
        =====   =============== =========
            0   Normal wall     Nothing (default = 1)
            1   Atherosclerotic Increase elasticity (default factor = 1.20)
            2   "Red" wall      Decrease elasticity (default factor = 0.95)
        =====   =============== =========

    The multiplying factors for the atherosclerotic and red wall must be
    provided, with default values given above. The function will look for the
    array named "WallType" for defining its operation or compute it on the fly.
    """

    # Compute the distance to neck array (here serving only as a neck contour)
    if gdistance_to_neck_array_name not in tools.GetPointArrays(vascular_surface):
        vascular_surface = ComputeGeodesicDistanceToAneurysmNeck(
                               vascular_surface,
                               mode=neck_comp_mode,
                               aneurysm_type=aneurysm_type,
                               parent_vascular_surface=parent_vascular_surface
                           )

    # Surface with thickness and distnce to neck
    npDistanceSurface = dsa.WrapDataObject(vascular_surface)

    distanceArray = npDistanceSurface.PointData.GetArray(
                        gdistance_to_neck_array_name
                    )

    # Array to hold the actual elasticity array
    elasticities = dsa.VTKArray(
                        np.zeros(
                            shape=vascular_surface.GetNumberOfPoints()
                        )
                    )

    # Mark regions based on distance array values
    onAneurysm  = distanceArray <= 0.0
    outAneurysm = distanceArray > 0.0

    elasticities[outAneurysm] = arteries_elasticity

    # One single aneurysm expected here
    if aneurysm_elasticity_mode == "uniform":

        elasticities[onAneurysm] = aneurysm_elasticity

    elif aneurysm_elasticity_mode == "linear":

        # Fundus and neck elasticity
        neckElasticity   = arteries_elasticity
        fundusElasticity = aneurysm_elasticity

        # Distances on the aneurysm are negative: max distance is actually min
        maxDistance = -min(distanceArray)

        # Angular coeff. for linear elasticity on the aneurysm sac
        angCoeff = \
            (neckElasticity - fundusElasticity)/maxDistance

        elasticities[onAneurysm] = \
            dsa.VTKArray([
                neckElasticity + angCoeff*distance
                for distance in distanceArray[onAneurysm]
            ])

    else:
        raise ValueError(
                  """Aneurysm elasticity mode either 'uniform'
                  or 'linear'. {} passed.""".format(
                      aneurysm_elasticity_mode
                  )
              )

    npDistanceSurface.PointData.append(
        elasticities,
        elasticity_field_name
    )

    vascular_surface = npDistanceSurface.VTKObject

    if abnormal_elasticity:
        vascular_surface = UpdateAbnormalHemodynamicsRegions(
                               vascular_surface,
                               field_name=elasticity_field_name,
                               atherosclerotic_factor=atherosclerotic_factor,
                               red_regions_factor=red_regions_factor
                           )

    # After array created, smooth it hard to remove discontinuity
    vascular_surface = tools.SmoothSurfacePointField(
                           vascular_surface,
                           elasticity_field_name,
                           niterations=nsmooth_iterations
                       )

    return vascular_surface

# PUT THIS ONE INTP THE Vascular Surface or ICA Surface Class Class
def SplitICAModelIntoBends(
        vascular_model: names.polyDataType,
        centerlines: names.polyDataType,
        bif_point: tuple
    )   -> names.polyDataType:
    """Split ICA vascular surface model into bends based on curvature and 
    torsion."""

    # Split centerlines into bends
    bendsCenterlines = SplitICACenterlineIntoBends(
                            centerlines,
                            bif_point
                        )

    return tools.ProjectPointArray(
                vascular_model,
                bendsCenterlines,
                names.BendsIdsFieldName
            )

def ClipVasculatureOffBifurcation(
        vascular_surface: names.polyDataType,
        centerlines: names.polyDataType,
        clip_vessel_field: str=names.vmtkAbscissasArrayName,
        inlet_vessel_clip_value: float=-40.0,
        outlet_vessel_clip_value: str=8.0,
        bif_point: tuple=None,
        aneurysm_point: tuple=None
    )   -> names.polyDataType:
    """Automatically clip a vessel structure away of a specified bifurcation
    and an aneurysm, if it exists.

    Given a vessel surface model and its centerlines, clip the vessel at points
    espefied by a distance away from a selected bifurcation and away from the
    bifurcation closer to a specified aneurysm. The function is intended to
    clip inlet and outlet boundary conditions in vascular models based a
    predefined distance from a bifurcation and an aneurysm. For example, you
    can select the ICA bifurcation and pass two values identified as the
    distance fom the ICA bifurcation where the vascular model surface will be
    clipped, if no aneurysm on the model. If an aneurysm is present and you
    want to clip after the aneurysm, you can pass the ICA bifurcation point and
    the aneurysm dome point (if not passed, the function prompts the user to
    interactively select it), so the function clips the vasculature after the
    aneurysm bifurcation.

    Both the clip values for inlet (before the bifurcation) as the clip value
    of outlets (after the bifurcation can be passed. The default field used to
    clip is the 'Abscissas' field of distance values along the centerline with
    origin its closest point to the selected bifurcation. If an aneurysm
    exists, then the clip arg. 'outlet_vessel_clip_value' is relative to the
    bifurcation closest to the aneurysm.

    The point closest to the selected bifurcation can be passed via the arg.
    'bif_point'. If None, the user is prompted to interactively select it.
    """

    if bif_point is None:

        bif_point = tools.SelectSurfacePoint(
                        vascular_surface,
                        input_text="Select point at the ICA bifurcation\n"
                    )

    # Compute centerlines abscissas and other attributes
    offsetCenterlines = cl.ComputeCenterlinePropertiesOffBifurcation(
                            centerlines,
                            bif_point
                        )

    # Computing the bifurcation reference system
    referenceSystems = cl.CenterlineReferenceSystems(offsetCenterlines)

    # Get ICA-MCA-ACA bifurcation
    # Get closest point to ICA at the bifurcation found
    bifGroupId  = int(
                      tools.GetFieldValueAtClosestPoint(
                          referenceSystems,
                          bif_point,
                          names.vmtkGroupIdsArrayName
                      )
                  )

    # Identify the bifurcation GroupId and its Abscissas closer to the aneurysm
    if aneurysm_point is None:

        aneurysm_point = tools.SelectSurfacePoint(
                             vascular_surface,
                             input_text="Select a  point on the aneurysm surface"
                         )

    # Get center of bifurcation closest to aneurysm point
    iaClosestPointToBif = tools.LocateClosestPointOnPolyData(
                                referenceSystems,
                                aneurysm_point
                            )

    onlyBifurcations = tools.ExtractPortion(
                           offsetCenterlines,
                           # only centerlines potions of bifs.
                           names.vmtkBlankingArrayName,
                           const.one
                       )

    # Get also the group id of the portion where the aneurysm is
    iaAbscissasClosestBif = tools.GetFieldValueAtClosestPoint(
                                  onlyBifurcations,
                                  iaClosestPointToBif,
                                  names.vmtkAbscissasArrayName
                            )

    # Get also the group id of the portion where the aneurysm is
    iaGroupIdClosestBif = tools.GetFieldValueAtClosestPoint(
                                onlyBifurcations,
                                aneurysm_point,
                                names.vmtkGroupIdsArrayName
                          )

    # Check whether passed clip values are within the clip field range
    offsetAbscissasRange = offsetCenterlines.GetPointData().GetArray(
                                names.vmtkAbscissasArrayName
                            ).GetRange()

    if inlet_vessel_clip_value < min(offsetAbscissasRange):

        raise ValueError(
                "{} smaller than min of {} (~ {}).\nSpecify higher value.".format(
                    inlet_vessel_clip_value,
                    clip_vessel_field,
                    round(min(offsetAbscissasRange), 3)
                  )
              )

    if outlet_vessel_clip_value + iaAbscissasClosestBif > max(offsetAbscissasRange):

        raise ValueError(
                "{} distance relative to aneurysm bifurcation is larger than max of {} (~ {}).\nSpecify smaller value.".format(
                    outlet_vessel_clip_value,
                    clip_vessel_field,
                    round(max(offsetAbscissasRange), 3)
                  )
              )

    # Cretae dict to better storing of separate centerlines
    individualCenterlines = cl.SplitCenterlineObject(offsetCenterlines)

    # Get longest centerline
    # ID of ICA clip will be identified in this portion
    idLongestCenterline = max(
                                individualCenterlines,
                                key=lambda idx: individualCenterlines[idx]["length"]
                            )

    longestCenterline = individualCenterlines[idLongestCenterline]["object"]
    npLongestCenterline = dsa.WrapDataObject(longestCenterline)

    # Clip inlet artery
    clipArray = npLongestCenterline.PointData.GetArray(clip_vessel_field)

    # Get id of the point where to clip
    icaClipPointId = (
                        np.abs(clipArray - inlet_vessel_clip_value)
                     ).argmin()

    icaClipPoint = tuple(npLongestCenterline.Points[icaClipPointId])

    icaClipNormal = tuple(
                        npLongestCenterline.PointData.GetArray(
                            names.vmtkFrenetTangentArrayName
                        )[icaClipPointId]
                    )

    # Clip vessel at inlet location
    vascular_surface = ClipVasculatureWithPlane(
                           vascular_surface,
                           plane_center=icaClipPoint,
                           plane_normal=icaClipNormal
                       )

    # Store the group id to whcih the point belong
    outletClipPoints = {}

    for cl_id, dict_ in individualCenterlines.items():

        npClPortion = dsa.WrapDataObject(dict_["object"])
        clClipField = npClPortion.PointData.GetArray(clip_vessel_field)

        clGroups = npClPortion.PointData.GetArray(
                        names.vmtkGroupIdsArrayName
                    )

        clGroupIdsList = list(set(clGroups))

        # Check whether the centerline have abscissas
        if bifGroupId in clGroupIdsList:

            # If the ica bif. and aneurysm group ids are on the centerline,
            # then use the updated clip value to clip (meaning to clip AFTER
            # the aneurysm abscissa)
            if iaGroupIdClosestBif in clGroupIdsList and \
               iaGroupIdClosestBif > bifGroupId:

                clipValue = iaAbscissasClosestBif + \
                            outlet_vessel_clip_value

            else:
                clipValue = outlet_vessel_clip_value

            # Use the GroupIds as keys to dict to avoid repetitive entries
            outletClipPointId = (
                                    np.abs(
                                        clClipField - clipValue
                                    )
                                ).argmin()

            outletClipPoint  = tuple(npClPortion.Points[outletClipPointId])
            outletClipNormal = tuple(
                                   npClPortion.PointData.GetArray(
                                       names.vmtkFrenetTangentArrayName
                                   )[outletClipPointId]
                               )

            outletClipPoints.update({
                clGroups[outletClipPointId]: {
                    "center": outletClipPoint,
                    "normal": outletClipNormal
                }
            })

        else:
            continue

    for dict_ in outletClipPoints.values():

        # Clip at outlets: invert normal at these positions
        vascular_surface = ClipVasculatureWithPlane(
                               vascular_surface,
                               plane_center=dict_["center"],
                               plane_normal=tuple(-val for val in dict_["normal"])
                           )

    return tools.CleanupArrays(vascular_surface)

def ComputeAneurysmSacRegions(
        vascular_surface: names.polyDataType,
        distance_to_neck_array_name: str=names.DistanceToNeckArrayName,
        sac_regions_array_name: str=names.SacRegionsArrayName,
        neck_to_body_fraction: float=0.2,
        body_to_dome_fraction: float=0.6
    )   -> names.polyDataType:
    """Compute the aneurysm sac regions based on the distance to neck array.

    This functions splits an intracranial aneurysm sac into three regions
    so-called "dome", "neck", and "body". These denominations are typically
    employed by neurosurgeons to split an aneurysm sac into very distinct
    patches. Although commonly employed in the medical practice, no formal
    mathematical deﬁnition of it exists, therefore, the one
    proposed and used by Salimi Ashkezari et al. in their paper:

        S. F. Salimi Ashkezari et al., “Blebs in intracranial aneurysms:
        prevalence and general characteristics,” J NeuroIntervent Surg, vol.
        13, no. 3, pp. 226–230, Mar. 2021, doi:
        10.1136/neurintsurg-2020-016274.

    is implemented here. It defines each region based on the geodesic distance
    to the neck contour: given the maximum geodesic distance to the aneurysm
    neck within the aneurysm, the neck is defined as the region within 20% of
    this distance, the body is defined as the region between 20% and 60% of
    this distance, and the dome is defined as the region between 60% and 100%
    of this distance. The rest of the aneurysm sac is considered out of the
    sac. These value can be adjusted by the user through the arguments
    'neck_to_body_value' and 'body_to_dome_value'.

    The vascular surface model must be provided with the distance to neck
    field defined on it. This can be created with the function
    "ComputeGeodesicDistanceToAneurysmNeck" of this module. The name of the
    field can also be passed as argument.

    The output is the same surface model with a new array called "SacRegions"
    (module 'names.SacRegionsArrayName') where regions are identified by the
    code:

    .. table:: Aneurysm sac regions
        :widths: auto

        =====   ===============
        Label   Regions
        =====   ===============
            0   Out of the sac
            1   Neck
            2   Body
            3   Dome
        =====   ===============

    and these values are defined in the dictionary
    'constants.IaSacRegionsTypes'.
    """

    # Interpolate the distance to neck aray to cell data
    surface = tools.PointFieldToCellField(
                  vascular_surface,
                  distance_to_neck_array_name
              )

    # Use numpy interface
    npSurface = dsa.WrapDataObject(surface)

    distanceToNeckArray = npSurface.GetCellData().GetArray(
                              distance_to_neck_array_name
                          )

    # Get the minimum value (negative values lie on the aneurysm sac)
    iaMaxGeodesicDistance = min(distanceToNeckArray)

    # Create new array based on:
    # > 0 -> out of sac -> 0
    # 0 <= distance <= abs(0.2*maxGeoDist) -> Neck -> 1
    # abs(0.2*maxGeoDist) < distance <= abs(0.6*maxGeoDist) -> Body -> 2
    # abs(0.6*maxGeoDist) < distance <= abs(maxGeoDist) -> Dome -> 3
    sacRegionArray = np.copy(distanceToNeckArray)

    # Get dome region
    sacRegionArray[
        np.logical_and(
            distanceToNeckArray >= iaMaxGeodesicDistance,
            distanceToNeckArray <  body_to_dome_fraction*iaMaxGeodesicDistance
        )
    ] = const.IaSacRegionsTypes["Dome"]

    # Get body region
    sacRegionArray[
        np.logical_and(
            distanceToNeckArray >= body_to_dome_fraction*iaMaxGeodesicDistance,
            distanceToNeckArray <  neck_to_body_fraction*iaMaxGeodesicDistance
        )
    ] = const.IaSacRegionsTypes["Body"]

    # Get neck region
    sacRegionArray[
        np.logical_and(
            distanceToNeckArray >= neck_to_body_fraction*iaMaxGeodesicDistance,
            distanceToNeckArray <= const.zero
        )
    ] = const.IaSacRegionsTypes["Neck"]

    # Get out of sac region
    sacRegionArray[
        distanceToNeckArray > const.zero
    ] = const.IaSacRegionsTypes["OutOfSac"]

    npSurface.CellData.append(
        sacRegionArray,
        sac_regions_array_name
    )

    return npSurface.VTKObject
