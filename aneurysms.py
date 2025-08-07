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

"""Collection of tools to characterize cerebral aneurysms.

The idea behind this library is to provide tools to manipulate and to model
the surface of cerebral aneurysms on a patient-specific vasculature, with
functions to compute its morphological parameters.
"""

import sys
import vtk
import numpy as np
from vtkmodules.numpy_interface import dataset_adapter as dsa

from vmtk import vtkvmtk
from vmtk import vmtkscripts
from scipy.spatial import ConvexHull

# Local modules
from vmtk4aneurysms.lib import names
from vmtk4aneurysms.lib import constants as const
from vmtk4aneurysms.lib import polydatatools as tools
from vmtk4aneurysms.lib import polydatageometry as geo
from vmtk4aneurysms.lib import polydatamath as pmath

from vmtk4aneurysms.vascular_classes import (
    VascularSurface,
    VascularTree
)

from vmtk4aneurysms.neck_extractor import (
    ComputeSacCenterlinePoints,
    ClipAneurysmSacSurface,
    ComputeGeodesicDistanceToAneurysmNeck,
    InteractiveNeckIdentification,
    Automatic3DNeckIdentification
)

from abc import ABC, abstractmethod

def _simple_cap(surface):
    """Cap a surface with an open profile with a simple centerpoint
    triangulation.

    Used to compute close the aneurysm and aneurysm convex hull surfaces with
    the same cap to measure their volume. Also computes the outwards normals.
    """

    capper = vtkvmtk.vtkvmtkCapPolyData()
    capper.SetInputData(surface)
    capper.SetDisplacement(0.0)
    capper.SetInPlaneDisplacement(0.0)
    capper.SetCellEntityIdsArrayName(names.CellEntityIdsArrayName)
    capper.SetCellEntityIdOffset(-1) # The cap surface will be 0
    capper.Update()

    return geo.Surface.Normals(
               capper.GetOutput(),
               auto_orient_if_closed=True
           )


def GenerateOstiumSurface(
        aneurysm_sac_surface: names.polyDataType,
        compute_normals: bool=True
    )   -> names.polyDataType:
    """ Generate an ostium surface based on the aneurysm neck.

    The ostium surface, by definition, is the imaginary surface that 'closes'
    the aneurysm neck. This functions estimates this surface by using the
    'smooth' capping method from vtkvtmk. It 'caps' the aneruysm sac with a
    surface that is smooth and, then, extracts it. The algorithm finally
    remeshes is for a better quality surface , but keeping its boundary (the
    neck contour) intact.

    The outward normals to the ostium surface may be optionally added through
    the option 'compute_normals'.
    """

    # Close the aneurysm with the 'smooth' method, which was the best to fit a
    # generic 3D contour
    capper = vtkvmtk.vtkvmtkSmoothCapPolyData()
    capper.SetInputData(aneurysm_sac_surface)

    # It is important to set cnostraint to zero to have 90 degrees angles on
    # corners
    capper.SetConstraintFactor(0.0)
    capper.SetNumberOfRings(15)
    capper.SetCellEntityIdsArrayName(names.CellEntityIdsArrayName)
    capper.SetCellEntityIdOffset(-1) # The neck surface will be 0
    capper.Update()

    triangulate = vtk.vtkTriangleFilter()
    triangulate.SetInputData(capper.GetOutput())
    triangulate.PassLinesOff()
    triangulate.PassVertsOff()
    triangulate.Update()

    surface = geo.Surface.Normals(
                  triangulate.GetOutput(),
                  auto_orient_if_closed=True
              ) if compute_normals else triangulate.GetOutput()

    # Get maximum id of the surfaces
    ostiumId = max(
                   surface.GetCellData().GetArray(
                       names.CellEntityIdsArrayName
                   ).GetRange()
               )

    ostiumSurface = tools.ExtractPortion(
                        surface,
                        names.CellEntityIdsArrayName,
                        ostiumId
                    )

    # Removed remeshing as it was complicating the Normals addition with the
    # correct orientation
    # Remesh: the smooth capping may add too deformed cells
    # ostiumSurface = tools.RemeshSurface(
    #                     tools.UnsGridToPolyData(ostiumSurface)
    #                 )

    # Add a little bit of smoothing
    # ostiumSurface = tools.SmoothSurface(ostiumSurface)

    return ostiumSurface

# Wallmotion-related functions that operate only on the aneurysm surface
# TODO: Refactored here so canbe used directly inside the Aneurysm class
# this will depend on a better understanding of how I will include the
# wall motion fields
def AneurysmPulsatility(
        displacement_surface: names.polyDataType,
        ps_displ_field_name: str,
        ld_displ_field_name: str,
        aneurysm_neck_array_name: str=names.DistanceToNeckArrayName
    )   -> float:
    """Return an aneurysm's wall pulsatility.

    The pulsatility, :math:`\delta_v`, of a cerebral aneurysm is defined as
    (Sanchez et al. (2014)):

    .. math::
        \delta_v = (V_{ps}/V_{ld}) - 1

    where "ld" indicates low diastole and "ps" indicates peak systole values,
    and V is the aneurysm sac volume. It uses the the lumen surface with the
    peak systole and low diastole displacement field on the surface.

    .. note::
        The input surface must alread have the aneurysm neck array, otherwise
        the function prompts the user to select the aneurysm neck contour.
    """

    # Warp whole surface at peak systole and low diastole
    ldLumenSurface = geo.WarpPolydata(displacement_surface,
                                      ld_displ_field_name)

    psLumenSurface = geo.WarpPolydata(displacement_surface,
                                      ps_displ_field_name)

    # With the surfaces warped by the displacement field,
    # now we just need to clip the aneurysm sac region.
    ldAneurysmSurface = tools.ClipWithScalar(ldLumenSurface,
                                             aneurysm_neck_array_name,
                                             const.NeckIsoValue)

    psAneurysmSurface = tools.ClipWithScalar(psLumenSurface,
                                             aneurysm_neck_array_name,
                                             const.NeckIsoValue)

    ldAneurysm = SaccularAneurysm(ldAneurysmSurface)
    psAneurysm = SaccularAneurysm(psAneurysmSurface)

    return psAneurysm.GetAneurysmVolume()/ldAneurysm.GetAneurysmVolume() - 1.0

def AneurysmPulsatility2(
        lumen_surface: names.polyDataType,
        displacement_over_time: dict,
        peak_systole_instant: float,
        low_diastole_instant: float,
        aneurysm_neck_array_name: str=names.DistanceToNeckArrayName
    )   -> float:
    """Compute aneurysm wall pulsatility (alternative version).

    Alternative version of the aneurysm pulsatility computation by using the
    lumen surface and the dictionary with the displacement field computed with
    the GetPatchFieldOverTime function. The input surface must alread have the
    aneurysm neck array, otherwise the function prompts the user to select the
    aneurysm neck contour.
    """

    ldDisplField = displacement_over_time.get(low_diastole_instant)
    psDisplField = displacement_over_time.get(peak_systole_instant)

    # Add both field to the surfaces, separately
    npLumenSurface = dsa.WrapDataObject(lumen_surface)
    npLumenSurface.GetCellData().append(ldDisplField, lowDiastoleDisplFieldName)
    npLumenSurface.GetCellData().append(psDisplField, peakSystoleDisplFieldName)

    lumenSurface = npLumenSurface.VTKObject

    # Project aneurysm neck contour to the surface
    if aneurysm_neck_array_name not in tools.GetPointArrays(lumenSurface):
        print("Neck array name not in surface. Computing it.")

        lumenSurface = ComputeGeodesicDistanceToAneurysmNeck(
                           lumenSurface,
                           mode="interactive"
                       )

    else:
        pass

    # Warp whole surface at peak systole and low diastole
    ldLumenSurface = geo.WarpPolydata(lumenSurface, lowDiastoleDisplFieldName)
    psLumenSurface = geo.WarpPolydata(lumenSurface, peakSystoleDisplFieldName)

    # Clip aneurysm
    ldAneurysmSurface = tools.ClipWithScalar(ldLumenSurface,
                                             aneurysm_neck_array_name,
                                             const.NeckIsoValue)

    psAneurysmSurface = tools.ClipWithScalar(psLumenSurface,
                                             aneurysm_neck_array_name,
                                             const.NeckIsoValue)

    # Initiate aneurysm model
    ldAneurysm = SaccularAneurysm(ldAneurysmSurface)
    psAneurysm = SaccularAneurysm(psAneurysmSurface)

    # Compute pulsatility
    return psAneurysm.GetAneurysmVolume()/ldAneurysm.GetAneurysmVolume() - 1.0

def WallTypeClassification(
        surface: names.polyDataType,
        low_wss: float=5.0,
        high_wss: float=10.0,
        low_osi: float=0.001,
        high_osi: float=0.01,
        aneurysmal_region_field_name: str=names.AneurysmalRegionArrayName,
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

    if aneurysmal_region_field_name not in arraysInSurface:
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
    isAneurysm = fieldsDf[aneurysmal_region_field_name] < const.NeckIsoValue

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

class SaccularAneurysm:
    """Representation for saccular cerebral aneurysms.

    Given a saccular aneurysm surface, i.e. delimited by its neck contour (be
    it a plane neck or a 3D contour), as a vtkPolyData object, return a
    computational representation of the aneurysm with its geometrical and
    morphological parameters, listed below:

    1D Size Metrics
    ===============

        - Maximum Diameter
        - Maximum Normal Height
        - Neck Diameter

    3D Size Metrics
    ===============

        - Aneurysm Surface Area
        - Aneurysm Volume
        - Convex Hull Surface Area
        - Convex Hull Volume
        - Ostium Surface Area

    2D Shape Metrics
    ================

        - Aspect Ratio
        - Bottleneck Factor
        - Conicity Parameter

    3D Shape Indices
    ================

        - Ellipticity Index
        - Non-sphericity Index
        - Undulation Index
        - Curvature-based indices: GAA, MAA, MLN, GLN

    Note: the calculations of aneurysm parameters performed here were orignally
    defined for a plane aneurysm neck, and based on the following works:

        [1] Ma B, Harbaugh RE, Raghavan ML. Three-dimensional geometrical
        characterization of cerebral aneurysms. Annals of Biomedical
        Engineering.  2004;32(2):264–73.

        [2] Raghavan ML, Ma B, Harbaugh RE. Quantified aneurysm shape and
        rupture risk. Journal of Neurosurgery. 2005;102(2):355–62.

    Nonetheless, the computations will still occur for a generic 3D neck
    contour. In this case, the 'ostium surface normal' is defined as the
    vector-averaged normal of the ostium surface, a triangulated surface
    created by joining the points of the neck contour and its barycenter.

    .. warning::
        The  input aneurysm surface must be open for correct computations.
    """

    def __init__(
            self,
            surface
        ):
        """Initiates aneurysm model.

        Given the aneurysm surface (vtkPolyData) initiates aneurysm model by
        computing simple size features: surface area, ostium surface area, and
        volume.

        Arguments:
            surface (vtkPolyData) -- the aneurysm surface
        """
        self._neck_index = int(const.zero)

        self._aneurysm_surface = tools.Cleaner(surface)
        self._neck_contour = self._compute_neck_contour()

        self._ostium_surface = GenerateOstiumSurface(
                                   self._aneurysm_surface,
                                   compute_normals=True
                               )

        self._ostium_normal_vector = pmath.SurfaceAverage(
                                         self._ostium_surface,
                                         names.normals
                                     )

        # Compute ostium surface area
        # Compute areas...
        self._surface_area = geo.Surface.Area(self._aneurysm_surface)
        self._ostium_area = geo.Surface.Area(self._ostium_surface)

        # ... and volume
        self._volume = geo.Surface.Volume(
                           _simple_cap(self._aneurysm_surface)
                       )

        # Computing hull surface and properties
        self._compute_aneurysm_convex_hull()

        # 1D size definitions
        self._neck_diameter = self._compute_neck_diameter()

        # Computes the maximum normal height and dome point
        self._dome_point = None
        self._max_normal_height = None
        self._compute_max_normal_height_vector_and_dome_point()

        self._max_diameter, self._bulge_height = self._compute_max_diameter()

        # Other metrics
        self._aspect_ratio        = self._max_normal_height/self._neck_diameter
        self._bottleneck_factor   = self._max_diameter/self._neck_diameter
        self._conicity_parameter  = 0.5 - self._bulge_height/self._max_normal_height
        self._nonsphericity_index = self._compute_non_sphericity_index()
        self._ellipticity_index   = self._compute_ellipticity_index()
        self._undulation_index    = 1.0 - self._volume/self._hull_volume

        # Compute curvature metrics: GAA, MAA, MLN, GLN
        self._compute_curvature_metrics()

        # The sac centerline will be computed later
        self._sac_centerline = None

    def __repr__(self):

        return f"Aneurysm surface representation."

    def _cap_aneurysm(self):
        """Cap aneurysm with the computed ostium surface.

        Return the aneurysm surface 'capped', i.e. with a surface covering the
        neck region. The surface is the same created as the ostium surface
        using the smooth method.
        """

        appendFilter = vtk.vtkAppendPolyData()
        appendFilter.AddInputData(self._aneurysm_surface)
        appendFilter.AddInputData(self._ostium_surface)
        appendFilter.Update()

        return appendFilter.GetOutput()

    def _compute_aneurysm_convex_hull(self):
        """Compute convex hull of closed surface.

        Given an open surface, compute the convex hull set of a surface and
        returns a triangulated surface representation of it.  It uses
        internally the scipy.spatial package.
        """

        # Get vertices only
        npIaSurface = dsa.WrapDataObject(self._aneurysm_surface)

        # Compute convex hull of points
        surfaceHull = ConvexHull(npIaSurface.GetPoints())

        # Build poly data for convex hull
        hullSurface = tools.BuildPolyData(
                          surfaceHull.points,
                          surfaceHull.simplices
                      )

        # The hull is closed at this point
        hullSurface = geo.Surface.Normals(
                          hullSurface,
                          auto_orient_if_closed=True
                      )

        # Best alternatve so far: to compute the signed distance beteen the
        # hull CELL CENTERS and the ostium
        distanceToOstiumArrayName = "DistanceVectors"

        # Extract hull cell centers
        hullCellCenters = vtk.vtkCellCenters()
        hullCellCenters.SetInputData(hullSurface)
        hullCellCenters.VertexCellsOn()
        hullCellCenters.Update()

        # Needs to compute the normals before if signed array is required
        # Update that in v4a
        surfaceDistance = vtkvmtk.vtkvmtkSurfaceDistance()
        surfaceDistance.SetInputData(hullCellCenters.GetOutput())
        surfaceDistance.SetReferenceSurface(self._ostium_surface)
        surfaceDistance.SetDistanceVectorsArrayName(
            distanceToOstiumArrayName
        )
        surfaceDistance.Update()

        npHullCellDistance = dsa.WrapDataObject(surfaceDistance.GetOutput())

        distanceVectors = npHullCellDistance.GetPointData().GetArray(
                              distanceToOstiumArrayName
                          )

        # The size of this array is the same as the cell of the hull
        ostiumSideArray = dsa.VTKArray(
                                [vtk.vtkMath.Dot(
                                    dVector,
                                    self._ostium_normal_vector
                                )   for dVector in distanceVectors]
                            )

        # Further filter: remove cells that have all points on boundary
        hullContourIds = tools.GetClosestContourOnSurface(
                                hullSurface,
                                self._neck_contour
                            )

        hullContourIdsArr = np.array([hullContourIds.GetId(idx)
                                      for idx in range(hullContourIds.GetNumberOfIds())])

        idInContour = np.asarray(
                            [not np.all([idx in hullContourIdsArr
                              for idx in simplex])
                              for simplex in surfaceHull.simplices]
                        )

        # Now get the cells IDS where OstiumSide > 0
        # Note: = 0  means any cells that lie on the ostium surface
        self._hull_surface = tools.BuildPolyData(
                                 surfaceHull.points,
                                 surfaceHull.simplices[
                                     np.logical_and(
                                         ostiumSideArray > 0,
                                         idInContour
                                     )
                                 ]
                             )

        # self._hull_cap = tools.BuildPolyData(
        #                       surfaceHull.points,
        #                       surfaceHull.simplices[ostiumSideArray <= 0]
        #                   )

        # Split the aneurysm hull by the neck surface
        # the result is if the hull 'started' by the neck contour
        self._hull_surface_area = geo.Surface.Area(self._hull_surface)

        # Compute volume (capped hull)
        self._hull_volume = geo.Surface.Volume(
                                _simple_cap(self._hull_surface)
                            )

    def _compute_neck_contour(self):
        """Return boundary of aneurysm surface (== neck contour)"""
        boundaryExtractor = vtkvmtk.vtkvmtkPolyDataBoundaryExtractor()
        boundaryExtractor.SetInputData(self._aneurysm_surface)
        boundaryExtractor.Update()

        return boundaryExtractor.GetOutput()

    def _neck_barycenter(self):
        """Return the neck contour barycenter as a Numpy array."""

        # Get neck contour
        return geo.ContourBarycenter(self._neck_contour)

    def _compute_max_normal_height_vector_and_dome_point(self):
        """Compute vector along the maximum normal height and its corresponding
        point.

        Compute the vector from the neck contour barycenter and the farthest
        point on the aneurysm surface that have the maximum normal distance
        from the ostium surface normal. The farthest point is interpreted as a
        dome point that may identify the aneurysm in a vasculature, for
        example.
        """

        vecNormal = -1*np.array(self._ostium_normal_vector)
        barycenter = np.array(self._neck_barycenter())

        # Get distance between every point and store it as dict to get maximum
        # distance later
        npAneurysmSurface = dsa.WrapDataObject(self._aneurysm_surface)

        # Build lists of normal distances and distance vectors for each
        # aneurysm vertex
        distanceVectors = np.array([
                              np.subtract(vertex, barycenter)
                              for vertex in npAneurysmSurface.GetPoints()
                          ])

        normalDistances = np.array([
                              abs(
                                  vtk.vtkMath.Dot(
                                      distVector,
                                      vecNormal
                                  )
                              )
                              for distVector in distanceVectors
                          ])

        self._max_normal_height = max(normalDistances)
        self._dome_point = npAneurysmSurface.GetPoints()[normalDistances.argmax()]
        self._max_normal_height_vector = distanceVectors[normalDistances.argmax()]

    # 1D Size Indices
    def _compute_neck_diameter(self):
        """Return the neck diameter.

        Compute neck diameter, defined as twice the averaged distance between
        the neck contour barycenter and each point on the neck contour.
        """

        return geo.ContourAverageDiameter(self._neck_contour)

    def _compute_max_diameter(self):
        """Find the maximum diameter of aneurysm sections.

        Compute the diameter of the maximum section, defined as the maximum
        diameter of the aneurysm cross sections that are parallel to the ostium
        surface, i.e. along the ostium normal vector. Returns a tuple with the
        maximum diameter and the bulge height, i.e. the distance between the
        neck barycenter and the location of the largest section, along a normal
        line to the ostium surface.
        """

        # Compute neck contour barycenter and normal vector
        normal = -1.0*np.array(self._ostium_normal_vector)
        barycenter = np.array(self._neck_barycenter())

        # Get maximum normal height
        Hnmax = self._max_normal_height

        # Form points of perpendicular line to neck plane
        nPoints = int(const.oneHundred)*int(const.ten)
        dimensions = int(const.three)

        t = np.linspace(const.zero, Hnmax, nPoints)

        parameters = np.array([t]*dimensions).T

        # Point along line (negative because normal vector is outwards)
        points = [tuple(point)
                  for point in barycenter + parameters*normal]

        # Collect contour of sections to avoid using if inside for
        # Also use the points along the search line to identify the
        # bulge position
        planeContours = dict(zip(
                            points,
                            map(
                                lambda point: tools.ContourCutWithPlane(
                                                  self._aneurysm_surface,
                                                  point,
                                                  normal
                                              ),
                                points
                            )
                        ))

        # Get contours that actually have cells
        planeContours = dict(
                            filter(
                                lambda pair: pair[1].GetNumberOfCells() > 0,
                                planeContours.items()
                            )
                        )

        # Compute diameters and get the maximum
        diameters = {point: geo.ContourHydraulicDiameter(contour)
                     for point, contour in planeContours.items()}

        # Get the max. diameter location (bulge location)
        bulgeLocation = np.array(max(diameters, key=diameters.get))

        # Compute bulge height
        bulgeHeight = geo.Distance(bulgeLocation, barycenter)

        # Find maximum
        maxDiameter = max(diameters.values())

        return maxDiameter, bulgeHeight

    def _compute_non_sphericity_index(self) -> float:

        factor = (18*const.pi)**(1.0/3.0)

        area = self._surface_area
        volume = self._volume

        return const.one - (factor/area)*(volume**(2./3.))

    def _compute_ellipticity_index(self) -> float:

        factor = (18*const.pi)**(1./3.)

        area = self._hull_surface_area
        volume = self._hull_volume

        return const.one - (factor/area)*(volume**(2./3.))

    def _compute_curvature_metrics(self):
        # Get arrays on the aneurysm surface
        arrayNames = tools.GetCellArrays(self._aneurysm_surface)

        # Check if there is any curvature array on the aneurysm surface
        if not all(array in arrayNames
                   for array in [names.MeanCurvatureArrayName,
                                 names.GaussCurvatureArrayName]):

            # TODO: find a procedure to remove points close to boundary
            # of the computation
            warningMessage = "Warning! I did not find any of the necessary " \
                             "curvature arrays on the surface.\nI will "     \
                             "compute them for the aneurysm surface, but "   \
                             "mind that the curvature values close to the "  \
                             "surface boundary are not correct and may "     \
                             "impact the curvature metrics.\n"

            print(warningMessage)

            # Compute curvature arrays for aneurysm surface
            curvatureSurface = geo.Surface.Curvatures(self._aneurysm_surface)
        else:
            curvatureSurface = self._aneurysm_surface

        # Get surface area
        surfaceArea = geo.Surface.Area(curvatureSurface)

        # Add the squares of Gauss and mean curvatures
        npCurvSurface = dsa.WrapDataObject(curvatureSurface)

        arrGaussCurv = npCurvSurface.CellData.GetArray(names.GaussCurvatureArrayName)
        arrMeanCurv  = npCurvSurface.CellData.GetArray(names.MeanCurvatureArrayName)

        npCurvSurface.CellData.append(
            arrGaussCurv**2,
            names.SqrGaussCurvatureArrayName
        )

        npCurvSurface.CellData.append(
            arrMeanCurv**2,
            names.SqrMeanCurvatureArrayName
        )

        curvatureSurface = npCurvSurface.VTKObject

        self._GAA = pmath.SurfaceAverage(
                    curvatureSurface,
                    names.GaussCurvatureArrayName
                )

        self._MAA = pmath.SurfaceAverage(
                    curvatureSurface,
                    names.MeanCurvatureArrayName
                )

        surfIntSqrGaussCurv = surfaceArea*pmath.SurfaceAverage(
                                curvatureSurface,
                                names.SqrGaussCurvatureArrayName
                            )
        surfIntSqrMeanCurv = surfaceArea*pmath.SurfaceAverage(
                                curvatureSurface,
                                names.SqrMeanCurvatureArrayName
                            )

        self._GLN = np.sqrt(surfaceArea*surfIntSqrGaussCurv)/(4*const.pi)
        self._MLN = np.sqrt(surfIntSqrMeanCurv)/(4*const.pi)

        # Trial with new curvature metric
        # Computing the hyperbolic L2-norm
        hyperbolicPatches = tools.ClipWithScalar(
                                curvatureSurface,
                                names.GaussCurvatureArrayName,
                                const.zero
                            )

        # Check if there is any hyperbolic areas
        if hyperbolicPatches is None:
            self._HGLN = const.zero

        else:
            hyperbolicArea = geo.Surface.Area(hyperbolicPatches)

            surfIntHypSqrGaussCurv = hyperbolicArea*pmath.SurfaceAverage(
                                                        hyperbolicPatches,
                                                        names.SqrGaussCurvatureArrayName
                                                    )

            self._HGLN = np.sqrt(hyperbolicArea*surfIntHypSqrGaussCurv)/(4*const.pi)

        # Remove temporary squared arrays of surface
        curvatureSurface.GetCellData().RemoveArray(
            names.SqrGaussCurvatureArrayName
        )

        curvatureSurface.GetCellData().RemoveArray(
            names.SqrMeanCurvatureArrayName
        )

    def _compute_distance_to_neck(self):
        """Based on neck contour, compute geodesic distance to neck field."""

        pointIds = tools.GetClosestContourOnSurface(
                       self._aneurysm_surface,
                       self._neck_contour
                   )

        # Compute the geodesic distance  from the approximate neck contour
        surface = geo.SurfaceGeodesicDistanceToContour(
                      self._aneurysm_surface,
                      pointIds,
                      gdistance_array_name=names.DistanceToNeckArrayName
                  )

        # Change sign to conform with names.DistanceToNeckArrayName values
        # from VascularTree models -> negative inside the aneurysm
        npSurface = dsa.WrapDataObject(surface)

        npSurface.PointData.append(
            -npSurface.PointData.GetArray(names.DistanceToNeckArrayName),
            names.DistanceToNeckArrayName
        )

        self._aneurysm_surface = npSurface.VTKObject

    # Public interface
    def GetMorphologyMetrics(self) -> dict:
        """Get dict of all morphology metrics."""

        return {
            names.iaMetricSurfaceArea       : self._surface_area,
            names.iaMetricOstiumArea        : self._ostium_area,
            names.iaMetricHullSurfaceArea   : self._hull_surface_area,
            names.iaMetricHullVolume        : self._hull_volume,
            names.iaMetricVolume            : self._volume,
            names.iaMetricNeckDiameter      : self._neck_diameter,
            names.iaMetricMaxNormalHeight   : self._max_normal_height,
            names.iaMetricMaxDiameter       : self._max_diameter,
            names.iaMetricAspectRatio       : self._aspect_ratio,
            names.iaMetricBottleneckFactor  : self._bottleneck_factor,
            names.iaMetricConicityParameter : self._conicity_parameter,
            names.iaMetricNonsphericityIndex: self._nonsphericity_index,
            names.iaMetricEllipticityIndex  : self._ellipticity_index,
            names.iaMetricUndulationIndex   : self._undulation_index,
            names.areaAvgGaussCurvature     : self._GAA,
            names.areaAvgMeanCurvature      : self._MAA,
            names.l2NormMeanCurvature       : self._MLN,
            names.l2NormGaussCurvature      : self._GLN,
            "HGLN": self._HGLN
        }

    def GetDomeTipPoint(self) -> tuple:
        """Return the aneurysm surface."""
        return tuple(self._dome_point)

    def GetSurface(self) -> names.polyDataType:
        """Return the aneurysm surface."""
        return self._aneurysm_surface

    def GetHullSurface(self) -> names.polyDataType:
        """Return the aneurysm' convex hull surface."""
        return self._hull_surface

    def GetOstiumSurface(self) -> names.polyDataType:
        """Return the aneurysm's ostium surface."""
        return self._ostium_surface

    def GetAneurysmSurfaceArea(self) -> float:
        """Return the aneurysm surface area."""
        return self._surface_area

    def GetOstiumArea(self) -> float:
        """Return the aneurysm ostium surface area."""
        return self._ostium_area

    def GetAneurysmVolume(self) -> float:
        """Return the aneurysm enclosed volume."""
        return self._volume

    def GetHullSurfaceArea(self) -> float:
        """Return the aneurysm' convex hull surface area."""
        return self._hull_surface_area

    def GetHullVolume(self) -> float:
        """Return the aneurysm's convex hull volume."""
        return self._hull_volume

    def GetNeckDiameter(self) -> float:
        """Return the aneurysm neck diameter.

        The neck diameter is defined as the the hydraulic diameter of the
        ostium surface:

        .. math::
            D_n = 4A_n/p_n

        where :math:`A_n` is the aneurysm ostium surface area, and :math:`p_n`
        is its perimeter.  The ideal computation would be based on a plane
        ostium section, but it also works ai 3D neck contour.
        """

        return self._neck_diameter

    def GetMaximumNormalHeight(self) -> float:
        """Return maximum normal height.

        The maximum normal aneurysm height is defined as the maximum distance
        between the neck barycenter and the aneurysm surface.
        """

        return self._max_normal_height

    def GetMaximumDiameter(self) -> float:
        """Return the diameter of the largest section."""

        return self._max_diameter

    # 2D Shape indices
    def GetAspectRatio(self) -> float:
        """Return the aspect ratio.

        The aspect ratio is defined as the ratio between the maximum
        perpendicular height and the neck diameter.
        """

        return self._aspect_ratio

    def GetBottleneckFactor(self) -> float:
        """Return the bottleneck factor.

        The bottleneck factor is defined as the ratio between the maximum
        diameter and the neck diameter. This index represents "the level to
        which the neck acts as a bottleneck to entry of blood during normal
        physiological function and to coils during endovascular procedures".
        """

        return self._bottleneck_factor

    def GetConicityParameter(self) -> float:
        """Return the conicity parameter.

        The conicity parameter was defined by Raghavan et al. (2005) as a shape
        metric for saccular cerebral aneurysms and measures how far is the
        'bulge' of the aneurysm, i.e. the section of largest section, from the
        aneurysm ostium surface. In the way it was defined, it can vary from
        -0.5 (the bulge is at the dome) to 0.5 (bulge closer to neck); 0.0
        indicates when the bulge is at the midway from neck to the maximum
        normal height.
        """

        return self._conicity_parameter

    # 3D Shape indices
    def GetNonSphericityIndex(self) -> float:
        """Return the non-sphericity index.

        The non-sphericity index of an aneurysm surface is defined as:

        .. math::
            NSI = 1 - (18\pi)^{1/3}V^{2/3}_a/S_a

        where :math:`V_a` and :math:`S_a` are the volume and surface area of
        the aneurysm.
        """

        return self._nonsphericity_index

    def GetEllipticityIndex(self) -> float:
        """Return the ellipticity index.

        The ellipiticity index of an aneurysm surface is given by:

        .. math::
            EI = 1 - (18\pi)^{1/3}V^{2/3}_{ch}/S_{ch}

        where :math:`V_{ch}` and :math:`S_{ch}` are the volume and surface area
        of the convex hull.
        """

        return self._ellipticity_index

    def GetUndulationIndex(self) -> float:
        """Return the undulation index.

        The undulation index of an aneurysm is defined as:

        .. math::
            UI = 1 - V_a/V_{ch}

        where :math:`V_a` is the aneurysm volume and :math:`V_{ch}` the volume
        of its convex hull.
        """
        return self._undulation_index

    def GetCurvatureMetrics(self) -> dict:
        """Get the curvature-based metrics.

        Based on local mean and Gaussian curvatures, compute their
        area-averaged values (MAA and GAA, respectively) and their L2-norm (MLN
        and GLN), as defined in

        Ma et al. (2004).  Three-dimensional geometrical characterization
        of cerebral aneurysms.

        Return a dictionary with the metrics (keys MAA, GAA, MLN, and GLN).

        .. warning::
            Assumes that both curvature arrays, Gaussian and mean, are defined
            on the aneurysm surface for a more accurate calculation, avoiding
            border effects.
        """
        return {
            names.areaAvgMeanCurvature : self._MAA,
            names.areaAvgGaussCurvature: self._GAA,
            names.l2NormMeanCurvature  : self._MLN,
            names.l2NormGaussCurvature : self._GLN,
            "HGLN": self._HGLN
        }

    def GetHemodynamicStats(
            self,
            n_percentile: float=99
        ) -> dict:
        """Compute the statistics of hemodynamic fields.

        If the loaded aneurysm surface contains the fields of hemodynamic
        variables, returns its descriptive statistics as a dict with the
        following statistics: average, maximum, minimum, percetile (value
        passed as optional by the user) and the surface-average over the
        aneurysm surface.
        """

        # Compute the statistis of the hemodynamics already on the surface
        dictHemodynamics =  {
            hwp: pmath.SurfaceFieldStatistics(
                     self._aneurysm_surface,
                     hwp,
                     n_percentile=n_percentile
                 )
            for hwp in names.listHWP
            if hwp in tools.GetCellArrays(self._aneurysm_surface)
        }

        # Add the other ones defined here
        dictHemodynamics.update({
            names.LowShearArea: self.GetLowTAWSSArea()
        })

        return dictHemodynamics

    def GetLowTAWSSArea(
            self
        )   -> float:
        """Computes the LSA based on the time-averaged WSS (TAWSS) field."""

        if names.TAWSS in tools.GetCellArrays(self._aneurysm_surface):

            # Compute low shear area
            lsaPortion = tools.ClipWithScalar(
                             self._aneurysm_surface,
                             names.TAWSS,
                             const.lowWSS
                         )

            lsaArea = geo.Surface.Area(lsaPortion)

            return lsaArea/self._surface_area

        else:
            return None

    def ComputeSacRegionsField(
            self,
            neck_to_body_fraction: float=0.2,
            body_to_dome_fraction: float=0.6
        ):
        """Compute the aneurysm sac regions based on the distance to neck
        array.

        This functions splits a saccular aneurysm sac surface into three
        regions called "dome", "neck", and "body". These denominations are
        typically employed by neurosurgeons to split an aneurysm sac into very
        distinct patches. Although commonly employed in the medical practice,
        no formal mathematical definition of it exists. Therefore, the one
        proposed and used by Salimi Ashkezari et al. in their paper:

            S. F. Salimi Ashkezari et al., “Blebs in intracranial aneurysms:
            prevalence and general characteristics,” J NeuroIntervent Surg, vol.
            13, no. 3, pp. 226–230, Mar. 2021, doi:
            10.1136/neurintsurg-2020-016274.

        is implemented here. It defines each region based on the geodesic
        distance to the neck contour: given the maximum geodesic distance to
        the aneurysm neck within the aneurysm, the neck is defined as the
        region within 20% of this distance, the body is defined as the region
        between 20% and 60% of this distance, and the dome is defined as the
        region between 60% and 100% of this distance. The rest of the aneurysm
        sac is considered out of the sac. These value can be adjusted by the
        user through the arguments 'neck_to_body_value' and
        'body_to_dome_value'.

        The method updates the aneurysm sac surface with a new field called
        "SacRegions" (module 'names.SacRegionsArrayName') where regions are
        identified by the code:

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

        if names.DistanceToNeckArrayName not in tools.GetPointArrays(
                self._aneurysm_surface
            ):

            # Add distance to neck field
            self._compute_distance_to_neck()

        # Update aneurysm model surface
        # Interpolate the distance to neck aray to cell data
        surface = tools.PointFieldToCellField(
                      self._aneurysm_surface,
                      names.DistanceToNeckArrayName
                  )

        # Use numpy interface
        npSurface = dsa.WrapDataObject(surface)

        distanceToNeckArray = npSurface.GetCellData().GetArray(
                                  names.DistanceToNeckArrayName
                              )

        # Get the minimum value (negative values lie on the aneurysm sac)
        iaMaxGeodesicDistance = min(distanceToNeckArray)

        # Create new array on aneurysm based on:
        # Define thresholds based on the negative iaMaxGeodesicDistance
        threshold_body_dome = body_to_dome_fraction*iaMaxGeodesicDistance
        threshold_neck_body = neck_to_body_fraction*iaMaxGeodesicDistance

        # Using numpy.select for a cleaner and more robust assignment
        # The order of conditions is important with np.select.
        # We define them from "out of sac" inwards, or from "most specific" to
        # "least specific".
        # Given the description, "Out of Sac" is the most general "background".
        # Then we define the regions within the sac from neck to dome.

        # Conditions list, ordered logically
        conditions = [
            # Out of the sac
            # > 0 -> out of sac -> 0
            distanceToNeckArray > const.zero,
            # Neck -> 0 <= distance <= abs(0.2*maxGeoDist)  -> 1
            (distanceToNeckArray >= threshold_neck_body)
            &
            (distanceToNeckArray <= const.zero),
            # Body -> abs(0.2*maxGeoDist) < distance <= abs(0.6*maxGeoDist) -> 2
            (distanceToNeckArray >= threshold_body_dome)
            &
            (distanceToNeckArray < threshold_neck_body),
            # Dome -> abs(0.6*maxGeoDist) < distance <= abs(maxGeoDist) -> 3
            (distanceToNeckArray >= iaMaxGeodesicDistance)
            &
            (distanceToNeckArray < threshold_body_dome)
        ]

        # Corresponding choices (IDs) for each condition
        choices = [
            const.IaSacRegionsTypes["OutOfSac"],
            const.IaSacRegionsTypes["Neck"],
            const.IaSacRegionsTypes["Body"],
            const.IaSacRegionsTypes["Dome"]
        ]

        # Create the sacRegionArray using numpy.select
        sacRegionArray = np.select(
                             conditions,
                             choices,
                             default=-const.one
                         )

        # Append the new array to CellData
        npSurface.CellData.append(
            sacRegionArray,
            names.SacRegionsArrayName
        )

        npSurface.VTKObject.GetCellData().RemoveArray(
            names.DistanceToNeckArrayName
        )

        # Updates object
        self._aneurysm_surface = npSurface.VTKObject

    def GetSacCenterline(self) -> names.polyDataType:
        """Return the aneurysm sac centerline.

        The centerline of an aneurysm is a 3D path that travels through the
        center of its sac. It was defined in a procedure to compute the
        aneurysm neck plane in the study:

            M. Piccinelli, D. A. Steinman, Y. Hoi, F. Tong, A. Veneziani, and
            L. Antiga, "Automatic neck plane detection and 3d geometric
            characterization of aneurysmal sacs", Annals of Biomedical
            Engineering, vol. 40, no. 10, pp. 2188–2211, 2012, doi:
            10.1007/s10439-012-0577-5.

        Based on the names.DistanceToNeckArrayName field, defined on the
        aneurysm sac surface when it is segmented from the vascular model, the
        algorithm computes the barycenter/centroid of the isocontours defined by
        the names.DistanceToNeckArrayName field, and uses them to compute the
        VTK polydata of the sac centerline.

        Note that, therefore, the exact centerline of the aneurysm sac here
        depends on the mode to compute the aneurysm neck.
        """

        if self._sac_centerline is None:
            if names.DistanceToNeckArrayName not in tools.GetPointArrays(
                    self._aneurysm_surface
                ):

                # Add distance to neck field
                self._compute_distance_to_neck()

            # Get centerline points from aneurysm model
            sac_centerline_pts, _ = ComputeSacCenterlinePoints(
                                        self._aneurysm_surface,
                                        distance_array=names.DistanceToNeckArrayName
                                    )

            self._sac_centerline = tools.Build3DCurvePolyData(
                                       sac_centerline_pts
                                   )

        return self._sac_centerline

class VascularTreeWithAneurysm(VascularTree, ABC):
    """Abstract base class to represent a vascular network tree model with a
    saccular aneurysm.

    Inherits from VascularTree and provides additional functionality for
    handling lateral aneurysms in the vascular model. The user must provide
    whether the aneurysm surface will be detected automatically (experimental)
    and a plane neck will be generated, or manually draw by the user, in which
    case a window is open allowing the user to select the aneurysm neck.
    """

    def __init__(
            self,
            vtk_poly_data: names.polyDataType,
            centerlines_data: names.polyDataType=None,
            clip_aneurysm_mode: str="interactive",
            dome_point: tuple=None
        ):
        """Initiate vascular model.

        Given a vascular surface (vtkPolyData), automatically compute its
        centerlines and bifurcations geometry. If the vasculature has an
        aneurysm, the flag 'with_aneurysm' enables its selection.

        Arguments:
        vtk_poly_data -- the vtkPolyData vascular model (default None)

        clip_aneurysm_mode (str, default: 'interactive') -- the method to clip
            the aneurysm, if present. Use the function
            'neck_extractor.ClipAneurysmSacSurface', hence the options
            are: 'interactive', 'automatic', or 'plane'.  Only enabled if the
            'with_aneurysm' arguments is True.  (default False).

        dome_point (tuple, optional, default 'None') -- tuple with
            coordinates of a point on the aneurysm dome (used only with the
            'plane' mode to extract the aneurysm).
        """

        super().__init__(
            vtk_poly_data,
            centerlines_data=centerlines_data
        )

        self._dome_point  = dome_point
        self._clip_aneurysm_mode = clip_aneurysm_mode

        self._aneurysm_thickness_computed = False
        self._aneurysmal_region_computed = False

        # These will be filled at the concrete classes
        self._healthy_vessel_surface = None
        self._sac_surface = None
        self._vascular_surface_no_aneurysm = None
        self._aneurysm_model = None

        if clip_aneurysm_mode == "automatic" or clip_aneurysm_mode == "plane":
            raise NotImplementedError(
                "The automatic clipping mode is not implemented for " +
                "VascularTreeWithAneurysm. Use 'interactive' mode " +
                "instead."
            )

    @classmethod
    def from_file(
            cls,
            file_name,
            centerlines_data=None,
            clip_aneurysm_mode="interactive",
            dome_point=None
        ):
        """Initialize vasculature object from vasculature surface file."""

        return cls(
            tools.ReadSurface(file_name),
            clip_aneurysm_mode=clip_aneurysm_mode,
            dome_point=dome_point
        )

    # TODO: now that the branchingof the surfaces work, implement function to
    # select the parent artery of the cases with an aneurysm and compute the
    # normalized to the parent artery metrics (I can select a portion of it
    # only for the computations based on the distance along the centerline
    # array)

    def _mark_aneurysm_wall_influence_region(self):
        """Mark the aneurysm neck contour with an array called DistanceToNeck
        with zero values at the neck and the negative-distance to it inside the
        aneurysm sac."""

        # TODO the computation of the thickness should depend only on
        # the 3D neck or interactive neck approaches, as the plane one
        # is not realistic
        # hence, the computation here should be specific and not tied
        # to the neck computation chosen by the user.
        # When it become automatic, then it should become an abstract method
        if not self._aneurysmal_region_computed:

            # Create the neck identification strategy
            # When working for all cases, use the automatic 3D strategy
            neckClipperStrategy = InteractiveNeckIdentification(
                                      self._vascular_surface,
                                      distance_to_neck_field_name=names.AneurysmalRegionArrayName
                                  )

            marked_neck_surface = neckClipperStrategy.MarkAneurysmNeck()

            # Use client function to mark the aneurysm sac surface
            # Updates the vascular surface with the marked aneurysm neck
            self._vasc_surface_obj = VascularSurface(marked_neck_surface)
            self._vascular_surface = self._vasc_surface_obj.GetSurface()

            self._aneurysmal_region_computed = True


    @abstractmethod
    def _clip_sac_surface(self):
        """Clip the aneurysm sac surface and initialize SaccularAneurysm.

        This method should be implemented in the subclasses to clip the
        aneurysm sac surface based on the specified clipping mode.
        """
        pass

    def ComputeVascularWallThickness(
            self,
            set_uniform_wlr: bool = False,
            uniform_wlr_value: float = const.WlrMedium,
            aneurysm_influence_dist: float = 0.5,
            scale_factor: float = 0.75,
            abnormal_thickness: bool = False,
            atherosclerotic_factor: float = 1.20,
            red_regions_factor: float = 0.95,
        ):
        """Computes the vascular wall thickness, including aneurysm-specific
        adjustments.

        Based on the vasculature thickness distribution, defined as the outside
        portion of the complete geometry from the neck selected by the user,
        estimates an aneurysm thickness by averaging the vasculature thickness
        using as weight function the inverse distance to the
        "aneurysm-influenced" region line. The estimated aneurysm thickness is,
        then, set on the aneurysm surface in the thickness array.

        The aneurysm-influenced neck line is defined as the region between the
        neck line (provided by the user or computed automatically) and the path
        that is at a distance of 'AneurysmInfluencedRegionDistance' value (in
        mm; default 0.5 mm) from the neck line. This strip around the aneurysm
        is imagined as a region of the original vasculature that had its
        thickness changed by the aneurysm growth.

        The aneurysm sac thickness may be estimated as 'uniform', the default
        behavior, or using the abnormal wall thickness based on the adjacent
        hemodynamics to the aneurysm wall: the TAWSS and OSI fields (controlled
        by setting the option 'abnormal_thickness' to True). In the latter, the
        passed suface must have these two field from a CFD simulation.

        The aneurysm abnormal thickness is computed based on a 'WallType'
        field that acts a scaling factor by increasing or deacreasing the sac
        thickness. The procedure is as follows: With a global thickness array
        already defined on the surface from the base class, update the
        thickness based on the wall type array created based on the
        hemodynamics variables, by multiplying it by a factor defined below.
        The three types of wall and the operation performed here for each are:

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
        provided, with default values given above. The function will look for
        the array named "WallType" for defining its operation or compute it on
        the fly.

        Arguments:
            set_uniform_wlr (bool): If True, use a uniform wall-to-lumen ratio
                for initial computation.

            uniform_wlr_value (float): The uniform wall-to-lumen ratio to use.

            aneurysm_influence_dist (float): Distance defining the aneurysm
                influenced region.

            scale_factor (float): Scaling factor for aneurysm thickness
                average. Default of 0.75.

            abnormal_thickness (bool): If True, apply abnormal thickness
                adjustments based on hemodynamics.

            atherosclerotic_factor (float): Factor for atherosclerotic regions.

            red_regions_factor (float): Factor for "red" regions.
        """
        if not self._aneurysm_thickness_computed:

            # First, call the parent's general thickness computation
            # This will add to the _vascular_surface the basic thickness
            # field and set self._thickness_computed = True
            super().ComputeVascularWallThickness(
                set_uniform_wlr=set_uniform_wlr,
                uniform_wlr_value=uniform_wlr_value
            )

            # Now, apply the aneurysm-specific adjustments
            # Compute the distance to neck array if not already present
            if  not self._aneurysmal_region_computed:
                # TODO the computation of the thickness should depend only on
                # the 3D neck or interactive neck approaches, as the plane one
                # is not realistic
                # hence, the computation here should be specific and not tied
                # to the neck computation chosen by the user.
                self._mark_aneurysm_wall_influence_region()

            # Surface with thickness and distnce to neck
            npDistanceSurface = dsa.WrapDataObject(self._vascular_surface)

            # Update both fields with selection
            thicknessArray = npDistanceSurface.GetPointData().GetArray(
                                 names.ThicknessArrayName
                             )

            distanceToNeckArray = npDistanceSurface.GetPointData().GetArray(
                                      names.AneurysmalRegionArrayName
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
                                       field_name=names.ThicknessArrayName,
                                       atherosclerotic_factor=atherosclerotic_factor,
                                       red_regions_factor=red_regions_factor
                                   )

            # After array created, smooth it hard
            vascular_surface = tools.SmoothSurfacePointField(
                                   vascular_surface,
                                   names.ThicknessArrayName,
                                   niterations=5
                               )

            # Updates vascular surface OBJECT
            self._vasc_surface_obj = VascularSurface(vascular_surface)
            self._vascular_surface = self._vasc_surface_obj.GetSurface()
            self._aneurysm_thickness_computed = True

        else:
            print("Aneurysm thickness field already computed. Skipping re-computation.")

    def ComputeVascularElasticConstants(
            self,
            elastic_const_field_name: str=names.ElasticityArrayName,
            aneurysm_elastic_const_mode: str="uniform",
            arteries_elastic_const: float=5e6,
            aneurysm_elastic_const: float=2e6,
            abnormal_elasticity: bool=False,
            atherosclerotic_factor: float=1.20,
            red_regions_factor: float=0.95
        )   -> names.polyDataType:
        """Calculate and set aneurysm and vascular elastic constant field.

        Based on a value for the aneurysm elasticity and the arterial
        elasticity, set them on the vascular surface. The arterial elasticity
        is considered to be uniform, whereas the aneurysm elasticity accepts
        two modes:

            * 'uniform': uniform elasticity;
            * 'linear': elasticity linearly varying from the arterial value to
                a value set by the user too.

        The aneurysm influence region is either provided by the user or
        computed automatically, through the array 'DistanceToNeck' that marks
        the neck contour with zero values. If the surface does not already have
        the 'DistanceToNeck' scalar array, then it will prompt the user to
        select the neck line, which will be stored on the surface.

        The option 'abnormal_elasticity' allows for the automatic update of the
        aneurysm elasticity based on the adjacent hemodynamics to the aneurysm
        wall: the TAWSS and OSI fields. In this last case, the passed surface
        must have these two field from a CFD simulation.

        The aneurysm abnormal elastic constant field is computed based on a
        'WallType' field that acts a scaling factor by increasing or
        deacreasing the sac elastic constant. The procedure is as follows: With
        a global elasticity array already defined on the surface, update the
        elasticity based on the wall type array created based on the
        hemodynamics variables, by multiplying it by a factor defined below.
        The three types of wall and the operation performed here for each are:

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
        provided, with default values given above. The function will look for
        the array named "WallType" for defining its operation or compute it on
        the fly.

        The function can be called any times to compute different elastic
        constant fields as necessary, as long as the name of the field changes.

        Arguments:
            elastic_const_field_name (str): Name of the elasticity field to be
                created. Default is 'E'.

            aneurysm_elastic_const_mode (str): Mode for aneurysm elasticity
                definition. Either 'uniform' or 'linear'. Default is 'uniform'.

            arteries_elastic_const (float): Elastic constant for the healthy
                arteries. Default is 5e6.

            aneurysm_elastic_const (float): Elastic constant for the aneurysm
                sac. Default is 2e6.

            abnormal_elasticity (bool): If True, apply abnormal elastic
                constant adjustments based on hemodynamics.

            atherosclerotic_factor (float): Factor for atherosclerotic regions.

            red_regions_factor (float): Factor for "red" regions.
        """
        # Compute the distance to neck array if not already present
        if not self._aneurysmal_region_computed:
            self._mark_aneurysm_wall_influence_region()

        # Surface with thickness and distnce to neck
        npDistanceSurface = dsa.WrapDataObject(self._vascular_surface)

        distanceArray = npDistanceSurface.PointData.GetArray(
                            names.AneurysmalRegionArrayName
                        )

        # Array to hold the actual elasticity array
        elasticities = dsa.VTKArray(
                            np.zeros(
                                shape=self._vascular_surface.GetNumberOfPoints()
                            )
                        )

        # Mark regions based on distance array values
        onAneurysm  = distanceArray <= 0.0
        outAneurysm = distanceArray > 0.0

        elasticities[outAneurysm] = arteries_elastic_const

        # One single aneurysm expected here
        if aneurysm_elastic_const_mode == "uniform":

            elasticities[onAneurysm] = aneurysm_elastic_const

        elif aneurysm_elastic_const_mode == "linear":

            # Fundus and neck elasticity
            neckElasticity   = arteries_elastic_const
            fundusElasticity = aneurysm_elastic_const

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
                          aneurysm_elastic_const_mode
                      )
                  )

        npDistanceSurface.PointData.append(
            elasticities,
            elastic_const_field_name
        )

        vascular_surface = npDistanceSurface.VTKObject

        if abnormal_elasticity:
            vascular_surface = UpdateAbnormalHemodynamicsRegions(
                                   vascular_surface,
                                   field_name=elastic_const_field_name,
                                   atherosclerotic_factor=atherosclerotic_factor,
                                   red_regions_factor=red_regions_factor
                               )

        # After array created, smooth it hard to remove discontinuity
        vascular_surface = tools.SmoothSurfacePointField(
                               vascular_surface,
                               elastic_const_field_name,
                               niterations=5
                           )

        # Updates vascular surface OBJECT
        self._vasc_surface_obj = VascularSurface(vascular_surface)
        self._vascular_surface = self._vasc_surface_obj.GetSurface()

    def GetAneurysm(self):
        """Return the aneurysm model."""
        if self._aneurysm_model is None:
            # Clip the aneurysm sac surface
            self._clip_sac_surface()

        return self._aneurysm_model

    def GetAneurysmExtractionMode(self):
        """Return the extraction model of the aneurysm."""
        return self._clip_aneurysm_mode

class VascularTreeWithLateralAneurysm(VascularTreeWithAneurysm):
    """Representation of a vascular network tree model with a lateral saccular
    aneurysm.

    Inherits from VascularTree and provides additional functionality for
    handling lateral aneurysms in the vascular model. The user must provide
    whether the aneurysm surface will be detected automatically (experimental)
    and a plane neck will be generated, or manually draw by the user, in which
    case a window is open allowing the user to select the aneurysm neck.
    """

    def __init__(
            self,
            vtk_poly_data: names.polyDataType,
            centerlines_data: names.polyDataType=None,
            clip_aneurysm_mode: str="interactive",
            dome_point: tuple=None
        ):
        """Initiate vascular model with lateral aneurysm.

        Given a vascular surface (vtkPolyData), automatically compute its
        centerlines and bifurcations geometry. If the vasculature has an
        aneurysm, the flag 'with_aneurysm' enables its selection.

        Arguments:
        vtk_poly_data -- the vtkPolyData vascular model (default None)

        clip_aneurysm_mode (str, default: 'interactive') -- the method to clip
            the aneurysm, if present. Use the function
            'vascular_operations.ClipAneurysmSacSurface', hence the options
            are: 'interactive', 'automatic', or 'plane'.  Only enabled if the
            'with_aneurysm' arguments is True.  (default False).

        dome_point (tuple, optional, default 'None') -- tuple with
            coordinates of a point on the aneurysm dome (used only with the
            'plane' mode to extract the aneurysm).
        """

        super().__init__(
            vtk_poly_data,
            centerlines_data,
            clip_aneurysm_mode,
            dome_point
        )

    # TODO: the healthy vessel reconstruction should be performed here
    # as the VascularTreeWithAneurysm is the only one that uses the
    # healthy_vessel_surface attribute. It should not be passed by the
    # user, but computed internally.

    def _clip_sac_surface(self):
        """Clip the aneurysm sac surface and initialize SaccularAneurysm."""

        # Use client function to clip the aneurysm sac surface
        clippedSurfaceTuple = ClipAneurysmSacSurface(
                                  self.GetVascularSurface(),
                                  mode=self._clip_aneurysm_mode,
                                  healthy_vessel_surface=self._healthy_vessel_surface,
                                  aneurysm_type="lateral",
                                  dome_point=self._dome_point
                              )

        # Clip the aneurysm sac (aneurysm marked with negative values)
        self._sac_surface, self._vascular_surface_no_aneurysm = clippedSurfaceTuple

        # Build aneurysm model
        self._aneurysm_model = SaccularAneurysm(self._sac_surface)

class VascularTreeWithBifurcationAneurysm(VascularTreeWithAneurysm):
    """Representation of a vascular network tree model with a bifurcation
    saccular aneurysm.

    Inherits from VascularTree and provides additional functionality for
    handling bifurcation aneurysms in the vascular model. The user must provide
    whether the aneurysm surface will be detected automatically (experimental)
    and a plane neck will be generated, or manually draw by the user, in which
    case a window is open allowing the user to select the aneurysm neck.
    """

    def __init__(
            self,
            vtk_poly_data: names.polyDataType,
            centerlines_data: names.polyDataType=None,
            clip_aneurysm_mode: str="interactive",
            dome_point: tuple=None
        ):
        """Initiate vascular model with bifurcation aneurysm.

        Given a vascular surface (vtkPolyData), automatically compute its
        centerlines and bifurcations geometry. If the vasculature has an
        aneurysm, the flag 'with_aneurysm' enables its selection.

        Arguments:
        vtk_poly_data -- the vtkPolyData vascular model (default None)

        clip_aneurysm_mode (str, default: 'interactive') -- the method to clip
            the aneurysm, if present. Use the function
            'vascular_operations.ClipAneurysmSacSurface', hence the options
            are: 'interactive', 'automatic', or 'plane'.  Only enabled if the
            'with_aneurysm' arguments is True.  (default False).

        dome_point (tuple, optional, default 'None') -- tuple with
            coordinates of a point on the aneurysm dome (used only with the
            'plane' mode to extract the aneurysm).
        """

        super().__init__(
            vtk_poly_data,
            centerlines_data,
            clip_aneurysm_mode,
            dome_point
        )

    # TODO: the healthy vessel reconstruction should be performed here
    # as the VascularTreeWithAneurysm is the only one that uses the
    # healthy_vessel_surface attribute. It should not be passed by the
    # user, but computed internally.

    def _clip_sac_surface(self):
        """Clip the aneurysm sac surface and initialize SaccularAneurysm."""

        # Use client function to clip the aneurysm sac surface
        clippedSurfaceTuple = ClipAneurysmSacSurface(
                                  self.GetVascularSurface(),
                                  mode=self._clip_aneurysm_mode,
                                  healthy_vessel_surface=self._healthy_vessel_surface,
                                  aneurysm_type="bifurcation",
                                  dome_point=self._dome_point
                              )

        # Clip the aneurysm sac (aneurysm marked with negative values)
        self._sac_surface, self._vascular_surface_no_aneurysm = clippedSurfaceTuple

        # Build aneurysm model
        self._aneurysm_model = SaccularAneurysm(self._sac_surface)
