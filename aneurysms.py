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
from vtk.numpy_interface import dataset_adapter as dsa

from vmtk import vtkvmtk
from vmtk import vmtkscripts
from scipy.spatial import ConvexHull

# Local modules
from vmtk4aneurysms.lib import names
from vmtk4aneurysms.lib import constants as const
from vmtk4aneurysms.lib import polydatatools as tools
from vmtk4aneurysms.lib import polydatageometry as geo
from vmtk4aneurysms.lib import polydatamath as pmath

from vmtk4aneurysms.vascular_operations import ComputeGeodesicDistanceToAneurysmNeck

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

    ldAneurysm = Aneurysm(ldAneurysmSurface)
    psAneurysm = Aneurysm(psAneurysmSurface)

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
                           mode="interactive",
                           gdistance_to_neck_array_name=aneurysm_neck_array_name
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
    ldAneurysm = Aneurysm(ldAneurysmSurface)
    psAneurysm = Aneurysm(psAneurysmSurface)

    # Compute pulsatility
    return psAneurysm.GetAneurysmVolume()/ldAneurysm.GetAneurysmVolume() - 1.0

class Aneurysm:
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

    def __init__(self, surface, aneurysm_type='', status='', label=''):
        """Initiates aneurysm model.

        Given the aneurysm surface (vtkPolyData), its type, status, and a
        label, initiates aneurysm model by computing simple size
        features: surface area, ostium surface area, and volume.

        Arguments:
        surface (vtkPolyData) -- the aneurysm surface
        aneurysm_type (str) -- aneurysm type: bifurcation or lateral
        (default '')
        status (str) -- rupture or unruptured (default '')
        label (str) -- an useful label (default '')
        """
        self.type = aneurysm_type
        self.label = label
        self.status = status
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

    # Public interface
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

        return self._max_normal_height/self._neck_diameter

    def GetBottleneckFactor(self) -> float:
        """Return the bottleneck factor.

        The bottleneck factor is defined as the ratio between the maximum
        diameter and the neck diameter. This index represents "the level to
        which the neck acts as a bottleneck to entry of blood during normal
        physiological function and to coils during endovascular procedures".
        """

        return self._max_diameter/self._neck_diameter

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

        return 0.5 - self._bulge_height/self._max_normal_height

    # 3D Shape indices
    def GetNonSphericityIndex(self) -> float:
        """Return the non-sphericity index.

        The non-sphericity index of an aneurysm surface is defined as:

        .. math::
            NSI = 1 - (18\pi)^{1/3}V^{2/3}_a/S_a

        where :math:`V_a` and :math:`S_a` are the volume and surface area of
        the aneurysm.
        """
        factor = (18*const.pi)**(1./3.)

        area = self._surface_area
        volume = self._volume

        return const.one - (factor/area)*(volume**(2./3.))

    def GetEllipticityIndex(self) -> float:
        """Return the ellipticity index.

        The ellipiticity index of an aneurysm surface is given by:

        .. math::
            EI = 1 - (18\pi)^{1/3}V^{2/3}_{ch}/S_{ch}

        where :math:`V_{ch}` and :math:`S_{ch}` are the volume and surface area
        of the convex hull.
        """

        factor = (18*const.pi)**(1./3.)

        area = self._hull_surface_area
        volume = self._hull_volume

        return const.one - (factor/area)*(volume**(2./3.))

    def GetUndulationIndex(self) -> float:
        """Return the undulation index.

        The undulation index of an aneurysm is defined as:

        .. math::
            UI = 1 - V_a/V_{ch}

        where :math:`V_a` is the aneurysm volume and :math:`V_{ch}` the volume
        of its convex hull.
        """
        return 1.0 - self._volume/self._hull_volume

    def GetCurvatureMetrics(self) -> dict:
        """Compute the curvature-based metrics.

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
        # Get arrays on the aneurysm surface
        arrayNames = tools.GetCellArrays(self._aneurysm_surface)

        curvatureArrays = {'Mean': names.MeanCurvatureArrayName,
                           'Gauss': names.GaussCurvatureArrayName}

        # Check if there is any curvature array on the aneurysm surface
        if not all(array in arrayNames for array in curvatureArrays.values()):

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

        arrGaussCurv = npCurvSurface.CellData.GetArray(curvatureArrays["Gauss"])
        arrMeanCurv  = npCurvSurface.CellData.GetArray(curvatureArrays["Mean"])

        nameSqrGaussCurv = "Squared_Gauss_Curvature"
        nameSqrMeanCurv  = "Squared_Mean_Curvature"

        npCurvSurface.CellData.append(
            arrGaussCurv**2,
            nameSqrGaussCurv
        )

        npCurvSurface.CellData.append(
            arrMeanCurv**2,
            nameSqrMeanCurv
        )

        curvatureSurface = npCurvSurface.VTKObject

        GAA = pmath.SurfaceAverage(
                    curvatureSurface,
                    curvatureArrays["Gauss"]
                )

        MAA = pmath.SurfaceAverage(
                    curvatureSurface,
                    curvatureArrays["Mean"]
                )

        surfIntSqrGaussCurv = surfaceArea*pmath.SurfaceAverage(
                                curvatureSurface,
                                nameSqrGaussCurv
                            )
        surfIntSqrMeanCurv = surfaceArea*pmath.SurfaceAverage(
                                curvatureSurface,
                                nameSqrMeanCurv
                            )

        GLN = np.sqrt(surfaceArea*surfIntSqrGaussCurv)/(4*const.pi)
        MLN = np.sqrt(surfIntSqrMeanCurv)/(4*const.pi)

        # Computing the hyperbolic L2-norm
        hyperbolicPatches = tools.ClipWithScalar(
                                curvatureSurface,
                                curvatureArrays["Gauss"],
                                float(const.zero)
                            )
        hyperbolicArea    = geo.Surface.Area(hyperbolicPatches)

        # Check if there is any hyperbolic areas
        if hyperbolicArea > 0.0:
            surfIntHypSqrGaussCurv = hyperbolicArea*pmath.SurfaceAverage(
                                                        hyperbolicPatches,
                                                        nameSqrGaussCurv
                                                    )

            HGLN = np.sqrt(hyperbolicArea*surfIntHypSqrGaussCurv)/(4*const.pi)
        else:
            HGLN = 0.0

        return {"MAA": MAA,
                "GAA": GAA,
                "MLN": MLN,
                "GLN": GLN,
                "HGLN": HGLN}

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

        return {hwp: pmath.SurfaceFieldStatistics(
                         self._aneurysm_surface,
                         hwp,
                         n_percentile=n_percentile
                     )
                for hwp in names.hwpList
                if hwp in tools.GetCellArrays(self._aneurysm_surface)}

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
