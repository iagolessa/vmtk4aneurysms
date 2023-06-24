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

from vmtk4aneurysms.vascular_operations import MarkAneurysmSacManually


def SelectParentArtery(surface: names.polyDataType) -> names.polyDataType:
    """Compute array marking the aneurysm' parent artery.

    Given a vasculature with an aneurysm, prompt the user to draw a contour
    that marks the separation between the aneurysm's parent artery and the rest
    of the vasculature. An array (field) is then defined on the surface with
    value 0 on the parent artery and 1 out of it. Return a copy of the vascular
    surface with 'ParentArteryContourArray' field defined on it.

    .. warning::
        The smoothing array script works better on good quality triangle
        surfaces, hence, it would be good to remesh the surface prior to use
        it.
    """

    parentArteryDrawer = vmtkscripts.vmtkSurfaceRegionDrawing()
    parentArteryDrawer.Surface = surface
    parentArteryDrawer.InsideValue = 0.0
    parentArteryDrawer.OutsideValue = 1.0
    parentArteryDrawer.ContourScalarsArrayName = names.ParentArteryArrayName
    parentArteryDrawer.Execute()

    smoother = vmtkscripts.vmtkSurfaceArraySmoothing()
    smoother.Surface = parentArteryDrawer.Surface
    smoother.Connexity = 1
    smoother.Iterations = 10
    smoother.SurfaceArrayName = parentArteryDrawer.ContourScalarsArrayName
    smoother.Execute()

    return smoother.Surface

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

    # It is important to set cnostraint to zero to have 90 degrees angles on corners
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

    surface = geo.Surface.Normals(triangulate.GetOutput()) \
              if compute_normals else triangulate.GetOutput()

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

# Hemodynamics-related functions that operate only on the aneurysm surface
# Refactored here so canbe used directly inside the Aneurysm class
def AneurysmStats(
        neck_surface: names.polyDataType,
        array_name: str,
        neck_array_name: str=names.AneurysmNeckArrayName,
        n_percentile: float=99,
        neck_iso_value: float=const.NeckIsoValue
    )   -> dict:
    """Compute fields statistics over aneurysm surface.

    Given a surface with the fields of WSS hemodynamics variables defined on
    it, computes the average, maximum, minimum, percetile (value passed as
    optional by the user) and the surface-average over the aneurysm surface.
    Return a dictionary with the statistics.

    .. note::
        Assumes that the surface also contain a field name 'neck_array_name'
        that indicates the aneurysm portion with 0 and 1 on the rest of the
        vasculature. The function uses this array to clip the aneurysm portion.
        If this is not present on the surface, the function prompts the user to
        delineate the aneurysm neck.
    """

    pointArrays = tools.GetPointArrays(neck_surface)
    cellArrays  = tools.GetCellArrays(neck_surface)

    neckArrayInSurface = neck_array_name in pointArrays or \
                         neck_array_name in cellArrays

    if not neckArrayInSurface:
        # Compute neck array
        neck_surface = MarkAneurysmSacManually(neck_surface)

    # Get aneurysm
    aneurysmSurface = tools.ClipWithScalar(
                          neck_surface,
                          neck_array_name,
                          neck_iso_value
                      )

    return pmath.SurfaceFieldStatistics(
               aneurysmSurface,
               array_name,
               n_percentile=n_percentile
           )

def LsaAverage(
        neck_surface: names.polyDataType,
        lowWSS: float,
        neck_array_name: str=names.AneurysmNeckArrayName,
        neck_iso_value: float=const.NeckIsoValue,
        avgMagWSSArray: str=names.TAWSS
    )   -> float:
    """Computes the LSA based on the time-averaged WSS (TAWSS) field.

    Calculates the LSA (low WSS area ratio) for aneurysms. The input is a
    surface with the time-averaged WSS over the surface and an array defined on
    it indicating the aneurysm neck.  The function then calculates the aneurysm
    surface area and the area where the WSS is lower than a reference value
    provided by the user.
    """
    try:
        # Try to read if file name is given
        surface = tools.ReadSurface(neck_surface)
    except:
        surface = neck_surface

    # Get aneurysm
    aneurysm = tools.ClipWithScalar(surface, neck_array_name, neck_iso_value)

    # Get aneurysm area
    aneurysmArea = geo.Surface.Area(aneurysm)

    # Get low shear area
    lsaPortion = tools.ClipWithScalar(aneurysm, avgMagWSSArray, lowWSS)
    lsaArea = geo.Surface.Area(lsaPortion)

    return lsaArea/aneurysmArea

# Wallmotion-related functions that operate only on the aneurysm surface
# Refactored here so canbe used directly inside the Aneurysm class
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

        lumenSurface = MarkAneurysmSacManually(
                           lumenSurface,
                           aneurysm_neck_array_name=aneurysm_neck_array_name
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

        self._ostium_normal_vector = self._compute_ostium_normal_vector()

        # Compute ostium surface area
        # Compute areas...
        self._surface_area = geo.Surface.Area(self._aneurysm_surface)
        self._ostium_area = geo.Surface.Area(self._ostium_surface)

        # ... and volume
        self._volume = geo.Surface.Volume(self._cap_aneurysm())

        # Computing hull properties
        self._hull_surface_area = 0.0
        self._hull_volume = 0.0
        self._hull_surface = self._aneurysm_convex_hull()

        # 1D size definitions
        self._neck_diameter = self._compute_neck_diameter()
        self._max_normal_height = self._compute_max_normal_height()
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

    def _make_vtk_id_list(self, it):
        vil = vtk.vtkIdList()
        for i in it:
            vil.InsertNextId(int(i))
        return vil

    def _aneurysm_convex_hull(self):
        """Compute convex hull of closed surface.

        Given an open surface, compute the convex hull set of a surface and
        returns a triangulated surface representation of it.  It uses
        internally the scipy.spatial package.
        """

        # Convert surface points to numpy array
        nPoints = self._aneurysm_surface.GetNumberOfPoints()
        vertices = list()

        for index in range(nPoints):
            vertex = self._aneurysm_surface.GetPoint(index)
            vertices.append(list(vertex))

        vertices = np.array(vertices)

        # Compute convex hull of points
        aneurysmHull = ConvexHull(vertices)

        # Get hull properties
        self._hull_volume = aneurysmHull.volume

        # Need to subtract neck area to
        # compute correct hull surface area
        self._hull_surface_area = aneurysmHull.area - self._ostium_area

        # Intantiate poly data
        polyData = vtk.vtkPolyData()

        # Get points
        points = vtk.vtkPoints()

        for xyzPoint in aneurysmHull.points:
            points.InsertNextPoint(xyzPoint)

        polyData.SetPoints(points)

        # Get connectivity matrix
        cellDataArray = vtk.vtkCellArray()

        for cellId in aneurysmHull.simplices:
            if type(cellId) is np.ndarray:
                cellDataArray.InsertNextCell(self._make_vtk_id_list(cellId))
            else:
                for cell in cellId:
                    cellDataArray.InsertNextCell(self._make_vtk_id_list(cell))

        polyData.SetPolys(cellDataArray)

        return polyData

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

    def _compute_ostium_normal_vector(self):
        """Calculate the normal vector to the aneurysm ostium surface/plane.

        The outwards normal unit vector to the ostium surface is computed by
        summing the normal vectors to each cell of the ostium surface.
        Rigorously, the neck plane vector should be computed with the actual
        neck *plane*, however, there are other ways to compute the aneurysm
        neck that are not based on a plane surface. In this scenario, it is
        robust enough to employ the approach used here because it provides a
        'sense of normal direction' to the neck line, be it a 3D curved path in
        space.

        In any case, if an actual plane is passed, the function will work.
        """

        # Use Numpy
        npSurface = dsa.WrapDataObject(self._ostium_surface)

        # Compute normalized sum of the normals
        neckNormalsVector = npSurface.GetCellData().GetArray(names.normals).sum(axis=0)
        neckNormalsVector /= np.linalg.norm(neckNormalsVector)

        return tuple(neckNormalsVector)

    def _max_normal_height_vector(self):
        """Compute vector along the maximum normal height.

        Compute the vector from the neck contour barycenter and the fartest
        point on the aneurysm surface that have the maximum normal distance
        from the ostium surface normal.
        """

        vecNormal = -1*np.array(self._ostium_normal_vector)
        barycenter = np.array(self._neck_barycenter())

        # Get point in which distance to neck line baricenter is maximum
        maxDistance = const.zero
        maxVertex = None

        nVertices = self._aneurysm_surface.GetPoints().GetNumberOfPoints()

        # Get distance between every point and store it as dict to get maximum
        # distance later
        pointDistances = {}

        for index in range(nVertices):
            # Get surface vertex
            vertex = np.array(self._aneurysm_surface.GetPoint(index))

            # Compute vector joinign barycenter to vertex
            distVector = np.subtract(vertex, barycenter)

            # Compute the normal height
            normalHeight = abs(vtk.vtkMath.Dot(distVector, vecNormal))

            # Convert Np array to tuple (np array is unhashable)
            pointDistances[tuple(distVector)] = normalHeight

        # Get the key with the max item
        maxNHeightVector = max(pointDistances, key=pointDistances.get)

        return maxNHeightVector

    # 1D Size Indices
    def _compute_neck_diameter(self):
        """Return the neck diameter.

        Compute neck diameter, defined as twice the averaged distance between
        the neck contour barycenter and each point on the neck contour.
        """

        return geo.ContourAverageDiameter(self._neck_contour)

    def _compute_max_normal_height(self):
        """Return the maximum normal height."""

        vecMaxHeight = self._max_normal_height_vector()
        vecNormal = self._ostium_normal_vector

        return abs(vtk.vtkMath.Dot(vecMaxHeight, vecNormal))

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

        curvatureArrays = {'Mean': 'Mean_Curvature',
                           'Gauss': 'Gauss_Curvature'}

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
