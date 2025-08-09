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

"""Collection of tools to define and clip a saccular aneurysm sac."""

import sys
import vtk
import numpy as np
import morphman.common as mplib
from morphman.manipulate_curvature import extract_single_line

from vmtk import vtkvmtk
from vmtk import vmtkscripts
from scipy import interpolate

from vtkmodules.numpy_interface import dataset_adapter as dsa

from vmtk4aneurysms.lib import names
from vmtk4aneurysms.lib import centerlines as cl
from vmtk4aneurysms.lib import constants as const
from vmtk4aneurysms.lib import polydatatools as tools
from vmtk4aneurysms.lib import polydatageometry as geo

from vmtk4aneurysms.vascular_classes import (
    VascularSurface,
    VascularCenterline
)

from vmtk4aneurysms.healthy_vessel_reconstruction import (
    LateralAneurysmReconstruction,
    BifurcationAneurysmReconstruction
)

from abc import ABC, abstractmethod

def _transf_normal(
        normal: tuple,
        tilt: float,
        azim: float
    )   -> tuple:
    """Rotates a normal vector to plane by a tilt and azimuth angles."""

    matrix = np.array([
        [ np.cos(azim), -np.sin(azim), const.zero],
        [np.sin(azim)*np.cos(tilt), np.cos(azim)*np.cos(tilt), -np.sin(tilt)],
        [np.sin(azim)*np.sin(tilt), np.cos(azim)*np.sin(tilt), np.cos(tilt)]
    ])

    return tuple(np.dot(matrix, normal))

def ComputeSacCenterlinePoints(
        aneurysm_sac: names.polyDataType,
        distance_array: str
    )   -> tuple:
    """Compute saccular aneurysm sac centerline.

    Compute spline that travels alongs the aneurysm sac from the intersection
    with the parent vessel tube. Its points are defined by the geometric place
    of the barycenters of iso-contours of a 'distance_array' defined on the
    aneurysm surface.

    The function returns a tuple with the spline vertices and tangents in a
    Numpy nd-array.
    """

    # Get wrapper object of vtk numpy interface
    surfaceWrapper = dsa.WrapDataObject(aneurysm_sac)
    distanceArray  = np.array(
                         surfaceWrapper.PointData.GetArray(distance_array)
                     )

    minTubeDist = float(distanceArray.min())
    maxTubeDist = float(distanceArray.max())

    # Build spline along with to perform the neck search
    nPoints     = int(const.oneHundred)
    barycenters = []

    aneurysm_sac.GetPointData().SetActiveScalars(distance_array)

    # Get barycenters of iso-contours
    for isovalue in np.linspace(minTubeDist, maxTubeDist, nPoints):

        # Get isocontour polyline
        isoContour = vtk.vtkContourFilter()
        isoContour.SetInputData(aneurysm_sac)
        isoContour.ComputeScalarsOff()
        isoContour.ComputeNormalsOff()
        isoContour.SetValue(0, isovalue)
        isoContour.Update()

        contour = isoContour.GetOutput()

        contourPoints  = contour.GetPoints()
        nContourPoints = contour.GetNumberOfPoints()
        nContourCells  = contour.GetNumberOfCells()

        if nContourPoints > 0 and nContourCells > 0:
            barycenter = const.nSpatialDimensions*[0]

            vtkvmtk.vtkvmtkBoundaryReferenceSystems.ComputeBoundaryBarycenter(
                contourPoints,
                barycenter
            )

            barycenters.append(barycenter)

    if barycenters:

        # Shift centers to compute interpoint distance
        shiftedBarycenters = [barycenters[0]] + barycenters[0:-1]

        barycenters        = np.array(barycenters)
        shiftedBarycenters = np.array(shiftedBarycenters)

        # Compute distance coordinates
        incrDistances = np.linalg.norm(
                            shiftedBarycenters - barycenters,
                            axis=1
                        )

        distanceCoord = np.cumsum(incrDistances)

        # Find spline of barycenters and get derivative == normals
        limitFraction = const.seven/const.ten

        tck, u = interpolate.splprep(barycenters.T, u=distanceCoord)

        # Max and min t of spline
        # Note that we decrease by two units the start of the spline because I
        # noticed that for some cases the initial point of the spline might be
        # pretty inside the aneurysm, skipping the "neck region"
        minSplineDomain = min(u)
        maxSplineDomain = max(u)

        domain = np.linspace(minSplineDomain, maxSplineDomain, 2*nPoints)

        deriv0 = interpolate.splev(domain, tck, der=0)
        deriv1 = interpolate.splev(domain, tck, der=1)

        # Spline points
        points = np.array(deriv0).T

        # Spline tangents
        tangents = np.array(deriv1).T

        return points, tangents

    else:
        raise ValueError(
                  "No barycenters found for sac centerline construction."
              )

class AneurysmRegionExtractor(ABC):
    """Abstract Base Class for extracting aneurysm-specific regions from the
    vascular surface.

    This strategy encapsulates the logic that varies based on aneurysm type
    (lateral/bifurcation).
    """
    def __init__(
            self,
            vascular_surface: names.polyDataType,
            dome_point: tuple=None,
            healthy_vessel_surface: names.polyDataType=None,
            inlet_centers: list=None,
            outlet_centers: list=None,
        ):
        """Initializes the aneurysm region extractor.

        Arguments:
            vascular_surface -- the vascular surface model containing the
            aneurysm.

            dome_point -- point on the aneurysm surface, used for
            reconstruction; if None, it will be selected interactively.

            healthy_vessel_surface -- optional precomputed healthy vessel
            surface (optional)

            inlet_centers -- list of inlet centers (optional)
            outlet_centers -- list of outlet centers (optional)
        """

        self._vascular_surface = vascular_surface
        self._dome_point = dome_point
        self._healthy_vessel_surface = healthy_vessel_surface
        self._divergence_tolerance = 0.01
        self._inlet_centers = inlet_centers
        self._outlet_centers = outlet_centers

        self._dome_point = tools.SelectSurfacePoint(
                               self._vascular_surface,
                               input_text="Select point on the aneurysm surface\n"
                           ) if dome_point is None else dome_point

        self._inception_surface = None

    @abstractmethod
    def ExtractAneurysmInceptionRegion(self) -> names.polyDataType:
        """Extracts the vessel portion where the aneurysm grew."""
        pass

    @abstractmethod
    def _reconstruct_healthy_vessel(self):
        """Reconstructs the healthy vessel model from the vascular surface."""
        pass

    def _mark_aneurysmal_region(
            self,
            surface_model: names.polyDataType,
            aneurysm_envelope: names.polyDataType,
            parent_tube: names.polyDataType
        )   -> names.polyDataType:
        """Compute the aneurysmal region surface from the original vascular
        model.

        Compute distance between the aneurysm envelope and parent vasculature
        tube function from the original vascular surface model. Mark the
        surface with a field such that its zero value is the difference between
        those two fields. This represents an approximation of the aneurysm neck
        contour.

        Arguments:
            surface_model --  the original vascular surface
            aneuysm_envelope -- the aneurysm surface computed from its Voronoi
            parent_tube -- tube surface of the parent vessel
        """

        # Array names
        tubeToModelArray     = 'ParentTubeModelDistanceArray'
        envelopeToModelArray = 'AneurysmEnvelopeModelDistanceArray'

        # Computes distance between original surface model and the aneurysm
        # envelope, and from the parent tube surface
        aneurysmEnvelopeDistance = tools.ComputeSurfacesDistance(
                                       surface_model,
                                       aneurysm_envelope,
                                       array_name=envelopeToModelArray
                                   )

        modelSurfaceWithDistance = tools.ComputeSurfacesDistance(
                                       aneurysmEnvelopeDistance,
                                       parent_tube,
                                       array_name=tubeToModelArray
                                   )

        # Compute difference between the arrays
        markedAneurysmalRegion = vmtkscripts.vmtkSurfaceArrayOperation()
        markedAneurysmalRegion.Surface         = modelSurfaceWithDistance
        markedAneurysmalRegion.Operation       = 'subtract'
        markedAneurysmalRegion.InputArrayName  = tubeToModelArray
        markedAneurysmalRegion.Input2ArrayName = envelopeToModelArray
        markedAneurysmalRegion.ResultArrayName = names.AneurysmalRegionArrayName
        markedAneurysmalRegion.Execute()

        aneurysmalSurface = markedAneurysmalRegion.Surface

        # Remove unnecessary fields
        aneurysmalSurface.GetPointData().RemoveArray(tubeToModelArray)
        aneurysmalSurface.GetPointData().RemoveArray(envelopeToModelArray)

        return aneurysmalSurface

    def _clip_aneurysm_Voronoi(
            self,
            VoronoiSurface: names.polyDataType,
            tubeSurface: names.polyDataType
        )   -> names.polyDataType:
        """Extract the Voronoi diagram of the aneurysmal portion."""

        # Compute distance between complete Voronoi and the parent vessel tube
        # surface
        DistanceArrayName = 'DistanceToTubeArray'

        VoronoiSurface  = tools.ComputeSurfacesDistance(
                              VoronoiSurface,
                              tubeSurface,
                              array_name=DistanceArrayName
                          )

        aneurysmVoronoi = tools.ClipWithScalar(
                              VoronoiSurface,
                              DistanceArrayName,
                              const.zero
                          )

        aneurysmVoronoi = tools.ExtractConnectedRegion(
                              aneurysmVoronoi,
                              'largest'
                          )

        return tools.Cleaner(aneurysmVoronoi)

    def ExtractAneurysmalRegion(self) -> names.polyDataType:
        """Marks the aneurysmal region with an array and extracts the vessel
        portion where the aneurysm grew.

        Based on the five first steps of Piccinelli's procedure, this function
        marks the vascular model passed with a field named
        'AneurysmalRegionArray' whose zero value marks the contour of the
        aneurysmal region. This part is common to both lateral and bifurcation
        once the inception region is determined.
        """
        # 1) Compute vasculature's Voronoi
        vascularVoronoi = cl.ComputeVoronoiDiagram(self._vascular_surface)

        # 2) Compute parent vasculature centerline tube
        if self._healthy_vessel_surface is None:

            # Compute healthy vasculature
            self._reconstruct_healthy_vessel()

        # Compute the centerline of the parent vascular surface
        parentCenterlines = VascularCenterline.GenerateCenterlines(
                                self._healthy_vessel_surface
                            )

        parentTubeSurface = cl.ComputeTubeSurface(parentCenterlines)

        # 3) Aneurysm Voronoi isolation
        aneurysmVoronoi   = self._clip_aneurysm_Voronoi(
                                vascularVoronoi,
                                parentTubeSurface
                            )

        aneurysmEnvelope  = cl.ComputeVoronoiEnvelope(aneurysmVoronoi)

        # 4) Aneurysmal surface isolation
        aneurysmalSurface = self._mark_aneurysmal_region(
                                self._vascular_surface,
                                aneurysmEnvelope,
                                parentTubeSurface
                            )

        return aneurysmalSurface

    def GetDomePoint(self) -> tuple:
        """Returns the aneurysm dome point."""

        return self._dome_point

class LateralAneurysmRegionExtractor(AneurysmRegionExtractor):
    """Concrete strategy for extracting a lateral-aneurysm-influenced region."""

    def __init__(
            self,
            vascular_surface: names.polyDataType,
            dome_point: tuple=None,
            healthy_vessel_surface: names.polyDataType=None,
            inlet_centers: list=None,
            outlet_centers: list=None,
        ):
        """Initializes the aneurysm region extractor.

        Arguments:
            vascular_surface -- the vascular surface model containing the
            aneurysm.

            dome_point -- point on the aneurysm surface, used for
            reconstruction; if None, it will be selected interactively.

            healthy_vessel_surface -- optional precomputed healthy vessel
            surface (optional)

            inlet_centers -- list of inlet centers (optional)
            outlet_centers -- list of outlet centers (optional)
        """

        super().__init__(
            vascular_surface,
            dome_point=dome_point,
            healthy_vessel_surface=healthy_vessel_surface,
            inlet_centers=inlet_centers,
            outlet_centers=outlet_centers
        )

        self._upstream_clip_point = None
        self._downstream_clip_point = None

    def _reconstruct_healthy_vessel(self):

        latStrategy = LateralAneurysmReconstruction(
                            self._vascular_surface,
                            dome_point=self._dome_point
                      )

        self._healthy_vessel_surface = latStrategy.Reconstruct()

    def ExtractAneurysmInceptionRegion(self) -> names.polyDataType:
        """Extract vessel portion where a lateral aneurysm grew.

        Given the vascular model surface with open inlet and outlet profiles,
        extract the portion of the tube surface where the aneurysm grew by
        calculating the divegence points of the centerlines. The user must
        select a point on the aneurysm's dome surface.

        Note that the algorithm will use the first outlet to compute the
        centerlines, so avoid any outlet profile between the inlet and the
        aneurysm region.
        """

        if self._inception_surface is None:
            if self._inlet_centers is None and self._outlet_centers is None:

                inletRefs, outletRefs = VascularSurface.ComputeOpenCenters(
                                            self._vascular_surface
                                        )

                self._inlet_centers = list(inletRefs.keys())
                self._outlet_centers = list(outletRefs.keys())

            # One inlet and one outlet (although the model can have more than one outlet),
            # lateral aneurysm
            # 1 -> "forward" centerline, inlet -> outlet, and aneurysm
            # 2 -> "backward" centerline, outlet -> inlet, with aneurysm
            # Note: the aneurysm is like a bifurcation, in this case

            relevantOutlets = self._outlet_centers[0:1]

            forwardCenterline = VascularCenterline.GenerateCenterlines(
                                    self._vascular_surface,
                                    self._inlet_centers,
                                    relevantOutlets + [self._dome_point]
                                )

            backwardCenterline = VascularCenterline.GenerateCenterlines(
                                    self._vascular_surface,
                                    relevantOutlets,
                                    self._inlet_centers + [self._dome_point]
                                )

            self._upstream_clip_point = cl.GetDivergingPoint(
                                            forwardCenterline,
                                            self._divergence_tolerance
                                        )

            self._downstream_clip_point = cl.GetDivergingPoint(
                                              backwardCenterline,
                                              self._divergence_tolerance
                                          )

            # Clip centerline portion of the forward centerline
            line = extract_single_line(forwardCenterline, 0)
            loc  = mplib.vtk_wrapper.get_vtk_point_locator(line)

            #Find closest points to clipping on parent centerline
            upstreamId   = loc.FindClosestPoint(self._upstream_clip_point)
            downstreamId = loc.FindClosestPoint(self._downstream_clip_point)

            aneurysmInceptionClPortion =  extract_single_line(
                                              line,
                                              0,
                                              start_id=upstreamId,
                                              end_id=downstreamId
                                          )

            self._inception_surface = cl.ComputeTubeSurface(
                                          aneurysmInceptionClPortion
                                      )

        return self._inception_surface

    def GetUpstreamClipPoint(self) -> tuple:
        """Returns the upstream clipping point of the aneurysm."""
        if self._upstream_clip_point is None:
            self._inception_surface = self.ExtractAneurysmInceptionRegion()

        return self._upstream_clip_point

    def GetDownstreamClipPoint(self) -> tuple:
        """Returns the downstream clipping point of the aneurysm."""
        if self._downstream_clip_point is None:
            self._inception_surface = self.ExtractAneurysmInceptionRegion()

        return self._downstream_clip_point

class BifurcationAneurysmRegionExtractor(AneurysmRegionExtractor):
    """Concrete strategy for extracting a bifurcation-aneurysm-influenced
    region."""

    def __init__(
            self,
            vascular_surface: names.polyDataType,
            dome_point: tuple=None,
            healthy_vessel_surface: names.polyDataType=None,
            inlet_centers: list=None,
            outlet_centers: list=None,
        ):
        """Initializes the aneurysm region extractor.

        Arguments:
            vascular_surface -- the vascular surface model containing the
            aneurysm.

            dome_point -- point on the aneurysm surface, used for
            reconstruction; if None, it will be selected interactively.

            healthy_vessel_surface -- optional precomputed healthy vessel
            surface (optional)

            inlet_centers -- list of inlet centers (optional)
            outlet_centers -- list of outlet centers (optional)
        """

        super().__init__(
            vascular_surface,
            dome_point=dome_point,
            healthy_vessel_surface=healthy_vessel_surface,
            inlet_centers=inlet_centers,
            outlet_centers=outlet_centers
        )

        self._bifurcation_clip_point = None
        self._daughter_branchs_clip_points = []

    def _reconstruct_healthy_vessel(self):

        bifStrategy = BifurcationAneurysmReconstruction(
                            self._vascular_surface,
                            dome_point=self._dome_point
                      )

        self._healthy_vessel_surface = bifStrategy.Reconstruct()

    def ExtractAneurysmInceptionRegion(self) -> names.polyDataType:
        """Extract vessel portion where a bifurcation aneurysm grew.

        Given the vascular model surface with open inlet and outlet profiles,
        extract the portion of the tube surface where the aneurysm grew by
        calculating the divegence points of the centerlines. The user must
        select a point on the aneurysm's dome surface.

        Note that the algorithm will use the two first outlets to compute the
        centerlines, so avoid any outlet profile between the inlet and the
        aneurysm.
        """
        if self._inception_surface is None:
            if self._inlet_centers is None and self._outlet_centers is None:

                inletRefs, outletRefs = VascularSurface.ComputeOpenCenters(
                                            self._vascular_surface
                                        )

                self._inlet_centers = list(inletRefs.keys())
                self._outlet_centers = list(outletRefs.keys())

            # One inlet and two outlets, bifurcation with the aneurysm
            # 1 -> centerline of the branches only
            # 2 -> centerline of the first outlet to the aneurysm and inlet
            # 3 -> centerline of the second outlet to the aneurysm and inlet
            relevantOutlets = self._outlet_centers[0:2]

            clWithoutAneurysm = VascularCenterline.GenerateCenterlines(
                                    self._vascular_surface,
                                    self._inlet_centers,
                                    self._outlet_centers
                                )

            self._bifurcation_clip_point= cl.GetDivergingPoint(
                                              clWithoutAneurysm,
                                              self._divergence_tolerance
                                          )

            lines = []
            for cl_id, outlet in enumerate(relevantOutlets):
                daughterCenterline = VascularCenterline.GenerateCenterlines(
                                         self._vascular_surface,
                                         [outlet],
                                         self._inlet_centers + [self._dome_point]
                                     )
                # Get clipping point on this branch
                dauClippingPoint = cl.GetDivergingPoint(
                                       daughterCenterline,
                                       self._divergence_tolerance
                                   )

                self._daughter_branchs_clip_points.append(dauClippingPoint)

                # Then clip the parent centerline
                line = extract_single_line(clWithoutAneurysm, cl_id)

                loc = mplib.vtk_wrapper.get_vtk_point_locator(line)

                #Find closest points to clipping on parent centerline
                dauId = loc.FindClosestPoint(dauClippingPoint)
                bifId = loc.FindClosestPoint(self._bifurcation_clip_point)

                lines.append(
                    extract_single_line(
                        line,
                        0,
                        start_id=bifId,
                        end_id=dauId
                    )
                )

            aneurysmInceptionClPortion = mplib.vtk_wrapper.vtk_merge_polydata(lines)

            self._inception_surface = cl.ComputeTubeSurface(
                                          aneurysmInceptionClPortion
                                      )

        return self._inception_surface

    def GetBifurcationClipPoint(self) -> tuple:
        """Returns the bifurcation clipping point of the aneurysm."""
        if self._bifurcation_clip_point is None:
            self._inception_surface = self.ExtractAneurysmInceptionRegion()

        return self._bifurcation_clip_point

    def GetDaughterBranchsClipPoints(self) -> list:
        """Returns the daughter branches clipping points of the aneurysm."""
        if not self._daughter_branchs_clip_points:
            self._inception_surface = self.ExtractAneurysmInceptionRegion()

        return self._daughter_branchs_clip_points

class AneurysmNeckIdentificationStrategy(ABC):
    """Abstract Base Class for aneurysm neck identification strategies.

    Each strategy will compute and add the 'DistanceToNeck' field to the
    surface. This field is zero on the aneurysm neck contour and elsewhere it
    marks the geodesic distance to the neck. Negative values are used inside
    the aneurysm sac and positive values outside the aneurysm neck contour.

    .. warning::
        Negative distance values are used inside the aneurysm neck contour
        (i.e., it marks the aneurysm sac) and positive values elsewhere.

    .. warning::
        Better results are expected if you "reduce" the vascular surface to
        only the region where the aneurysm is, ie to clip the surface so only
        the parent vessel and the daughter branches are left.

    .. warning::
        The algorithms involved in this class computations are much faster if
        triangular meshes are used. Hence, it is recommended to triangulate
        the input vascular surface before using this class.
    """
    def __init__(
            self,
            vascular_surface: names.polyDataType,
            aneurysm_extractor: AneurysmRegionExtractor=None,
            distance_to_neck_field_name: str=names.DistanceToNeckArrayName
        ):

        self._vascular_surface = vascular_surface
        self._aneurysm_extractor = aneurysm_extractor
        self._distance_to_neck_field_name = distance_to_neck_field_name

        # These will be set by the concrete strategies
        self._marked_surface = None
        self._sac_surface = None
        self._vascular_surface_no_aneurysm = None

    @abstractmethod
    def MarkAneurysmNeck(self) -> names.polyDataType:
        """Identifies the aneurysm neck and marks the surface with
        'DistanceToNeck' array.

        Returns the vascular surface with the 'DistanceToNeck' array.
        """
        pass

    def ClipSac(self) -> tuple[names.polyDataType, names.polyDataType]:
        """Clips the aneurysm sac from the vascular surface model.

        Returns the aneurysm sac surface and the vascular surface without the
        aneurysm.
        """
        pass

class InteractiveNeckIdentification(AneurysmNeckIdentificationStrategy):
    """Strategy for interactively marking the aneurysm neck contour.

    Given a vasculature with an aneurysm, prompt the user to manually draw the
    aneurysm neck on the surface. An scalar array (field) is then defined on
    the surface with value 0 on the aneurysm neck contour defined and its other
    values as the geodesic distance to the neck contour.
    """
    def __init__(
            self,
            vascular_surface: names.polyDataType,
            distance_to_neck_field_name: str=names.DistanceToNeckArrayName
        ):

        super().__init__(
            vascular_surface,
            aneurysm_extractor=None,
            distance_to_neck_field_name=distance_to_neck_field_name
        )

    def MarkAneurysmNeck(
            self,
            screen_msg: str="Mark, interactively, the neck contour\n"
        ) -> names.polyDataType:

        # For optimization, check whether the field was already computed
        if self._marked_surface is None:
            print("Executing InteractiveNeckIdentification strategy...")
            surface = tools.Cleaner(self._vascular_surface)

            getContour = tools.SelectContourPointsIds()
            getContour.Surface = surface
            getContour.ScreenInfo = screen_msg
            getContour.Execute()

            surface = geo.SurfaceGeodesicDistanceToContour(
                          surface,
                          getContour.ContourIds,
                          gdistance_array_name=self._distance_to_neck_field_name
                      )

            # Smooth the computed distance field
            self._marked_surface = tools.SmoothSurfacePointField(
                                       surface,
                                       self._distance_to_neck_field_name,
                                       niterations=10
                                   )

        return self._marked_surface

    def ClipSac(
            self
        )   -> tuple[names.polyDataType, names.polyDataType]:
        """Clips the aneurysm sac from the vascular surface model.

        Returns the aneurysm sac surface and the vascular surface without the
        aneurysm.
        """
        if self._sac_surface is None:
            # The strategy is responsible for marking the surface with
            # DistanceToNeckArrayName
            marked_surface = self.MarkAneurysmNeck()

            # Perform the actual clipping based on the 'DistanceToNeck' array
            self._sac_surface = tools.ClipWithScalar(
                                       marked_surface,
                                       self._distance_to_neck_field_name,
                                       const.zero,
                                       inside_out=True
                                   )

            self._vascular_surface_no_aneurysm = tools.ClipWithScalar(
                                                     marked_surface,
                                                     self._distance_to_neck_field_name,
                                                     const.zero,
                                                     inside_out=False
                                                 )

            # Clean up the temporary array from the clipped surface
            # Leave it on the aneurysm surface
            if self._vascular_surface_no_aneurysm.GetPointData().HasArray(
                    self._distance_to_neck_field_name
                ):

                self._vascular_surface_no_aneurysm.GetPointData().RemoveArray(
                    self._distance_to_neck_field_name
                )

        return self._sac_surface, self._vascular_surface_no_aneurysm

class Automatic3DNeckIdentification(AneurysmNeckIdentificationStrategy):
    """Strategy for automatically identifying the aneurysm neck using geodesic
    distance to the aneurysmal region's boundary.
    """

    def __init__(
            self,
            vascular_surface: names.polyDataType,
            aneurysm_extractor: AneurysmRegionExtractor,
            distance_to_neck_field_name: str=names.DistanceToNeckArrayName
        ):

        super().__init__(
            vascular_surface,
            aneurysm_extractor,
            distance_to_neck_field_name=distance_to_neck_field_name
        )

    def MarkAneurysmNeck(self) -> names.polyDataType:
        """Automatically marks a 3D contour as the neck of an aneurysm.

        Based on the five first steps of Piccinelli's procedure, this function
        marks the vascular model passed with an array whose zero value marks
        the contour of the aneurysmal region. The rest of the array is given by
        the geodesic distance of the point to the neck contour.

        .. warning::
            Negative distance values are used inside the aneurysm neck contour
            (i.e., it marks the aneurysm sac) and positive values elsewhere.

        .. warning::
            Better results are expected if you "reduce" the vascular surface to
            only the region where the aneurysm is, ie clip the surface so only
            the parent vessel and the daughter branches are left.

        Return
        surface (vtkPolyData) -- vascular surface with an array defined on it
        marking the aneurysm neck contour.
        """
        if self._marked_surface is None:
            surface = tools.Cleaner(self._vascular_surface)

            # Delegate aneurysm region extraction to the specific extractor strategy
            aneurysmalSurface = self._aneurysm_extractor.ExtractAneurysmalRegion()

            # Project aneurysmal region array
            surface = tools.ProjectPointArray(
                          surface,
                          aneurysmalSurface,
                          names.AneurysmalRegionArrayName
                      )

            # Add a little bit of smoothing to the Distance field to remove corner
            # discontnuities
            surface = tools.SmoothSurfacePointField(
                           surface,
                           names.AneurysmalRegionArrayName,
                           niterations=5
                       )

            # The best approach I found to extract the closest path with the
            # surface model points was through the clip: the clip used
            # subsequentely wtih the boundary extractor provides a set of points
            # that are ORIENTED along the neck line. On the other hand, The initial
            # tests I did were with the contour filter, which, as far as I could
            # assess, generates a polyline that does not have its points oriented
            # along the path, which inhibited the use of the selection filter to
            # get the aneurysmal region and change the sign of the geodeseic
            # distance to neck array (note, the coumputation of the geodesic
            # distance per se did not require the points to be oriented).
            aneurysmalSurface = tools.ClipWithScalar(
                                    surface,
                                    names.AneurysmalRegionArrayName,
                                    const.zero
                                )

            surface.GetPointData().RemoveArray(
                names.AneurysmalRegionArrayName
            )

            if self._aneurysm_extractor.GetDomePoint():
                aneurysmalSurface = tools.ExtractConnectedRegion(
                                        aneurysmalSurface,
                                        "closest",
                                        closest_point=self._aneurysm_extractor.GetDomePoint()
                                    )

            else:
                aneurysmalSurface = tools.ExtractConnectedRegion(
                                        aneurysmalSurface,
                                        "largest"
                                    )

            # Extract the bounday of the cutted cells
            # This provides a rough approximation of where the neck contour cuts
            # the surface
            boundaryExtractor = vtkvmtk.vtkvmtkPolyDataBoundaryExtractor()
            boundaryExtractor.SetInputData(aneurysmalSurface)
            boundaryExtractor.Update()

            neckContour = boundaryExtractor.GetOutput()

            # Locator to find closest points
            pointIds = tools.GetClosestContourOnSurface(
                            surface,
                            neckContour
                        )

            # Compute the geodesic distance  from the approximate neck contour
            surface = geo.SurfaceGeodesicDistanceToContour(
                           surface,
                           pointIds,
                           gdistance_array_name=self._distance_to_neck_field_name
                       )

            # Smooth the computed distance field
            self._marked_surface = tools.SmoothSurfacePointField(
                                       surface,
                                       self._distance_to_neck_field_name,
                                       niterations=10
                                   )

        return self._marked_surface

    def ClipSac(
            self
        )   -> tuple[names.polyDataType, names.polyDataType]:
        """Clips the aneurysm sac from the vascular surface model.

        Returns the aneurysm sac surface and the vascular surface without the
        aneurysm.
        """
        if self._sac_surface is None:
            # The strategy is responsible for marking the surface with
            # DistanceToNeckArrayName
            marked_surface = self.MarkAneurysmNeck()

            # Perform the actual clipping based on the 'DistanceToNeck' array
            self._sac_surface = tools.ClipWithScalar(
                                       marked_surface,
                                       self._distance_to_neck_field_name,
                                       const.zero,
                                       inside_out=True
                                   )

            self._vascular_surface_no_aneurysm = tools.ClipWithScalar(
                                                     marked_surface,
                                                     self._distance_to_neck_field_name,
                                                     const.zero,
                                                     inside_out=False
                                                 )

            # Clean up the temporary array from the clipped surface
            # Leave it on the aneurysm surface to be used later for other
            # calculations
            if self._vascular_surface_no_aneurysm.GetPointData().HasArray(
                    self._distance_to_neck_field_name
                ):

                self._vascular_surface_no_aneurysm.GetPointData().RemoveArray(
                    self._distance_to_neck_field_name
                )

        return self._sac_surface, self._vascular_surface_no_aneurysm

class PlaneNeckIdentification(AneurysmNeckIdentificationStrategy):
    """Strategy for identifying the aneurysm neck using a plane-based approach.

    This strategy returns the surface with the 'DistanceToNeck' array marked
    based on the discovered neck plane.
    """
    def __init__(
            self,
            vascular_surface: names.polyDataType,
            aneurysm_extractor: AneurysmRegionExtractor,
            distance_to_neck_field_name: str=names.DistanceToNeckArrayName
        ):

        super().__init__(
            vascular_surface,
            aneurysm_extractor,
            distance_to_neck_field_name=distance_to_neck_field_name
        )

        self._neck_plane = None
        self._neck_center = None
        self._neck_normal = None

    def _search_neck_plane(
            self,
            aneurysm_sac: names.polyDataType,
            centers: np.ndarray,
            normals: np.ndarray,
            min_variable="area"
        )   -> names.planeType:
        """Search neck plane of aneurysm by minimizing a contour variable.

        This function effectively searches for the aneurysm neck plane: it
        interactively cuts the aneurysm surface with planes defined by the vertices
        and normals to a spline travelling through the aneurysm sac.

        The cut plane is further precessed by a tilt and azimuth angle and the
        minimum search between them, as originally proposed by Piccinelli et al.
        (2009).

        It returns the local minimum solution: the neck plane as a vtkPlane object.
        """

        # For each center on the sac centerline (list), create the rotated and
        # tilted plane normals (list) and compute its area (or min_variable)

        # Rotation angles: from original work
        tiltIncr = const.two
        azimIncr = const.ten
        tiltMax = 32
        azimMax = 360

        tilts = np.arange(const.zero, tiltMax, tiltIncr)*const.degToRad
        azims = np.arange(const.zero, azimMax, azimIncr)*const.degToRad

        globalMinimumAreas = {} # can be used for debug
        previousArea = 0.0

        # These normals point to the aneurysm direction
        for center, normal in zip(map(tuple, centers), map(tuple, normals)):

            # More readable option
            planeContours = {(tilt, azim): tools.ContourCutWithPlane(
                                              aneurysm_sac,
                                              center,
                                              _transf_normal(normal, tilt, azim)
                                          )
                             for tilt in tilts for azim in azims}

            # Compute area of the closed contours for each normal direction
            planeSectionAreas = {key: geo.ContourPerimeter(contour) \
                                     if min_variable == "perimeter" \
                                     else geo.ContourPlaneArea(contour)
                                 for key, contour in planeContours.items()
                                 if contour.GetNumberOfCells() > 0 and \
                                    geo.ContourIsClosed(contour)}

            if planeSectionAreas:
                # Get the normal direction of max. area
                minCenter    = center
                minDirection = min(planeSectionAreas, key=planeSectionAreas.get)
                minPlaneArea = min(planeSectionAreas.values())
                minPlaneNormal = _transf_normal(normal, *minDirection)

                # Associate this with each center
                # globalMinimumAreas.update({
                #     center: {
                #         "normal": minPlaneNormal,
                #         "area"  : minPlaneArea
                #     }
                # })

                if minPlaneArea <= previousArea:
                    previousArea = minPlaneArea
                    continue

                else:
                    break

            else:
                continue

        # Create plane
        neckPlane = vtk.vtkPlane()
        neckPlane.SetOrigin(minCenter)
        neckPlane.SetNormal(minPlaneNormal)

        return neckPlane

    def MarkAneurysmNeck(self) -> names.polyDataType:
        """Automatically marks a plane neck of an aneurysm.

        Based on the five first steps of Piccinelli's procedure, this function
        marks the vascular model passed with an array whose zero value marks
        the plane contour of the aneurysmal region. The rest of the array is
        given by the geodesic distance of the point to the neck contour.

        .. warning::
            Negative distance values are used inside the aneurysm neck contour
            (i.e., it marks the aneurysm sac) and positive values elsewhere.

        .. warning::
            Better results are expected if you "reduce" the vascular surface to
            only the region where the aneurysm is, ie clip the surface so only
            the parent vessel and the daughter branches are left.

        Return
        surface (vtkPolyData) -- vascular surface with an array defined on it
        marking the aneurysm neck contour.
        """
        if self._marked_surface is None:

            surface = tools.Cleaner(self._vascular_surface)

            # Delegate aneurysm region extraction to the specific extractor
            # strategy
            aneurysmalSurface = self._aneurysm_extractor.ExtractAneurysmalRegion()

            # Clip aneurysmal portion (scalars < 0)
            clippedAneurysmalSurface = tools.ClipWithScalar(
                                           aneurysmalSurface,
                                           names.AneurysmalRegionArrayName,
                                           const.zero
                                       )

            clippedAneurysmalSurface = tools.ExtractConnectedRegion(
                                           clippedAneurysmalSurface,
                                           'largest'
                                       )

            clippedAneurysmalSurface.GetPointData().RemoveArray(
                names.AneurysmalRegionArrayName
            )

            # Get the portion where the aneurysm grew
            aneurysmInceptionPortion = self._aneurysm_extractor.ExtractAneurysmInceptionRegion()

            # The authors of the study used the distance to the clipped tube
            # surface to compute the sac centerline. I am currently using the same
            # array used to clip the aneurysmal region
            tubeToAneurysmDistance = "ClippedTubeToAneurysmDistanceArray"

            aneurysmalSurface = tools.ComputeSurfacesDistance(
                                    tools.Cleaner(clippedAneurysmalSurface),
                                    aneurysmInceptionPortion,
                                    array_name=tubeToAneurysmDistance,
                                    signed_array=False
                                )

            # Create sac centerline
            barycenters, normals = ComputeSacCenterlinePoints(
                                       aneurysmalSurface,
                                       tubeToAneurysmDistance
                                   )

            aneurysmalSurface.GetPointData().RemoveArray(
                tubeToAneurysmDistance
            )

            # Search neck plane
            self._neck_plane = self._search_neck_plane(
                                    aneurysmalSurface,
                                    barycenters,
                                    normals,
                                    min_variable="area"
                                )

            # 8) Detach aneurysm sac from parent vasculature
            # It is impotant to clip the aneurysm here to clip only the aneurysmal
            # region surface
            self._neck_center = self._neck_plane.GetOrigin()
            self._neck_normal = self._neck_plane.GetNormal()

            # Clip final aneurysm surface: the side to where the normal point
            planeContour = tools.ContourCutWithPlane(
                               aneurysmalSurface,
                               self._neck_center,
                               self._neck_normal
                           )

            # Get point on the plane contour closest to the neck center
            locator = vtk.vtkPointLocator()
            locator.SetDataSet(planeContour)
            locator.BuildLocator()

            seedPointId = locator.FindClosestPoint(self._neck_center)
            seedPoint = planeContour.GetPoint(seedPointId)

            # Create the neck plane marker directly on the VASCULAR surface
            vascular_surface = tools.SeamPlaneTubularStructureMarker(
                                    surface,
                                    self._neck_center,
                                    self._neck_normal,
                                    seed_point=seedPoint,
                                    seam_scalar_array_name=names.SeamScalarsArrayName
                                )

            # Clip the aneurysm sac surface with the neck plane
            # This will provide the aneurysm sac surface here as the clip after
            # the creation of the DistanceToNeck field render the neck plane
            # not a plane
            self._sac_surface = tools.ClipWithScalar(
                                     vascular_surface,
                                     names.SeamScalarsArrayName,
                                     const.zero,
                                     inside_out=False
                                 )

            self._vascular_surface_no_aneurysm = tools.ClipWithScalar(
                                                     vascular_surface,
                                                     names.SeamScalarsArrayName,
                                                     const.zero,
                                                     inside_out=True
                                                 )

            # The next algorithm NEEDS to use a poly data result of a CLIP!
            # The best approach I found to extract the closest path with the surface
            # model points was through the clip: the clip used subsequentely wtih the
            # boundary extractor provides a set of points that are ORIENTED along the
            # neck line.
            # On the other hand, The initial tests I did were with the contour
            # filter, which, as far as I could assess, generates a polyline that
            # does not have its points oriented along the path, which inhibited the
            # use of the selection filter to get the aneurysmal region and change
            # the sign of the geodeseic distance to neck array (note, the
            # coumputation of the geodesic distance per se did not require the
            # points to be oriented).

            # Extract the bounday of the cutted cells
            # This provides a rough approximation of where the neck contour
            # cuts the surface
            boundaryExtractor = vtkvmtk.vtkvmtkPolyDataBoundaryExtractor()
            boundaryExtractor.SetInputData(self._sac_surface)
            boundaryExtractor.Update()

            neckContour = boundaryExtractor.GetOutput()

            # Locator to find closest points
            pointIds = tools.GetClosestContourOnSurface(
                           surface,
                           neckContour
                       )

            # Compute the geodesic distance  from the approximate neck contour
            self._marked_surface = geo.SurfaceGeodesicDistanceToContour(
                                       surface,
                                       pointIds,
                                       gdistance_array_name=self._distance_to_neck_field_name
                                   )

        return self._marked_surface

    def ClipSac(
            self
        )   -> tuple[names.polyDataType, names.polyDataType]:
        """Clips the aneurysm sac from the vascular surface model.

        Returns the aneurysm sac surface and the vascular surface without the
        aneurysm.
        """
        if self._sac_surface is None:
            # In this strategy, this method already clips the aneurysm
            marked_surface = self.MarkAneurysmNeck()

        # Clean up the temporary array from the clipped surface
        # Leave it on the aneurysm surface to be used later for other
        # calculations
        if self._vascular_surface_no_aneurysm.GetPointData().HasArray(
                self._distance_to_neck_field_name
            ):

            self._vascular_surface_no_aneurysm.GetPointData().RemoveArray(
                self._distance_to_neck_field_name
            )

        return self._sac_surface, self._vascular_surface_no_aneurysm

# --- FACTORY/INTERFACE FOR THE USER ---
def ClipAneurysmSacSurface(
        vascular_surface: names.polyDataType,
        aneurysm_type: str,
        mode: str="automatic",
        healthy_vessel_surface: names.polyDataType=None,
        dome_point: tuple=None
    ) -> tuple:
    """Clip the aneurysm sac surface from the vascular surface model.

    Given the vascular model with an aneurysm, clip the aneurysm sac surface
    based on the neck contour computed via three alternative strategies.
    Returns a tuple with the aneurysm and the rest of the surface clipped. The
    sac surface is clipped based on the 'names.DistanceToNeckArrayName' array,
    which is kept on the surface polydata for later use.

    Arguments
    ---------
    vascular_surface (names.polyDataType) -- the original vasculature surface
    with the aneurysm

    Optional
    --------
    mode (str, default: 'interactive') -- the method to clip the aneurysm:
    'interactive', 'automatic', or 'plane'

    healthy_vessel_surface (names.polyDataType, default: None) --
    reconstructed parent vasculature

    aneurysm_type (str, default: "", ["lateral", "bifurcation"]): mandatory if
    'mode' is 'automatic' and 'parent_vascular_surface' is not passed, because
    it is used in its computation

    dome_point (tuple, default: None) -- point on the aneurysm surface,

    Return
    (aneurysm sac surface, clipped surface) (tuple) -- the surface of the
    aneurysm sac clipped from the vascular surface model and the vascular model
    surface clipped.
    """
    # First, determine the AneurysmRegionExtractor strategy based on
    # aneurysm_type
    if mode != "interactive":
        if aneurysm_type == "lateral":
            region_extractor = LateralAneurysmRegionExtractor(
                                    vascular_surface,
                                    dome_point=dome_point,
                                    healthy_vessel_surface=healthy_vessel_surface
                                )

        elif aneurysm_type == "bifurcation":
            region_extractor = BifurcationAneurysmRegionExtractor(
                                    vascular_surface,
                                    dome_point=dome_point,
                                    healthy_vessel_surface=healthy_vessel_surface
                                )

        else:
            raise ValueError(
                    f"Aneurysm type must be 'lateral' or 'bifurcation'. '{aneurysm_type}' passed."
                )

    # Then, determine the AneurysmNeckIdentificationStrategy based on mode
    if mode == "interactive":
        neckClipperStrategy = InteractiveNeckIdentification(
                            vascular_surface
                        )

    elif mode == "automatic":
        neckClipperStrategy = Automatic3DNeckIdentification(
                            vascular_surface,
                            region_extractor
                        )

    elif mode == "plane":
        neckClipperStrategy = PlaneNeckIdentification(
                            vascular_surface,
                            region_extractor
                        )

    else:
        raise ValueError(
                    f"""Neck computation mode either 'interactive', 'automatic', or 'plane'; {mode} passed."""
            )

    return neckClipperStrategy.ClipSac()

def ComputeGeodesicDistanceToAneurysmNeck(
        vascular_surface: names.polyDataType,
        mode: str="automatic", # Changed default for demonstration
        healthy_vessel_surface: names.polyDataType=None,
        aneurysm_type: str="lateral", # Added default for demonstration
        dome_point: tuple=None
    ) -> names.polyDataType:
    """Mark the aneurysm neck contour and compute the geodesic distance to it.

    Given a vascular surface with an aneurysm, computes the geodesic distance
    to the aneurysm neck by three different methods:

        *   'interactive': the user is prompted to interactively draw the
            aneurysm contour;

        *   'automatic': automatically marks a 3D contour on the aneurysm
            surface that separates the sac from the vasculature based on the
            procedure described in Piccinelli's publication.

        *   'plane': computes an approximate neck plane, based on Piccinelli's
            publication.

    The automatic mode uses the hypothetically healthy vessel surface to
    estimate the neck contour, if this surface is not passed, it will be
    computed.

    .. warning::
        Adds a little bit of smoothing in the resulting distance field to avoid
        discontnuities in the original field due to its dependency on the
        underlying discretization of the surface. For the 'plane' mode, it may
        distance the neck contour plane from an actual plane.
    """
    # First, determine the AneurysmRegionExtractor strategy based on
    # aneurysm_type
    if mode != "interactive":
        if aneurysm_type == "lateral":
            region_extractor = LateralAneurysmRegionExtractor(
                                    vascular_surface,
                                    dome_point=dome_point,
                                    healthy_vessel_surface=healthy_vessel_surface
                                )

        elif aneurysm_type == "bifurcation":
            region_extractor = BifurcationAneurysmRegionExtractor(
                                    vascular_surface,
                                    dome_point=dome_point,
                                    healthy_vessel_surface=healthy_vessel_surface
                                )

        else:
            raise ValueError(
                    f"Aneurysm type must be 'lateral' or 'bifurcation'. '{aneurysm_type}' passed."
                )

    # Then, determine the AneurysmNeckIdentificationStrategy based on mode
    if mode == "interactive":
        neckClipperStrategy = InteractiveNeckIdentification(
                                    vascular_surface
                                )

    elif mode == "automatic":
        neckClipperStrategy = Automatic3DNeckIdentification(
                            vascular_surface,
                            region_extractor
                        )

    elif mode == "plane":
        neckClipperStrategy = PlaneNeckIdentification(
                            vascular_surface,
                            region_extractor
                        )

    else:
        raise ValueError(
                    f"""Neck computation mode either 'interactive', 'automatic', or 'plane'; {mode} passed."""
            )

    return neckClipperStrategy.MarkAneurysmNeck()
