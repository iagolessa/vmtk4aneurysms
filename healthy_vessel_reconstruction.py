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

import sys
import vtk
import numpy as np
import morphman.common as mplib

from abc import ABC, abstractmethod
from morphman.manipulate_curvature import extract_single_line
from vtkmodules.numpy_interface import dataset_adapter as dsa

from vmtk import vtkvmtk
from vmtk import vmtkscripts
from scipy.spatial import ConvexHull

# Local modules
from vmtk4aneurysms.lib import names
from vmtk4aneurysms.lib import constants as const
from vmtk4aneurysms.lib import centerlines as cl
from vmtk4aneurysms.lib import polydatatools as tools
from vmtk4aneurysms.lib import polydatageometry as geo
from vmtk4aneurysms.lib import polydatamath as pmath

from vmtk4aneurysms.vascular_classes import (
    VascularSurface,
    VascularCenterline
)


def _set_portion_in_cl_patch(
        surface: names.polyDataType,
        patch_centerline: names.polyDataType,
        patch_id: int,
        filter_array_name: str
    ) -> names.polyDataType:
    """Marks a surface's portions that are within a centerline patch.

    Given a surface with an array of zeros defined on it, change its values to
    one where the surface lies within the bound of the polyball function
    defined on a centerlines patch of the same surface. The radius array must
    also be defined on the surface.
    """

    # Convert surface and create array
    nPoints = surface.GetNumberOfPoints()
    npSurface = dsa.WrapDataObject(surface)
    pointData = npSurface.GetPointData()

    if filter_array_name not in tools.GetPointArrays(surface):
        pointData.append(
            dsa.VTKArray(np.zeros(nPoints, dtype=int)),
            filter_array_name
        )

    inPatchArray = pointData.GetArray(filter_array_name)

    # Compute cylinder params of centerline patch
    tangent, center, radius = cl.ComputeClPatchEndPointParameters(
                                    patch_centerline,
                                    patch_id
                                )

    # Extract patch
    patch = extract_single_line(patch_centerline, patch_id)

    tubeFunction = vtkvmtk.vtkvmtkPolyBallLine()
    tubeFunction.SetInput(patch)
    tubeFunction.SetPolyBallRadiusArrayName(names.VascularRadiusArrayName)

    lastSphere = vtk.vtkSphere()
    lastSphere.SetRadius(radius * 1.5)
    lastSphere.SetCenter(center)

    # Compute array on the surface to select point in the tube
    points = npSurface.GetPoints()
    voronoiVectors = points - center

    voronoiVectorDots = dsa.VTKArray(
                            [vtk.vtkMath.Dot(voronoiVector, tangent)
                             for voronoiVector in voronoiVectors]
                        )

    tubeValues = dsa.VTKArray(
                    [tubeFunction.EvaluateFunction(point)
                     for point in points]
                )

    sphereValues = dsa.VTKArray(
                        [lastSphere.EvaluateFunction(point)
                         for point in points]
                    )

    # Set conditions
    inTube = tubeValues <= 0.0
    inSphere = sphereValues < 0.0
    notOnPatch = voronoiVectorDots < 0.0

    # Modify filter array
    inPatchArray[inTube] = 1
    inPatchArray[inSphere & notOnPatch] = 0

    return npSurface.VTKObject


class HealthyVesselReconstructionStrategy(ABC):
    """Abstract Base Class for reconstructing a healthy vessel from a vascular
    model with an aneurysm, following the Ford et al. (2009) procedure.

    This class defines the common interface for aneurysm reconstruction
    strategies (e.g., for bifurcation or lateral aneurysms).

    Based on the procedure proposed by

        Ford et al. An objective approach to digital removal of saccular
        aneurysms: technique and applications. The British Journal of
        Radiology. 2009;82:S55–61

    and implemented in VMTK by Ms. Piccinelli, this function extracts the
    'hypothetical healthy' vessel of a vascular model with an intracranial
    aneurysm.

    .. warning::
        It is important to "reduce" the vascular surface to only the region
        where the aneurysm is, ie clip the surface so only the parent vessel
        and the daughter branches are left.

    .. warning::
        Try to select, or pass, a dome point that lies on the farthest location
        form the neck and that is centered to the neck.

    Arguments
    vascular_surface (vtkPolyData) -- the vascular surface model clipped at the
        inlet and two outlets (if a bifurcation aneurysm) or one outlet (if a
        lateral aneurysm)

    Optional
    dome_point (tuple) -- a point on the aneurysm dome surface. It is used
        to identify the aneurysm, and must be located at the tip of the
        aneurysm dome, preferentially. If None is passed, the user is prompted
        to select one interactively (default None).

    Returns
        healthy_surface (vtkPolyData) -- the surface model without the aneurysm.
    """

    def __init__(
            self,
            vascular_surface: names.polyDataType,
            inlet_ref_systems: dict=None,
            outlet_ref_systems: dict=None,
            dome_point: tuple=None
        ):

        self.vascular_surface = vascular_surface
        self._inlet_ref_systems = inlet_ref_systems
        self._outlet_ref_systems = outlet_ref_systems

        self._dome_point = tools.SelectSurfacePoint(
                               self.vascular_surface,
                               input_text="Select point on the aneurysm surface\n"
                           ) if dome_point is None else dome_point

    @abstractmethod
    def _get_clipping_points(
            self,
            inlet_points: list,
            outlet_points: list
        )   -> dict:
        """Abstract method to get the specific clipping points based on
        aneurysm type. Must be implemented by derived classes.
        """
        pass

    @abstractmethod
    def _order_clipping_points(
            self,
            dict_clip_points: dict
        )   -> list:
        """Abstract method to order the clipping points from the dictionary
        into a list, specific to each aneurysm type.
        """
        pass

    def Reconstruct(
            self,
        )   -> names.polyDataType:
        """Performs the healthy vessel reconstruction based on the selected
        strategy.

        Arguments:
        vascular_surface (vtkPolyData) -- the vascular surface model clipped at
            the inlet and two outlets (if a bifurcation aneurysm) or one outlet
            (if a lateral aneurysm)

        dome_point (tuple) -- a point on the aneurysm dome surface. It is used
            to identify the aneurysm, and must be located at the tip of the
            aneurysm dome, preferentially. If None is passed, the user is
            prompted to select one interactively (default None).

        inlet_ref_systems (dict) -- Pre-computed inlet reference systems
            (center:normal)

        outlet_ref_systems (dict) -- Pre-computed outlet reference systems
            (center:normal)

        Returns:
            healthy_surface (vtkPolyData) -- the surface model without the
            aneurysm.
        """
        # Get inlets and outlets ref. systems
        if self._inlet_ref_systems is None and self._outlet_ref_systems is None:

            inlet_ref_systems, outlet_ref_systems = VascularSurface.ComputeOpenCenters(
                self.vascular_surface
            )

            self._inlet_ref_systems = inlet_ref_systems
            self._outlet_ref_systems = outlet_ref_systems

        inletCenter = list(self._inlet_ref_systems.keys())
        outletCenters = list(self._outlet_ref_systems.keys())

        voronoi = cl.ComputeVoronoiDiagram(self.vascular_surface)

        centerlines = VascularCenterline.GenerateCenterlines(
                          self.vascular_surface,
                          source_points=inletCenter,
                          target_points=outletCenters
                      )

        # Smooth the Voronoi diagram
        smoothedVoronoi = mplib.voronoi_operations.smooth_voronoi_diagram(
                              voronoi,
                              centerlines,
                              # Smoothing factor, recommended by Piccinelli
                              0.25
                          )

        # 1) Compute parent centerline reconstruction
        # Delegate clipping points calculation to the specific strategy
        dictClipPoints = self._get_clipping_points(
                             inlet_points=inletCenter,
                             outlet_points=outletCenters
                         )

        # Order clipping points based on aneurysm type (implemented in derived classes)
        orderedClipPoints = self._order_clipping_points(dictClipPoints)

        # Store as VTK points
        clippingPoints = vtk.vtkPoints()
        for point in orderedClipPoints:
            clippingPoints.InsertNextPoint(point)

        # Extract patch centerlines
        # Siphon for lateral, bif for bifurcation
        isSiphon = not isinstance(self, BifurcationAneurysmReconstruction)

        patchCenterlines = mplib.vessel_reconstruction_tools.create_parent_artery_patches(
                               centerlines,
                               clippingPoints,
                               siphon=isSiphon,
                               bif=isinstance(self, BifurcationAneurysmReconstruction)
                           )

        # 2) Interpolate patch centerlines using splines
        parentCenterlines = mplib.vessel_reconstruction_tools.interpolate_patch_centerlines(
                                patchCenterlines,
                                centerlines,
                                additionalPoint=None,
                                lower='bif', # ... Investigate this param
                                version=True
                            )

        # 3) Clip Voronoi Diagram along centerline patches
        filterArrayName = "InPatchArray"

        for cl_id in range(patchCenterlines.GetNumberOfCells()):

            # Mark points on the Voronoi that are only on the patched
            # centerlines
            smoothedVoronoi = _set_portion_in_cl_patch(
                                  smoothedVoronoi,
                                  patchCenterlines,
                                  cl_id,
                                  filterArrayName
                              )

        # Apply filter and get portion with values == 1
        clippedVoronoi = tools.ExtractPortion(
                             smoothedVoronoi,
                             filterArrayName,
                             int(const.one)
                         )

        # As required by Morphman, also pass the clipping points as a Numpy
        # array
        clipPointsArray = np.array(
                            [clippingPoints.GetPoint(i)
                             for i in range(clippingPoints.GetNumberOfPoints())]
                        )

        # 4) Interpolate Voronoi diagram along interpolated centerline
        newVoronoi = mplib.vessel_reconstruction_tools.interpolate_voronoi_diagram(
                         parentCenterlines,
                         patchCenterlines,
                         clippedVoronoi,
                         [clippingPoints, clipPointsArray],
                         # This param 'bif' might also need dynamic handling
                         # based on strategy
                         bif=[],
                         cylinder_factor=1.0
                     )

        # 5) Compute parent surface from new Voronoi
        healthyVessel = cl.ComputeVoronoiEnvelope(newVoronoi)

        # Clip the parent vascular surface
        self._inlet_ref_systems.update(self._outlet_ref_systems)

        for center, normal in self._inlet_ref_systems.items():
            # Invert normal and displace center by one profile diameter
            center_np = np.array(center)
            normal_np = np.array(normal)
            displaced_center = tuple(center_np - normal_np)

            healthyVessel = tools.SeamPlaneTubularStructureMarker(
                                healthyVessel,
                                displaced_center,
                                normal,
                                seam_scalar_array_name=names.SeamScalarsArrayName
                            )

            healthyVessel = tools.ClipWithScalar(
                                healthyVessel,
                                names.SeamScalarsArrayName,
                                const.zero
                            )

        healthyVessel.GetPointData().RemoveArray(names.SeamScalarsArrayName)

        return healthyVessel


class BifurcationAneurysmReconstruction(HealthyVesselReconstructionStrategy):
    """ Concrete strategy for reconstructing a healthy vessel for a bifurcation
    aneurysm."""

    def _get_clipping_points(
            self,
            inlet_points: list,
            outlet_points: list
        ) -> dict:
        """Extract vessel portion where a bifurcation aneurysm grew."""
        # Note: The original _bifurcation_aneurysm_clipping_points had
        # logic for noEndPoints, which is now handled by the parent
        # reconstruct method or by the caller ensuring points are passed.
        # We assume inlet_points and outlet_points are already determined.

        inlets = inlet_points
        outlets = outlet_points

        # Tolerance distance to identify the bifurcation
        divTolerance = 0.01

        # One inlet and two outlets, bifurcation with the aneurysm
        # 1 -> centerline of the branches only
        # 2 -> centerline of the first outlet to the aneurysm and inlet
        # 3 -> centerline of the second outlet to the aneurysm and inlet
        # Using first two outlets for bifurcation
        # TODO: this assumes only two outlets. How to handle more if larger
        # tree?
        relevantOutlets = outlets[0:2]

        clWithoutAneurysm = VascularCenterline.GenerateCenterlines(
                                self.vascular_surface,
                                inlets,
                                relevantOutlets
                            )

        clippingPoints = {}
        clippingPoints["bif"] = cl.GetDivergingPoint(
                                    clWithoutAneurysm,
                                    divTolerance
                                )

        for cl_id, outlet in enumerate(relevantOutlets):
            daughterCenterline = VascularCenterline.GenerateCenterlines(
                                    self.vascular_surface,
                                    [outlet],
                                    inlets + [self._dome_point]
                                )

            # Get clipping point on this branch
            clippingPoints["dau" + str(cl_id)] = cl.GetDivergingPoint(
                                                    daughterCenterline,
                                                    divTolerance
                                                )
        return clippingPoints

    def _order_clipping_points(
            self,
            dict_clip_points: dict
        )   -> list:

        return [
            dict_clip_points.get("bif", None),
            dict_clip_points.get("dau0", None),
            dict_clip_points.get("dau1", None)
        ]


class LateralAneurysmReconstruction(HealthyVesselReconstructionStrategy):
    """Concrete strategy for reconstructing a healthy vessel for a lateral
    aneurysm."""

    def _get_clipping_points(
            self,
            inlet_points: list,
            outlet_points: list
        ) -> dict:
        """Extract vessel portion where a lateral aneurysm grew."""

        inlets = inlet_points
        outlets = outlet_points

        # Tolerance distance to identify the bifurcation
        divTolerance = 0.01

        # One inlet and one outlet (although the model can have more than one outlet),
        # lateral aneurysm
        # Using only the first outlet for lateral
        relevantOutlets = outlets[0:1]

        forwardCenterline = VascularCenterline.GenerateCenterlines(
                                self.vascular_surface,
                                inlets,
                                relevantOutlets + [self._dome_point]
                            )

        backwardCenterline = VascularCenterline.GenerateCenterlines(
                                self.vascular_surface,
                                relevantOutlets,
                                inlets + [self._dome_point]
                            )

        upstreamClipPoint = cl.GetDivergingPoint(
                                forwardCenterline,
                                divTolerance
                            )

        downstreamClipPoint = cl.GetDivergingPoint(
                                backwardCenterline,
                                divTolerance
                            )

        return {
            "upstream": upstreamClipPoint,
            "downstream": downstreamClipPoint
        }

    def _order_clipping_points(
            self,
            dict_clip_points: dict
        )   -> list:

        return [
            dict_clip_points.get("upstream", None),
            dict_clip_points.get("downstream", None)
        ]


def HealthyVesselReconstruction(
        vascular_surface: names.polyDataType,
        aneurysm_type: str,
        dome_point: tuple = None,
        inlet_ref_systems: dict = None,
        outlet_ref_systems: dict = None
    ) -> names.polyDataType:
    """
    Given vasculature model with aneurysm, extract vessel without aneurysm.

    This is the client interface that uses the Strategy Pattern.
    """
    if aneurysm_type == "bifurcation":
        strategy = BifurcationAneurysmReconstruction(vascular_surface)

    elif aneurysm_type == "lateral":
        strategy = LateralAneurysmReconstruction(vascular_surface)

    else:
        raise ValueError("Aneurysm type must be 'bifurcation' or 'lateral'.")

    return strategy.Reconstruct(
                dome_point=dome_point,
                inlet_ref_systems=inlet_ref_systems,
                outlet_ref_systems=outlet_ref_systems
            )

