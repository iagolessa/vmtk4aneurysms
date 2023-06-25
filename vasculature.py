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

"""Collection of tools to represent vascular models."""

import vtk
from vmtk import vtkvmtk
from vmtk import vmtkscripts

from numpy import delete, where
from vtk.numpy_interface import dataset_adapter as dsa
from vmtk4aneurysms.aneurysms import Aneurysm
from vmtk4aneurysms.vascular_operations import ExtractAneurysmSacSurface

from vmtk4aneurysms.lib import names
from vmtk4aneurysms.lib import centerlines as cnt
from vmtk4aneurysms.lib import constants as const
from vmtk4aneurysms.lib import polydatatools as tools
from vmtk4aneurysms.lib import polydatageometry as geo

from .pypescripts import v4aScripts

class Branch():
    """Branch segment representation."""

    def __init__(self, polydata):
        """Initialize from branch vtkPolyData."""

        self._branch = polydata

    def GetBranch(self):
        """Return branch vtkPolyData."""
        return self._branch

    def GetLength(self):
        """Compute length of branch."""
        # Get arrays
        pointArrays = tools.GetPointArrays(self._branch)
        abscissasArray = "Abscissas"

        if abscissasArray not in pointArrays:
            attributes = vmtkscripts.vmtkCenterlineAttributes()
            attributes.Centerlines = self._branch
            attributes.Execute()

            self._branch = attributes.Centerlines

        # Compute length by Abscissas array
        distanceRange = self._branch.GetPointData().GetArray(
            abscissasArray
        ).GetRange()

        length = max(distanceRange) - min(distanceRange)

        return length

class Bifurcation:
    """Model of a bifurcation of a vascular network.

    Based on the work by Piccinelli et al. (2009), who proposed a framework to
    identify and quantitatively analyze bifurcations in vascular models, this
    class implements some of their geometric definitions.  Its inputs are the
    bifurcation reference system (vtkPolyData) containing its center and
    normal, and the bifurcation vectors. Both can be computed with
    'vmtkbifurcationreferencesystem' and 'vmtkbifurcationvectors' scripts of
    the VMTK library.
    """

    def __init__(
            self,
            referenceSystem,
            vectors
        ):

        self._bif_reference_system = referenceSystem
        self._bif_vectors_object = vectors

        # Characterize bifurcation plane
        self._bif_center = referenceSystem.GetPoints().GetPoint(0)

        self._bif_plane_normal = referenceSystem.GetPointData().GetArray(
                                     'Normal'
                                 ).GetTuple(0)


        self._bif_plane_normal = referenceSystem.GetPointData().GetArray(
                                     'UpNormal'
                                 ).GetTuple(0)

        # Characterize branches directions in bifurcation
        npBifVectors = dsa.WrapDataObject(vectors)
        bifPointData = npBifVectors.GetPointData()

        self._nbranches = vectors.GetPoints().GetNumberOfPoints()
        self._branches_ids = bifPointData.GetArray(
                                 "GroupIds"
                             )

        # Collect points and vectors (in plane and out plane)
        self._bif_points  = npBifVectors.GetPoints()

        self._bif_vectors = bifPointData.GetArray(
                                "BifurcationVectors"
                            )

        # In plane vector are the bif vectors projected in the bif plane
        self._inplane_bif_vectors = bifPointData.GetArray(
                                        "InPlaneBifurcationVectors"
                                    )
        self._outplane_bif_vectors = bifPointData.GetArray(
                                        "OutOfPlaneBifurcationVectors"
                                    )

        # In plane vector are the bif vectors projected in the bif plane
        self._inplane_bif_angles = bifPointData.GetArray(
                                        "InPlaneBifurcationVectorAngles"
                                    )

        self._outplane_bif_angles = bifPointData.GetArray(
                                        "OutOfPlaneBifurcationVectorAngles"
                                    )

    def GetBifurcationReferenceSystem(self):
        """Get the bifurcation reference system."""

        return self._bif_reference_system

    def GetBifurcationVectorsObject(self):
        """Get the bifurcation vector object."""

        return self._bif_vectors_object

    def GetBifurcationPlaneAngles(self):
        """Get the bifurcation vectors."""

        return self._inplane_bif_angles

    def GetBifurcationVectors(self):
        """Get the bifurcation vectors."""

        return self._bif_vectors

    def GetBifurcationPlane(self):
        """Get the bifurcation plane."""

        bifPlane = vtk.vtkPlane()
        bifPlane.SetOrigin(self._bif_center)
        bifPlane.SetNormal(self._bif_plane_normal)

        return bifPlane

    # def GetDaugtherBranchesAngle(self):
    #     """Return the angle between daughter branches (for two-branched
    #     bifurcations)."""

    #     # Get the two largest values of the GroupIds array, that identifies the
    #     # branches departing from a bifurcation
    #     if self._nbranches == 3:

    #         daughterIds = np.delete(
    #                             self._branches_ids,
    #                             where(
    #                                 self._branches_ids == min(self._branches_ids)
    #                             )
    #                         )



        # else:
        #     raise NotImplementedError(
        #               "Angle between branches not implemented for bifurcations "\
        #               "with more than two daugther branches."
        #           )

class Vasculature:
    """Representation of a vascular network tree model.

    This class presents an interface to work with vascular models represented
    as surfaces (vtkPolyData). The surface model must contain the open
    boundaries of the domain, i.e.  its inlets and outlets of blood flow as
    perpendicular open sections, as traditionally used for CFD. So far, it
    handles only vasculatures with a single inlet, defined as the one with
    largest radius.

    At construction, it automatically computes the centerline and the
    morphology of the vasculature. Internally, it uses VMTK to compute all the
    geometrical features of the centerline path to fully characterize the
    vasculature topology. Furthermore, the surface curvature characterization
    is added as arrays to the surface: mean and Gaussian curvatures with an
    array defining the local curvature type.

    The vasculature may contain an aneurysm: this must be explicitly informed
    by the user through the switch 'with_aneurysm'. If true, the user can also
    determine whether the aneurysm surface will be detected automatically
    (experimental yet) and a plane neck will be generated, or manually draw by
    the user, in which case a window is open allowing the user to select the
    aneurysm neck.
    """

    def __init__(
            self,
            vtk_poly_data: names.polyDataType,
            with_aneurysm: bool=False,
            clip_aneurysm_mode: str="interactive",
            parent_vascular_surface: names.polyDataType=None,
            aneurysm_prop: dict={}
        ):
        """Initiate vascular model.

        Given a vascular surface (vtkPolyData), automatically compute its
        centerlines and bifurcations geometry. If the vasculature has an
        aneurysm, the flag 'with_aneurysm' enables its selection.

        Arguments:
        vtk_poly_data -- the vtkPolyData vascular model (default None)

        with_aneurysm -- bool to indicate that the vasculature
        has an aneurysm (default False)

        clip_aneurysm_mode (str, default: 'interactive') -- the method to clip
        the aneurysm, if present. Use the function
        'vascular_operations.ExtractAneurysmSacSurface', hence the options are:
        'interactive', 'automatic', or 'plane'.  Only enabled if the
        'with_aneurysm' arguments is True.  (default False).

        aneurysm_prop -- optional dictionary with properties of the aneurysms:
        type, status, label.
        """

        self._surface_model   = None
        self._centerlines     = None
        self._inlet_centers   = None
        self._outlet_centers  = None
        self._aneurysm_point  = None
        self._aneurysm_model  = None
        self._with_aneurysm   = with_aneurysm
        self._clip_aneurysm_mode = clip_aneurysm_mode

        # If the vasculature has an aneurysm, allow to also input the parent
        # vasculature
        self._parent_vascular_surface = parent_vascular_surface

        self._nbifurcations  = int(const.zero)
        self._bifurcations   = []
        self._branches       = []

        self._inlet_centers, self._outlet_centers = cnt.ComputeOpenCenters(
                                                        vtk_poly_data
                                                    )

        # Morphology first to avoid some weird bug when using the array Normals
        # inside the computation of the open centers
        self._centerlines = cnt.GenerateCenterlines(
                                vtk_poly_data
                            )

        self._centerlines = cnt.ComputeCenterlineGeometry(
                                self._centerlines
                            )

        # Initiate surface model
        self._surface_model  = geo.Surface(vtk_poly_data)

        # Compute branches and the array for branch splitting
        self._extract_branches()

        # Collecting bifurcations and branches.'
        self._compute_bifurcations_geometry()
        self._split_branches()

        if self._with_aneurysm:

            aneurysmType = aneurysm_prop["aneurysm_type"]

            # This function already computes the parent vessels
            # but then the type is required here
            aneurysm_surface = ExtractAneurysmSacSurface(
                                   self._surface_model.GetSurfaceObject(),
                                   mode=self._clip_aneurysm_mode,
                                   parent_vascular_surface=self._parent_vascular_surface,
                                   aneurysm_type=aneurysmType
                               )

            self._aneurysm_model = Aneurysm(
                                        aneurysm_surface,
                                        **aneurysm_prop
                                    )

    @classmethod
    def from_file(
            cls,
            file_name,
            with_aneurysm=False,
            clip_aneurysm_mode="interactive",
            parent_vascular_surface=None,
            aneurysm_prop={}
        ):
        """Initialize vasculature object from vasculature surface file."""

        return cls(tools.ReadSurface(file_name),
                   with_aneurysm=with_aneurysm,
                   clip_aneurysm_mode=clip_aneurysm_mode,
                   parent_vascular_surface=parent_vascular_surface,
                   aneurysm_prop=aneurysm_prop)

    def _extract_branches(self):
        """Split the vasculature centerlines into branches."""
        # Split centerline into branches
        branches = vmtkscripts.vmtkBranchExtractor()
        branches.Centerlines = self._centerlines
        branches.Execute()

        # The centerlines object with the arrays defining the branches portions
        self._branched_centerlines = branches.Centerlines

        # Array Names
        self._radius_array_name   = branches.RadiusArrayName

        # Defines the branches and the bifurcations
        self._blanking_array_name = branches.BlankingArrayName

        # Defines *each* branch and bifurcations
        self._group_ids_array_name = branches.GroupIdsArrayName

        # Roughly defines the before and after the bifurcations
        self._tract_ids_array_name = branches.TractIdsArrayName

        # Centerline ids array
        self._centerlines_ids_array_name = branches.CenterlineIdsArrayName

    def _compute_bifurcations_geometry(self):
        """Collect bifurcations and computes their geometry.

        Identifies the bifurcations of the input surface model and gather their
        information in a list of bifurcations.
        """

        # Computing the bifurcation reference system
        bifsRefSystem = vmtkscripts.vmtkBifurcationReferenceSystems()

        bifsRefSystem.Centerlines       = self._branched_centerlines
        bifsRefSystem.RadiusArrayName   = self._radius_array_name
        bifsRefSystem.BlankingArrayName = self._blanking_array_name
        bifsRefSystem.GroupIdsArrayName = self._group_ids_array_name
        bifsRefSystem.Execute()

        # Get bifuraction list
        referenceSystems    = bifsRefSystem.ReferenceSystems
        self._nbifurcations = referenceSystems.GetPoints().GetNumberOfPoints()

        bifsIdsArray = referenceSystems.GetPointData().GetArray(
                           self._group_ids_array_name
                       )

        bifurcationsIds = [bifsIdsArray.GetValue(index)
                           for index in range(self._nbifurcations)]

        if self._nbifurcations > const.zero:
            # Compute bifurcation
            bifVectors = vmtkscripts.vmtkBifurcationVectors()

            bifVectors.ReferenceSystems  = bifsRefSystem.ReferenceSystems
            bifVectors.Centerlines       = self._branched_centerlines
            bifVectors.RadiusArrayName   = self._radius_array_name
            bifVectors.GroupIdsArrayName = self._group_ids_array_name
            bifVectors.TractIdsArrayName = self._tract_ids_array_name
            bifVectors.BlankingArrayName = self._blanking_array_name
            bifVectors.CenterlineIdsArrayName = self._centerlines_ids_array_name

            bifVectors.ReferenceSystemsNormalArrayName   = \
                    bifsRefSystem.ReferenceSystemsNormalArrayName

            bifVectors.ReferenceSystemsUpNormalArrayName = \
                    bifsRefSystem.ReferenceSystemsUpNormalArrayName

            bifVectors.NormalizeBifurcationVectors = True
            bifVectors.Execute()

            for index in bifurcationsIds:
                # Filter bifurcation reference system to get only one
                # bifurcation
                bifsRefSystem.ReferenceSystems.GetPointData().SetActiveScalars(
                    self._group_ids_array_name
                )

                bifurcationSystem = vtk.vtkThresholdPoints()
                bifurcationSystem.SetInputData(bifsRefSystem.ReferenceSystems)
                bifurcationSystem.ThresholdBetween(index, index)
                bifurcationSystem.Update()

                bifVectors.BifurcationVectors.GetPointData().SetActiveScalars(
                    bifVectors.BifurcationGroupIdsArrayName
                )

                bifurcationVectors = vtk.vtkThresholdPoints()
                bifurcationVectors.SetInputData(bifVectors.BifurcationVectors)
                bifurcationVectors.ThresholdBetween(index, index)
                bifurcationVectors.Update()

                system  = bifurcationSystem.GetOutput()
                vectors = bifurcationVectors.GetOutput()

                self._bifurcations.append(
                    Bifurcation(system, vectors)
                )

    def _split_branches(self):
        """Split vasculature into branches.

        Given the vasculature centerlines, slits it into its constituent
        branches. Generates a list of branch objects.
        """
        # Extract only the branches portion
        # (the blanking array separates the branches from the bifurcations)
        branchesId = 0
        branches = tools.ExtractPortion(
                       self._branched_centerlines,
                       self._blanking_array_name,
                       branchesId
                   )

        # Get only the branch group ids
        npBranches = dsa.WrapDataObject(branches)
        branchesIds = set(
                          npBranches.GetCellData().GetArray(
                              self._group_ids_array_name
                          )
                      )

        for branchId in branchesIds:
            try:
                branch = tools.ExtractPortion(
                             branches,
                             self._group_ids_array_name,
                             branchId
                         )

                self._branches.append(
                    Branch(branch)
                )

            except(ValueError):
                pass

    # TODO: maybe evaluate suitability to turn this method into a classmethod
    def ComputeWallThicknessArray(self):
        """Add thickness array to the vascular surface."""

        vasculatureThickness = v4aScripts.vmtkSurfaceVasculatureThickness()
        vasculatureThickness.Surface = self._surface_model.GetSurfaceObject()
        vasculatureThickness.Centerlines = self._centerlines
        vasculatureThickness.Aneurysm = self._with_aneurysm
        vasculatureThickness.SelectAneurysmRegions = False

        vasculatureThickness.SmoothingIterations = 20
        vasculatureThickness.GenerateWallMesh = False
        vasculatureThickness.Execute()

        # Recomputes surface model
        self._surface_model = geo.Surface(vasculatureThickness.Surface)

    def GetSurface(self):
        """Return the vascular surface."""
        return self._surface_model

    def GetAneurysm(self):
        """Return the aneurysm model, if any."""
        return self._aneurysm_model

    def GetCenterlines(self):
        """Return the vasculature's centerlines."""
        return self._centerlines

    def GetInletCenters(self):
        """Return the inlet center."""
        return self._inlet_centers

    def GetOutletCenters(self):
        """Return the outlet center(s)."""
        return self._outlet_centers

    def GetBifurcations(self):
        """Return the vascular model's bifurcations."""
        return self._bifurcations

    def GetNumberOfBifurcations(self):
        """Return the number of bifurcations."""
        return self._nbifurcations

    def GetBranches(self):
        """Return the vascular model's branches."""
        return self._branches
