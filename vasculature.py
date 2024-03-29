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

from . import aneurysms as aneu
from .vascular_operations import AneurysmNeckPlane

from .lib import names
from .lib import centerlines as cnt
from .lib import constants as const
from .lib import polydatatools as tools
from .lib import polydatageometry as geo

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

    def __init__(self, referenceSystem, vectors):
        try:
            self.center = referenceSystem.GetPoints().GetPoint(0)
            self.normal = referenceSystem.GetPointData().GetArray(
                'Normal'
            ).GetTuple(0)

            self.upnormal = referenceSystem.GetPointData().GetArray(
                'UpNormal'
            ).GetTuple(0)

            self.numberOfBranches = vectors.GetPoints().GetNumberOfPoints()

            # Collect point and vectors
            points = vectors.GetPoints()
            pointData = vectors.GetPointData()

            self.points = [
                points.GetPoint(i)
                for i in range(self.numberOfBranches)
            ]

            self.vectors = [
                pointData.GetArray('BifurcationVectors').GetTuple(index)
                for index in range(self.numberOfBranches)
            ]

            self.inPlaneVectors = [
                pointData.GetArray('InPlaneBifurcationVectors').GetTuple(i)
                for i in range(self.numberOfBranches)
            ]

        except AttributeError:
            # TODO: improve this init
            errorMessage = "Error building bifurcation!" + \
                "It seems you did not pass me a vtkPolyData."

            print(errorMessage)


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
            vtk_poly_data,
            with_aneurysm=False,
            manual_aneurysm=False,
            aneurysm_prop={}
        ):
        """Initiate vascular model.

        Given a vascular surface (vtkPolyData), automatically compute its
        centerlines and bifurcations geometry. If the vasculature has an
        aneurysm, the flag 'with_aneurysm' enables its selection.

        Arguments:
        vtk_poly_data -- the vtkPolyData vascular model (default None)

        with_aneurysm -- bool to indicate that the vasculature
        has an aneurysm (default False)

        manual_aneurysm -- bool that enable the manual selection of the
        aneurysm neck, otherwise, try to automatically extract the aneurysm
        neck *plane*, based on the user input of the aneurysm tip point and the
        algorithm proposed in

        Piccinelli et al. (2012).  Automatic neck plane detection and 3d
        geometric characterization of aneurysmal sacs.  Annals of Biomedical
        Engineering, 40(10), 2188–2211.  DOI :10.1007/s10439-012-0577-5.

        Only enabled if the 'with_aneurysm' arguments is True.  (default
        False).

        aneurysm_prop -- dictionary with properties required by Aneurysm class:
        type, status, label.
        """

        print('Initiating model.', end='\n')
        self._surface_model   = None
        self._centerlines     = None
        self._inlet_centers   = None
        self._outlet_centers  = None
        self._aneurysm_point  = None
        self._aneurysm_model  = None
        self._with_aneurysm   = with_aneurysm
        self._manual_aneurysm = manual_aneurysm

        self._nbifurcations  = int(const.zero)
        self._bifurcations   = []
        self._branches       = []

        self._inlet_centers, self._outlet_centers = cnt.ComputeOpenCenters(
                                                        vtk_poly_data
                                                    )

        # Morphology first to avoid some weird bug when using the array Normals
        # inside the computation of the open centers
        print('Computing centerlines.', end='\n')

        self._centerlines = cnt.GenerateCenterlines(
                                vtk_poly_data
                            )

        self._centerlines = cnt.ComputeCenterlineGeometry(
                                self._centerlines
                            )

        # Initiate surface model
        self._surface_model  = geo.Surface(vtk_poly_data)

        print('Collecting bifurcations and branches.', end='\n')
        self._compute_bifurcations_geometry()
        self._split_branches()

        if self._with_aneurysm:
            print("Extracting aneurysm surface.")

            if self._manual_aneurysm:
                extractAneurysm = v4aScripts.vmtkExtractAneurysm()
                extractAneurysm.Surface = self._surface_model.GetSurfaceObject()
                extractAneurysm.Execute()

                aneurysm_surface = extractAneurysm.AneurysmSurface

            else:
                # Extract aneurysm surface with plane neck
                aneurysm_surface = AneurysmNeckPlane(
                                        self._surface_model.GetSurfaceObject()
                                    )

            self._aneurysm_model = aneu.Aneurysm(
                                        aneurysm_surface,
                                        **aneurysm_prop
                                    )

    @classmethod
    def from_file(
            cls,
            file_name,
            with_aneurysm=False,
            manual_aneurysm=False,
            aneurysm_prop={}
        ):
        """Initialize vasculature object from vasculature surface file."""

        return cls(tools.ReadSurface(file_name),
                   with_aneurysm=with_aneurysm,
                   manual_aneurysm=manual_aneurysm,
                   aneurysm_prop=aneurysm_prop)

    def _extract_branches(self):
        """Split the vasculature centerlines into branches."""
        pass

    def _compute_bifurcations_geometry(self):
        """Collect bifurcations and computes their geometry.

        Identifies the bifurcations of the input surface model and gather their
        information in a list of bifurcations.
        """

        # Split centerline into branches
        branches = vmtkscripts.vmtkBranchExtractor()
        branches.Centerlines = self._centerlines
        branches.Execute()

        # Array Names
        radiusArrayName   = branches.RadiusArrayName
        blankingArrayName = branches.BlankingArrayName
        groupIdsArrayName = branches.GroupIdsArrayName
        tractIdsArrayName = branches.TractIdsArrayName

        # Computing the bifurcation reference system
        bifsRefSystem = vmtkscripts.vmtkBifurcationReferenceSystems()

        bifsRefSystem.Centerlines       = branches.Centerlines
        bifsRefSystem.RadiusArrayName   = radiusArrayName
        bifsRefSystem.BlankingArrayName = blankingArrayName
        bifsRefSystem.GroupIdsArrayName = groupIdsArrayName
        bifsRefSystem.Execute()

        # Get bifuraction list
        referenceSystems    = bifsRefSystem.ReferenceSystems
        self._nbifurcations = referenceSystems.GetPoints().GetNumberOfPoints()

        bifsIdsArray = referenceSystems.GetPointData().GetArray(
                           groupIdsArrayName
                       )

        bifurcationsIds = [
            bifsIdsArray.GetValue(index)
            for index in range(self._nbifurcations)
        ]

        if self._nbifurcations > const.zero:
            # Compute bifurcation
            bifVectors = vmtkscripts.vmtkBifurcationVectors()

            bifVectors.ReferenceSystems  = bifsRefSystem.ReferenceSystems
            bifVectors.Centerlines       = branches.Centerlines
            bifVectors.RadiusArrayName   = radiusArrayName
            bifVectors.GroupIdsArrayName = groupIdsArrayName
            bifVectors.TractIdsArrayName = tractIdsArrayName
            bifVectors.BlankingArrayName = blankingArrayName
            bifVectors.CenterlineIdsArrayName = branches.CenterlineIdsArrayName

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
                    groupIdsArrayName
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
        branches. Return a list of branch objects.
        """
        # Split centerline into branches
        branches = vmtkscripts.vmtkBranchExtractor()
        branches.Centerlines = self._centerlines
        branches.Execute()

        # Array Names
        radiusArrayName = branches.RadiusArrayName
        blankingArrayName = branches.BlankingArrayName
        groupIdsArrayName = branches.GroupIdsArrayName
        tractIdsArrayName = branches.TractIdsArrayName

        # Extract only the branches portion
        branchesId = 0
        branches = tools.ExtractPortion(
            branches.Centerlines,
            blankingArrayName,
            branchesId
        )

        maxGroupId = max(
            branches.GetCellData().GetArray(groupIdsArrayName).GetRange()
        )

        for branchId in range(int(maxGroupId) + 1):
            branch = tools.ExtractPortion(
                branches,
                groupIdsArrayName,
                branchId
            )

            if branch.GetLength() != 0.0:
                self._branches.append(Branch(branch))


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

    @staticmethod
    def ClipVasculature(
            vascular_surface: names.polyDataType
        )   -> names.polyDataType:
        """Clip a vascular surface segment, by selecting end points.

        Given a vascular surface, the user is prompted to select points
        on the surface that 1) identifies the surface's bulk and 2) where the
        vasculature should be clipped. Uses, internally, the
        'vmtksurfaceendclipper' script.
        """

        centerlines = cnt.GenerateCenterlines(vascular_surface)
        geoCenterlines = cnt.ComputeCenterlineGeometry(centerlines)

        FrenetTangentArrayName = "FrenetTangent"

        surfaceEndClipper = vmtkscripts.vmtkSurfaceEndClipper()
        surfaceEndClipper.Surface = vascular_surface
        surfaceEndClipper.CenterlineNormals = 1
        surfaceEndClipper.Centerlines = geoCenterlines
        surfaceEndClipper.FrenetTangentArrayName = FrenetTangentArrayName
        surfaceEndClipper.Execute()

        return surfaceEndClipper.Surface

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
