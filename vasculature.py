"""Provide Vasculature class.

This module provides the Vasculature class that models a portion of the
vascular system. The basic input is a surface model of the vasculature as a
vtkPolyData or a file name. When instantiated, the bifurcations and branches
are automatically identified after the calculation of the vasculature's
centerlines by using the VMTK\R library. Both centerline and surface
geometrical characterization is performed and stored as arrays in the model.
"""

import vtk
from vmtk import vtkvmtk
from vmtk import vmtkscripts

from . import aneurysm
from .aneurysm_neck import AneurysmNeckPlane 

from . import centerlines as cnt
from . import constants as const
from . import polydataobjects as objs
from . import polydatatools as tools
from . import polydatageometry as geo

from .vmtkextend import customscripts

class Branch():
    """Branch segment representation."""

    def __init__(self, polydata):
        """Initialize from branch vtkPolyData."""

        self._branch = polydata

    @classmethod
    def from_centerline(cls, centerline, start_point, end_point):
        """Initialize branch object from centerline and end points."""
        pass

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

    Based on the work of Piccinelli et al. (2009), who 
    proposed a framework to identify and quantitatively
    analyze bifurcations in vascular models, this class
    implements some of their geometric definitions. 
    It inputs are the bifurcation reference system
    vtkPolyData containg its center and normal, and 
    the bifurcation vectors. Both can be computed with
    'vmtkbifurcationreferencesystem' and 'vmtkbifurca-
    tionvectors' scripts of the VMTK library.
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
    """Class of vascular model represented by a surface.

    This class presents an interface to work with vascular models represented
    as surfaces (vtkPolyData). The surface model must contain the open
    boundaries of the domain, i.e.  its inlets and outlets of blood flow as
    perpendicular open sections as traditionally used for CFD. So far, it
    handles only vasculatures with a single inlet, defined as the one with
    largest radius. At construction, the class automatically computes the
    centerline and the morphology of the vasculature. Internally, it uses VMTK
    to compute all the geometrical features of the centerline path to fully
    characterize the vasculature topology. Furthermore, the surface curvature 
    characterization is added as arrays to the surface: mean and Gaussian 
    curvatures with an array containing the local curvature type.

    The vasculature may contain an aneurysm: this must be explicitly informed 
    by the user through the switch 'with_aneurysm'. If true, the user can also 
    determine if the aneurysm surface will detected automatically and a plane 
    neck will be generated, or manually draw by the user, in which case a 
    window is open allowing the user to select the aneurysm neck. 
    """

    def __init__(self,
                 vtk_poly_data,
                 with_aneurysm=False,
                 manual_aneurysm=False,
                 aneurysm_prop={}):
        """Initiate vascular model.

        Given vascular surface (vtkPolyData), automatically compute its
        centerlines and bifurcations geometry. If the vasculature has an
        aneurysm, the flag 'with_aneurysm' enables its selection.

        Arguments:
        surface -- the vtkPolyData vascular model (default None)

        with_aneurysm -- bool to indicate that the vasculature
            has an aneurysm (default False)

        manual_aneurysm -- bool that enable the manual selection of the
            aneurysm neck, otherwise, try to automatically extract the aneurysm
            neck *plane*, based on the user input of the aneurysm tip point and
            the algorithm proposed in 

                Piccinelli et al. (2012).  
                Automatic neck plane detection and 3d geometric
                characterization of aneurysmal sacs. 
                Annals of Biomedical Engineering, 40(10), 2188–2211. 
                DOI :10.1007/s10439-012-0577-5.

            Only enabled if the 'with_aneurysm' arguments is True.  (default
            False).

        aneurysm_prop -- dictionary with properties required by Aneurysm
            class: type, status, label.
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
        self._surface_model  = objs.Surface(vtk_poly_data)

        print('Collecting bifurcations and branches.', end='\n')
        self._compute_bifurcations_geometry()
        self._split_branches()

        if self._with_aneurysm:
            print("Extracting aneurysm surface.")

            if self._manual_aneurysm:
                extractAneurysm = customscripts.vmtkExtractAneurysm()
                extractAneurysm.Surface = self._surface_model.GetSurfaceObject()
                extractAneurysm.Execute()

                aneurysm_surface = extractAneurysm.AneurysmSurface

            else:
                # Extract aneurysm surface with plane neck
                aneurysm_surface = AneurysmNeckPlane(
                                        self._surface_model.GetSurfaceObject()
                                    )

            self._aneurysm_model = aneurysm.Aneurysm(
                                        aneurysm_surface,
                                        **aneurysm_prop
                                    )

    @classmethod
    def from_surface_file(cls, surface_file):
        """Initialize vasculature object from vasculature surface file."""
        pass

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

        vasculatureThickness = customscripts.vmtkSurfaceVasculatureThickness()
        vasculatureThickness.Surface = self._surface_model.GetSurfaceObject()
        vasculatureThickness.Centerlines = self._centerlines
        vasculatureThickness.Aneurysm = self._with_aneurysm
        vasculatureThickness.SelectAneurysmRegions = False

        vasculatureThickness.SmoothingIterations = 20
        vasculatureThickness.GenerateWallMesh = False
        vasculatureThickness.Execute()

        # Recomputes surface model
        self._surface_model = objs.Surface(vasculatureThickness.Surface)

    def GetSurface(self):
        return self._surface_model

    def GetAneurysm(self):
        return self._aneurysm_model

    def GetCenterlines(self):
        return self._centerlines

    def GetInletCenters(self):
        return self._inlet_centers

    def GetOutletCenters(self):
        return self._outlet_centers

    def GetBifurcations(self):
        return self._bifurcations

    def GetNumberOfBifurcations(self):
        return self._nbifurcations

    def GetBranches(self):
        return self._branches
