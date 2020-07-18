"""Provide Vasculature class.

This module provides the Vasculature class that models
a vasculature portion of the vascular system. The basic
input is a surface model of the vasculature as a vtkPolyData
or a file name.
"""

import vtk
from vmtk import vtkvmtk
from vmtk import vmtkscripts

import aneurysm
import centerlines
from aneurysm_neck import aneurysmNeckPlane 

from constants import *
import polydatatools as tools
import polydatageometry as geo

from vmtkextend import customscripts

class Branch():
    """Branch segment representation."""

    def __init__(self, polydata):
        """Initialize from branch vtkPolyData."""

        self._branch = polydata

    @classmethod
    def from_centerline(cls, centerline, start_point, end_point):
        """Initialize branch object from centerline and end points."""

        pass

    def getBranch(self):
        """Return branch vtkPolyData."""
        return self._branch

    def getLength(self):
        """Compute length of branch."""
        # Get arrays
        pointArrays = tools.getPointArrays(self._branch)
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
    """Class of vascular models represented by surfaces.

    This class presents an interface to work with vascular 
    models represented as surfaces (vtkPolyData). The surface
    model must contain the open boundaries of the domain, i.e.
    its inlets and outlets of blood flow as perpendicular
    open sections (so far, it handles only vasculatures with 
    a single inlet, defined as the one with largest radius).
    At construction, the class automatically computes the 
    centerlines (can be changed with the switch 
    'manual_centerline') and the morphology of the vasculature:
    the centerlines is computed together with its complete
    set of morphological attributes.
    """

    def __init__(self,
                 surface,
                 with_aneurysm=False,
                 manual_aneurysm=False,
                 aneurysm_prop=dict()):
        """Initiate vascular model.

        Given vascular surface(vtkPolyData), automatically compute 
        its centerlines and bifurcations geometry. If the vasculature
        has an aneurysm, the flag 'with_aneurysm' enables its selection.

        Arguments:
        surface -- the vtkPolyData vascular model (default None)

        with_aneurysm -- bool to indicate that the vasculature
            has an aneurysm (default False)

        manual_aneurysm -- bool that enable the manual selection
            of the aneurysm neck, otherwise, try to automatically 
            extract the aneurysm neck *plane*, based on the user
            input of the aneurysm tip point and the algorithm
            proposed in 

                Piccinelli et al. (2012). 
                Automatic neck plane detection and 3d geometric 
                characterization of aneurysmal sacs. 
                Annals of Biomedical Engineering, 40(10), 2188–2211. 
                DOI :10.1007/s10439-012-0577-5.

            Only enabled if the 'with_aneurysm' arguments is True.
            (default False).

        aneurysm_prop -- dictionary with properties required by Aneurysm
            class: type, status, label.
        """

        print('Initiating model.', end='\n')

        # Computes the curvature of the vasculature
        self._surface = geo.surfaceCurvature(surface)
        self._centerlines = None
        self._aneurysm_point = None

        # Compute open boundaries centers
        self._inlet_centers = None
        self._outlet_centers = None

        # Flags
        self._with_aneurysm = with_aneurysm
        self._manual_aneurysm = manual_aneurysm
        self._aneurysm_model = None

        # Compute morphology
        self._inlet_centers, self._outlet_centers = centerlines.computeOpenCenters(self._surface)

        print('Computing centerlines.', end='\n')
        self._centerlines = centerlines.generateCenterlines(
                                self._surface
                            )

        self._centerlines = centerlines.computeCenterlineGeometry(
                                self._centerlines
                            )

        print('Collecting bifurcations.', end='\n')
        self._nbifurcations = 0
        self._bifurcations = list()
        self._compute_bifurcations_geometry()

        print('Collecting branches.', end='\n')
        self._branches = list()
        self._split_branches()

        # Delineating aneurysm
        if self._with_aneurysm:
            print("Extracting aneurysm surface.")

            if self._manual_aneurysm:
                extractAneurysm = customscripts.vmtkExtractAneurysm()
                extractAneurysm.Surface = self._surface
                extractAneurysm.Execute()

                aneurysm_surface = extractAneurysm.AneurysmSurface


            else:
                # Extract aneurysm surface with plane neck
                aneurysm_surface = aneurysmNeckPlane(self._surface)

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

        Identifies the bifurcations of the input surface model 
        and gather their information in a list of bifurcations.
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

        # Computing the bifurcation reference system
        bifsRefSystem = vmtkscripts.vmtkBifurcationReferenceSystems()
        bifsRefSystem.Centerlines = branches.Centerlines
        bifsRefSystem.RadiusArrayName = radiusArrayName
        bifsRefSystem.BlankingArrayName = blankingArrayName
        bifsRefSystem.GroupIdsArrayName = groupIdsArrayName
        bifsRefSystem.Execute()

        # Get bifuraction list
        systems = bifsRefSystem.ReferenceSystems
        self._nbifurcations = systems.GetPoints().GetNumberOfPoints()

        bifsIdsArray = systems.GetPointData().GetArray(
            groupIdsArrayName
        )

        bifurcationsIds = [
            bifsIdsArray.GetValue(index)
            for index in range(self._nbifurcations)
        ]

        if self._nbifurcations > intZero:
            # Compute bifurcation
            bifVectors = vmtkscripts.vmtkBifurcationVectors()
            bifVectors.ReferenceSystems = bifsRefSystem.ReferenceSystems
            bifVectors.Centerlines = branches.Centerlines
            bifVectors.RadiusArrayName = radiusArrayName
            bifVectors.GroupIdsArrayName = groupIdsArrayName
            bifVectors.TractIdsArrayName = tractIdsArrayName
            bifVectors.BlankingArrayName = blankingArrayName
            bifVectors.CenterlineIdsArrayName = branches.CenterlineIdsArrayName

            bifVectors.ReferenceSystemsNormalArrayName = bifsRefSystem.ReferenceSystemsNormalArrayName
            bifVectors.ReferenceSystemsUpNormalArrayName = bifsRefSystem.ReferenceSystemsUpNormalArrayName

            bifVectors.NormalizeBifurcationVectors = True
            bifVectors.Execute()

            for index in bifurcationsIds:
                # Filter bifurcation reference system to get only one bifurcation
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

                system = bifurcationSystem.GetOutput()
                vectors = bifurcationVectors.GetOutput()

                self._bifurcations.append(Bifurcation(system, vectors))


    def _split_branches(self):
        """Split vasculature into branches.

        Given the vasculature centerlines, slits it into
        its constituent branches. Return a list of branch
        objects.

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
        branches = tools.extractPortion(
            branches.Centerlines,
            blankingArrayName,
            branchesId
        )

        maxGroupId = max(
            branches.GetCellData().GetArray(
                groupIdsArrayName
            ).GetRange()
        )

        for branchId in range(int(maxGroupId) + 1):
            branch = tools.extractPortion(
                branches,
                groupIdsArrayName,
                branchId
            )

            if branch.GetLength() != 0.0:
                self._branches.append(Branch(branch))


    def computeWallThicknessArray(self):
        """Add thickness array to the vascular surface."""

        vasculatureThickness = customscripts.vmtkSurfaceVasculatureThickness()
        vasculatureThickness.Surface = self._surface
        vasculatureThickness.Centerlines = self._centerlines
        vasculatureThickness.Aneurysm = self._with_aneurysm
        vasculatureThickness.SelectAneurysmRegions = False

        vasculatureThickness.SmoothingIterations = 20
        vasculatureThickness.GenerateWallMesh = False
        vasculatureThickness.Execute()

        self._surface = vasculatureThickness.Surface

    def getSurface(self):
        return self._surface

    def getAneurysm(self):
        return self._aneurysm_model

    def getCenterlines(self):
        return self._centerlines

    def getInletCenters(self):
        return self._inlet_centers

    def getOutletCenters(self):
        return self._outlet_centers

    def getBifurcations(self):
        return self._bifurcations

    def getNumberOfBifurcations(self):
        return self._nbifurcations

    def getBranches(self):
        return self._branches

if __name__ == '__main__':
    # Testing: generate a report on the vasculature being loaded
    import sys
    from pprint import pprint

    filename = sys.argv[1]
    withAneurysm = int(sys.argv[2])
    manual = int(sys.argv[3])
    renderSurfaces = int(sys.argv[4])
    outFile = sys.argv[5]

    print("-------------- Processing file "+filename+" -------------", end='\n')
    vasculatureSurface = tools.readSurface(filename)

    case = Vasculature(
        vasculatureSurface,
        with_aneurysm=withAneurysm,
        manual_aneurysm=manual
    )

    # Inspection
    if renderSurfaces:
        tools.viewSurface(case.getSurface(), array_name="Local_Shape_Type")
        tools.viewSurface(case.getCenterlines())

    print("Centerline arrays", end='\n')
    for index in range(case.getCenterlines().GetPointData().GetNumberOfArrays()):
        print('\t'+case.getCenterlines().GetPointData().GetArray(index).GetName(),
              end='\n')

    # Inlet and outlets
    print("Inlet:", end='\n')
    pprint(case.getInletCenters())

    print("Outlets: ", end='\n')
    pprint(case.getOutletCenters())

    print('\n')

    # Bifurcations
    print("Bifurcation number = ", case.getNumberOfBifurcations(), end='\n')
    pprint(case.getBifurcations()[0].inPlaneVectors)

    # Compute wall thickness
    # case.computeWallThicknessArray()
    # tools.viewSurface(case.getSurface(),array_name="Thickness")
    tools.writeSurface(case.getSurface(), outFile)
    # tools.writeSurface(case.getSurface(), '/home/iagolessa/tmp.vtp')

    print('\n')
    # Inspect branches
    print("Branches number = ", len(case.getBranches()), end='\n')

    for branch in case.getBranches():
        if renderSurfaces:
            tools.viewSurface(branch.getBranch())
        print('\tBranch Length = ', branch.getLength(), end='\n')

    print('\n')
    # If has aneurysm
    if withAneurysm:
        print("Showing aneurysm properties", end='\n')
        if renderSurfaces:
            tools.viewSurface(case.getAneurysm().getSurface())
            tools.viewSurface(case.getAneurysm().getHullSurface())

        # Get aneurysm filename 
        aneurysmFileName = filename.replace('model.vtp', 'aneurysm.vtp')

        # tools.writeSurface(case.getAneurysm().getSurface(), aneurysmFileName)
        obj = case.getAneurysm()

        print("\tAneurysms parameters: ", end='\n')
        for parameter in dir(obj):
            if parameter.startswith('get'):
                attribute = getattr(obj, parameter)()

                if type(attribute) == float or type(attribute) == tuple:
                    print('\t'+ parameter.strip('get') +
                          ' = '+str(attribute), end='\n')