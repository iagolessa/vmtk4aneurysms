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

from constants import *
import polydatatools as tools 



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
            self.normal = referenceSystem.GetPointData().GetArray('Normal').GetTuple(0)
            self.upnormal = referenceSystem.GetPointData().GetArray('UpNormal').GetTuple(0)
            self.numberOfBranches = vectors.GetPoints().GetNumberOfPoints()

            # Collect point and vectors
            points    = vectors.GetPoints()
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
            errorMessage = "Error building bifurcation!"+ \
                            "It seems you did not pass me a vtkPolyData."

            print(errorMessage)


class Vasculature:
    """Class of vascular models represented by surfaces.

    This class presents an interface to work with vascular 
    models represented as surfaces as vtkPolyData. The surface
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
        self._surface     = surface
        self._centerlines = None
        self._aneurysm_point   = None

        # Compute open boundaries centers
        self._inlet_centers  = list()
        self._outlet_centers = list()

        # Flags
        self._with_aneurysm   = with_aneurysm
        self._manual_aneurysm = manual_aneurysm

        # Compute morphology
        self._compute_open_centers()

        print('Computing centerlines...', end='\n')
        self._centerlines = self._generate_centerlines(
                                self._surface,
                                self._inlet_centers,
                                self._outlet_centers
                            )

        self._compute_centerline_geometry()

        # Collect bifurcations to this list
        print('Collecting bifurcations...', end='\n')
        self.bifurcations = list()
        self._compute_bifurcations_geometry()

        # Delineating aneurysm
        if self._with_aneurysm:
            print("Extracting aneurysm surface")

            if self._manual_aneurysm:
                extractAneurysm = vmtkscripts.vmtkExtractAneurysm()
                extractAneurysm.Surface = self._surface
                extractAneurysm.Execute()

                aneurysm_surface = extractAneurysm.AneurysmSurface

                self._aneurysm_model = aneurysm.Aneurysm(
                                            aneurysm_surface,
                                            **aneurysm_prop
                                        )

            else:
                self._select_aneurysm_point()

    def _compute_open_centers(self):
        """Compute barycenters of inlets and outlets.

        Computes the geometric center of each open boundary
        of the model. Computes two lists: one with the inlet
        coordinates (tuple) and another with the outlets
        coordinates also as tuples of three components:

            Inlet coords:  [(xi, yi, zi)]
            Outlet coords: [(xo1,yo1,zo1),
                            (xo2,yo2,zo2),
                            ...
                            (xon,yon,zon)]

        for a model with a single inlet and n outlets. The
        inlet is defined as the open boundary with largest radius.
        """

        print('Getting open boundaries centers.', end='\n')
        radiusArray = 'Radius'
        normalsArray = 'Normals'
        pointArrays = ['Point1', 'Point2']

        referenceSystems = vtkvmtk.vtkvmtkBoundaryReferenceSystems()
        referenceSystems.SetInputData(self._surface)
        referenceSystems.SetBoundaryRadiusArrayName(radiusArray)
        referenceSystems.SetBoundaryNormalsArrayName(normalsArray)
        referenceSystems.SetPoint1ArrayName(pointArrays[intZero])
        referenceSystems.SetPoint2ArrayName(pointArrays[intOne])
        referenceSystems.Update()

        openBoundariesRefSystem = referenceSystems.GetOutput()
        nOpenBoundaries = openBoundariesRefSystem.GetPoints().GetNumberOfPoints()

        maxRadius = openBoundariesRefSystem.GetPointData().GetArray(
            radiusArray).GetRange()[intOne]

        for i in range(nOpenBoundaries):
            # Get radius and center
            center = tuple(openBoundariesRefSystem.GetPoints().GetPoint(i))
            radius = openBoundariesRefSystem.GetPointData().GetArray(radiusArray).GetValue(i)

            if radius == maxRadius:
                self._inlet_centers.append(center)
            else:
                self._outlet_centers.append(center)

    def _generate_centerlines(self, 
                              surface, 
                              source_points, 
                              target_points):
        """Compute centerlines automatically, given source and target points."""

        # Get inlet and outlet centers of surface
        CapDisplacement  = 0.0
        FlipNormals      = 0
        CostFunction     = '1/R'
        AppendEndPoints  = 1
        CheckNonManifold = 0

        Resampling = 1
        ResamplingStepLength = 0.1
        SimplifyVoronoi = 0
        RadiusArrayName = "MaximumInscribedSphereRadius"

        # Clean and triangulate
        surface = tools.cleaner(self._surface)

        surfaceTriangulator = vtk.vtkTriangleFilter()
        surfaceTriangulator.SetInputData(surface)
        surfaceTriangulator.PassLinesOff()
        surfaceTriangulator.PassVertsOff()
        surfaceTriangulator.Update()

        # Cap surface
        surfaceCapper = vtkvmtk.vtkvmtkCapPolyData()
        surfaceCapper.SetInputConnection(surfaceTriangulator.GetOutputPort())
        surfaceCapper.SetDisplacement(CapDisplacement)
        surfaceCapper.SetInPlaneDisplacement(CapDisplacement)
        surfaceCapper.Update()

        centerlineInputSurface = surfaceCapper.GetOutput()

        # Get source and target ids of closest point
        sourceSeedIds = vtk.vtkIdList()
        targetSeedIds = vtk.vtkIdList()

        pointLocator = vtk.vtkPointLocator()
        pointLocator.SetDataSet(centerlineInputSurface)
        pointLocator.BuildLocator()

        for point in source_points:
            id_ = pointLocator.FindClosestPoint(point)
            sourceSeedIds.InsertNextId(id_)

        for point in target_points:
            id_ = pointLocator.FindClosestPoint(point)
            targetSeedIds.InsertNextId(id_)

        # Compute centerlines
        centerlineFilter = vtkvmtk.vtkvmtkPolyDataCenterlines()
        centerlineFilter.SetInputData(centerlineInputSurface)

        centerlineFilter.SetSourceSeedIds(sourceSeedIds)
        centerlineFilter.SetTargetSeedIds(targetSeedIds)

        centerlineFilter.SetRadiusArrayName(RadiusArrayName)
        centerlineFilter.SetCostFunction(CostFunction)
        centerlineFilter.SetFlipNormals(FlipNormals)
        centerlineFilter.SetAppendEndPointsToCenterlines(AppendEndPoints)
        centerlineFilter.SetSimplifyVoronoi(SimplifyVoronoi)

        centerlineFilter.SetCenterlineResampling(Resampling)
        centerlineFilter.SetResamplingStepLength(ResamplingStepLength)
        centerlineFilter.Update()

        return centerlineFilter.GetOutput()


    def _compute_centerline_geometry(self):
        """Compute centerline sections and geometry."""

        calcGeometry = vmtkscripts.vmtkCenterlineGeometry()
        calcGeometry.Centerlines = self._centerlines
        calcGeometry.Execute()

        # Computation of centerlines attributes (parallel theory)
        calcAttributes = vmtkscripts.vmtkCenterlineAttributes()
        calcAttributes.Centerlines = calcGeometry.Centerlines
        calcAttributes.Execute()

        self._centerlines = calcAttributes.Centerlines

    def _compute_bifurcations_geometry(self):
        """Collect bifurcations and computes their geometry.

        Identifies the bifurcations of the input surface model 
        and gather their information in a list of bifurcations.
        """

        # Split surface into branches
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
        self.numberOfBifurcations = bifsRefSystem.ReferenceSystems.GetPoints().GetNumberOfPoints()
        bifsIdsArray = bifsRefSystem.ReferenceSystems.GetPointData().GetArray(groupIdsArrayName)

        bifurcationsIds = [
                bifsIdsArray.GetValue(index) 
                for index in range(self.numberOfBifurcations)
            ]

        if self.numberOfBifurcations > intZero:
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
                bifsRefSystem.ReferenceSystems.GetPointData().SetActiveScalars(groupIdsArrayName)
                bifurcationSystem = vtk.vtkThresholdPoints()
                bifurcationSystem.SetInputData(bifsRefSystem.ReferenceSystems)
                bifurcationSystem.ThresholdBetween(index, index)
                bifurcationSystem.Update()

                bifVectors.BifurcationVectors.GetPointData().SetActiveScalars(
                    bifVectors.BifurcationGroupIdsArrayName)
                bifurcationVectors = vtk.vtkThresholdPoints()
                bifurcationVectors.SetInputData(bifVectors.BifurcationVectors)
                bifurcationVectors.ThresholdBetween(index, index)
                bifurcationVectors.Update()

                system = bifurcationSystem.GetOutput()
                vectors = bifurcationVectors.GetOutput()

                self.bifurcations.append(Bifurcation(system, vectors))

    def _select_aneurysm_point(self):
        """Enable selection of aneurysm point."""

        # Select aneurysm tip point 
        pickPoint = tools.PickPointSeedSelector()
        pickPoint.SetSurface(self._surface)
        pickPoint.InputInfo("Select point on the aneurysm surface")
        pickPoint.Execute()

        self._aneurysm_point = pickPoint.PickedSeeds.GetPoint(0)


    def computeWallThicknessArray(self):
        """Add thickness array to the vascular surface."""

        vasculatureThickness = vmtkscripts.vmtkSurfaceVasculatureThickness()
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
