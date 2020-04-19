import vtk
from vmtk import vmtkscripts
from vmtk import vtkvmtk

from constants import *


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
            points = vectors.GetPoints()
            pointData = vectors.GetPointData()

            self.points = [points.GetPoint(i)
                           for i in range(self.numberOfBranches)]
            self.vectors = [pointData.GetArray('BifurcationVectors').GetTuple(
                i) for i in range(self.numberOfBranches)]
            self.inPlaneVectors = [pointData.GetArray('InPlaneBifurcationVectors').GetTuple(
                i) for i in range(self.numberOfBranches)]

        except AttributeError:
            print(
                "Error building bifurcation! It seems you did not pass me a vtkPolyData.", end='\n')


class Vasculature:
    """Class of vascular models represented by surfaces.

    This class presents an interface to work with vascular 
    models represented as surfaces as vtkPolyData. The surface
    model must contain the open boundaries of the domain, i.e.
    its inlets and outlets of blood flow as perpendicular
    open sections (so far, it handles only vasculatures with 
    a single inlet, defined as the one with largest radius).
    """

    def __init__(self, surface, manual_centerline=True, with_aneurysm=True):
        print('Initiating model.', end='\n')
        self.surface = surface

        # Compute open boundaries centers
        self.inletCenters = []
        self.outletCenters = []

        # Switches
        self.ManualPickpoint = manual_centerline
        self.WithAneurysm = with_aneurysm

        # Compute morphology
        self._compute_open_centers()
        self._compute_centerline()
        self._compute_centerline_geometry()

        # Collect bifurcations to this list
        self.bifurcations = []
        self._compute_bifurcations_geometry()

    def _compute_open_centers(self):
        """Compute barycenters of inlets and outlets.

        Computes the geometric center of each open boundary
        of the model. Computes two lists: one with the inlet
        coordinates (tuple) and another with the outlets 
        coordinates also as tuples of three components:

            Inlet coords: [(xi, yi, zi)]
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
        referenceSystems.SetInputData(self.surface)
        referenceSystems.SetBoundaryRadiusArrayName(radiusArray)
        referenceSystems.SetBoundaryNormalsArrayName(normalsArray)
        referenceSystems.SetPoint1ArrayName(pointArrays[intZero])
        referenceSystems.SetPoint2ArrayName(pointArrays[intOne])
        referenceSystems.Update()

        openBoundariesRefSystem = referenceSystems.GetOutput()
        numberOfOpenBoundaries = openBoundariesRefSystem.GetPoints().GetNumberOfPoints()

        maxRadius = openBoundariesRefSystem.GetPointData().GetArray(
            radiusArray).GetRange()[intOne]

        for i in range(numberOfOpenBoundaries):
            # Get radius and center
            center = tuple(openBoundariesRefSystem.GetPoints().GetPoint(i))
            radius = openBoundariesRefSystem.GetPointData().GetArray(radiusArray).GetValue(i)

            if radius == maxRadius:
                self.inletCenters.append(center)
            else:
                self.outletCenters.append(center)

    def _compute_centerline(self):
        """Compute complete centerline.

        This code computes centerlines for a generic number of inlets
        It does that by computing the centerlines for each source barycenter
        and then appending the resulting centerlines with 'vmtksurfaceappend'
        """

        if self.ManualPickpoint:
            centerline = vmtkscripts.vmtkCenterlines()
            centerline.Surface = self.surface
            centerline.SeedSelectorName = 'openprofiles'
            centerline.AppendEndPoints = 1
            centerline.Execute()

            self.centerline = centerline.Centerlines

        else:
            # Aneurysm top point to get aneurysm centerline
            aneurysmTopPoint = []
            centerlinesList = []

            # Compute sequential list of outlets
            outletCenters = list(self.outletCenters[intZero])
            inletCenters = list(self.inletCenters[intZero])

            for center in self.outletCenters[intOne:]:
                outletCenters += center

            for center in self.inletCenters[intOne:]:
                inlettCenters += center

            for source in self.inletCenters:
                # Instantiate vmtkcenterline object
                centerline = vmtkscripts.vmtkCenterlines()
                centerline.Surface = self.surface
                centerline.SeedSelectorName = 'pointlist'
                centerline.SourcePoints = inletCenters
                centerline.TargetPoints = outletCenters + aneurysmTopPoint
                centerline.Execute()

                centerlinesList.append(centerline.Centerlines)

            # Provide managing exception if there is only one centerline
            centerlineMain = centerlinesList[intZero]

            # If more than one inlet, then the centerlines for
            # each inlet is appended here by a vmtk script
            if len(centerlinesList) > intOne:
                for centerline in centerlinesList[intOne:]:

                    centerlinesAppend = vmtkscripts.vmtkSurfaceAppend()
                    centerlinesAppend.Surface = centerlineMain
                    centerlinesAppend.Surface2 = centerline
                    centerlinesAppend.Execute()

                    # Store final centerlines to centerline main
                    # therefore, centerlineMain will store the final centerline
                    centerlineMain = centerlinesAppend.Surface

            self.centerline = centerlineMain

    def _compute_centerline_geometry(self):
        """Compute centerline sections and geometry."""

        calcGeometry = vmtkscripts.vmtkCenterlineGeometry()
        calcGeometry.Centerlines = self.centerline
        calcGeometry.Execute()

        # Computation of centerlines attributes (parallel theory)
        calcAttributes = vmtkscripts.vmtkCenterlineAttributes()
        calcAttributes.Centerlines = calcGeometry.Centerlines
        calcAttributes.Execute()

        self.centerline = calcAttributes.Centerlines

    def _compute_bifurcations_geometry(self):
        """Collect bifurcations and computes their geometry.

        Identifies the bifurcations of the input surface model 
        and gather their information in a list of bifurcations.
        """

        # Split surface into branches
        branches = vmtkscripts.vmtkBranchExtractor()
        branches.Centerlines = self.centerline
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
        bifurcationsIds = [bifsIdsArray.GetValue(
            i) for i in range(self.numberOfBifurcations)]

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

    def ComputeWallThicknessArray(self):
        """Add thickness array to the vascular surface."""

        vasculatureThickness = vmtkscripts.vmtkSurfaceVasculatureThickness()
        vasculatureThickness.Surface = self.surface
        vasculatureThickness.Centerlines = self.centerline
        vasculatureThickness.Aneurysm = self.WithAneurysm

        vasculatureThickness.SmoothingIterations = 20
        vasculatureThickness.GenerateWallMesh = False
        vasculatureThickness.Execute()

        self.surface = vasculatureThickness.Surface
