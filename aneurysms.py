"""Module defining the Aneurysm class."""

import sys
import vtk
import numpy as np
from vtk.numpy_interface import dataset_adapter as dsa

from vmtk import vtkvmtk
from vmtk import vmtkscripts
from scipy.spatial import ConvexHull

# Local modules
from .lib import names
from .lib import constants as const
from .lib import polydatatools as tools
from .lib import polydatageometry as geo

_cellEntityIdsArrayName = "CellEntityIds"

# Field names
AneurysmNeckArrayName = 'AneurysmNeckContourArray'
ParentArteryArrayName = 'ParentArteryContourArray'

NeckIsoValue = 0.5

def SelectAneurysm(surface: names.polyDataType) -> names.polyDataType:
    """Compute array marking the aneurysm neck.

    Given a vasculature with the an aneurysm, prompts the user to draw the
    aneurysm neck on the surface. Th function then defines an array on the
    surface with value 0 on the aneurysm and 1 out of the aneurysm .

    Note: VMTK uses its length dimensions in millimeters. Since this function
    is intended to operate on surfaces that were used in an OpenFOAM
    simulation, it must be already in meters. So we scaled it to millimeters
    here so the smoothing algorithm works as intended. Also, the smoothing
    array script works better on good quality triangle surfaces, hence the
    function operates on a remeshed surface with good quality triangles and map
    the results back to the original surface.
    """

    # Keep reference to surface, because the region drawing script triangulates
    # the output
    originalSurface = surface

    scaledSurface = tools.ScaleSurface(surface, const.millimeterToMeterFactor)

    # It is better to remesh the triangulated surface because the triangulate
    # filter applied to a polygonal surface yields poor quality triangles
    triangulate = vtk.vtkTriangleFilter()
    triangulate.SetInputData(scaledSurface)
    triangulate.Update()

    remesher = vmtkscripts.vmtkSurfaceRemeshing()
    remesher.Surface = triangulate.GetOutput()
    remesher.ElementSizeMode = 'edgelength'
    remesher.TargetEdgeLength = 0.2
    remesher.TargetEdgeLengthFactor = 1.0
    remesher.PreserveBoundaryEdges = 1
    remesher.Execute()

    # Compute aneurysm contour
    aneurysmSelection = vmtkscripts.vmtkSurfaceRegionDrawing()
    aneurysmSelection.Surface = tools.Cleaner(remesher.Surface)
    aneurysmSelection.InsideValue  = 0.0 # the aneurysm portion
    aneurysmSelection.OutsideValue = 1.0
    aneurysmSelection.ContourScalarsArrayName = AneurysmNeckArrayName
    aneurysmSelection.Execute()

    smoother = vmtkscripts.vmtkSurfaceArraySmoothing()
    smoother.Surface = aneurysmSelection.Surface
    smoother.Connexity  = 1
    smoother.Iterations = 10
    smoother.SurfaceArrayName = aneurysmSelection.ContourScalarsArrayName
    smoother.Execute()

    # Scale bacj to meters
    rescaledSurface = tools.ScaleSurface(smoother.Surface,
                                         1.0/const.millimeterToMeterFactor)

    # Map the field back to the original surface
    surfaceProjection = vtkvmtk.vtkvmtkSurfaceProjection()
    surfaceProjection.SetInputData(originalSurface)
    surfaceProjection.SetReferenceSurface(rescaledSurface)
    surfaceProjection.Update()

    return surfaceProjection.GetOutput()

def SelectParentArtery(surface: names.polyDataType) -> names.polyDataType:
    """Compute array masking the parent vessel region."""
    parentArteryDrawer = vmtkscripts.vmtkSurfaceRegionDrawing()
    parentArteryDrawer.Surface = surface
    parentArteryDrawer.InsideValue = 0.0
    parentArteryDrawer.OutsideValue = 1.0
    parentArteryDrawer.ContourScalarsArrayName = ParentArteryArrayName
    parentArteryDrawer.Execute()

    smoother = vmtkscripts.vmtkSurfaceArraySmoothing()
    smoother.Surface = parentArteryDrawer.Surface
    smoother.Connexity = 1
    smoother.Iterations = 10
    smoother.SurfaceArrayName = parentArteryDrawer.ContourScalarsArrayName
    smoother.Execute()

    return smoother.Surface

class Aneurysm:
    """Representation for saccular intracranial aneurysms.

    Given a saccular aneurysm surface as a vtkPolyData object, creates a
    representation of the aneurysm with its geometrical parameters. The  input
    aneurysm surface must be open for correct computations. Note that the
    calculations of aneurysm parameters performed here are intended for a plane
    aneurysm neck. However, the computations will still occur for a generic
    neck contour and be relatively correct.
    """


    def __init__(self, surface, aneurysm_type='', status='', label=''):
        """Initiates aneurysm model.

        Given the aneurysm surface and its characteristics, initiates aneurysm
        model by computing its simplest size features: surface area, neck
        surface area, and volume.

        Arguments:
            surface (vtkPolyData) -- the aneurysm surface
            aneurysm_type (str) -- aneurysm type: terminal or lateral
            status (str) -- if rupture or unruptured
            label (str) -- an useful label
        """
        self.type = aneurysm_type
        self.label = label
        self.status = status
        self._neck_index = int(const.zero)

        self._aneurysm_surface = tools.Cleaner(surface)
        self._ostium_surface = self._gen_ostium_surface()
        self._ostium_normal_vector = self._gen_ostium_normal_vector()

        # Compute neck surface area
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
        """Cap aneurysm neck with triangles.

        Returns the aneurysm surface 'capped' with a surface covering the
        neck region. The surface is created with the vtkvmtkCapPolyData()
        filter and build this 'neck surface' by joining the neck vertices
        with the contour barycenter sing triangles. The original aneurysm
        surface and the neck one are defined by a CellEntityIds array defined
        on them, with zero values on the neck surface.
        """

        # TODO: I noticed that sometimes the cap
        # algorithm does not generate correct array values for each cap
        # Investigate that

        # The centerpoint approach seems to be the best
        capper = vtkvmtk.vtkvmtkCapPolyData()
        capper.SetInputData(self._aneurysm_surface)
        capper.SetDisplacement(const.zero)
        capper.SetInPlaneDisplacement(const.zero)

        # Alternative strategy: using the simple approach
        # capper = vtkvmtk.vtkvmtkSimpleCapPolyData()
        # capper.SetInputData(self._aneurysm_surface)

        # Common attributes
        capper.SetCellEntityIdsArrayName(_cellEntityIdsArrayName)
        capper.SetCellEntityIdOffset(-1) # The neck surface will be 0
        capper.Update()

        return capper.GetOutput()

    def _make_vtk_id_list(self, it):
        vil = vtk.vtkIdList()
        for i in it:
            vil.InsertNextId(int(i))
        return vil

    def _aneurysm_convex_hull(self):
        """Compute convex hull of closed surface.

        This function computes the convex hull set of an aneurysm surface
        provided as a polyData set of VTK.  It uses internally the
        scipy.spatial package.
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

    def _neck_contour(self):
        """Get boundary of aneurysm surface (== neck contour)"""
        boundaryExtractor = vtkvmtk.vtkvmtkPolyDataBoundaryExtractor()
        boundaryExtractor.SetInputData(self._aneurysm_surface)
        boundaryExtractor.Update()

        return boundaryExtractor.GetOutput()

    def _neck_barycenter(self):
        """Computes and return the neck line barycenter as a Numpy array."""

        # Get neck contour
        neckContour = self._neck_contour()

        return geo.ContourBarycenter(neckContour)


    def _gen_ostium_surface(self):
        """Generate aneurysm neck plane/surface.

        Fill the ostium region with a surface, defined as the aneurysm neck
        surface, or plane, if an algoithm to actually 'cut' the neck plane
        was used.
        """

        # Use thrshold filter to get neck plane
        # Return a vtkUnstructuredGrid -> needs conversion to vtkPolyData
        getNeckSurface = vtk.vtkThreshold()
        getNeckSurface.SetInputData(self._cap_aneurysm())
        getNeckSurface.SetInputArrayToProcess(0,0,0,1,_cellEntityIdsArrayName)
        getNeckSurface.ThresholdBetween(self._neck_index, self._neck_index)
        getNeckSurface.Update()

        # Converts vtkUnstructuredGrid -> vtkPolyData
        gridToSurfaceFilter = vtk.vtkGeometryFilter()
        gridToSurfaceFilter.SetInputData(getNeckSurface.GetOutput())
        gridToSurfaceFilter.Update()

        ostiumRemesher = vmtkscripts.vmtkSurfaceRemeshing()
        ostiumRemesher.Surface = tools.Cleaner(gridToSurfaceFilter.GetOutput())
        ostiumRemesher.ElementSizeMode = 'edgelength'
        ostiumRemesher.TargetEdgeLength = 0.15
        ostiumRemesher.TargetEdgeLengthFactor = 1.0
        ostiumRemesher.PreserveBoundaryEdges = 1
        ostiumRemesher.Execute()

        ostiumSmoother = vmtkscripts.vmtkSurfaceSmoothing()
        ostiumSmoother.Surface = tools.CleanupArrays(ostiumRemesher.Surface)
        ostiumSmoother.Method = 'taubin'
        ostiumSmoother.NumberOfIterations = 30
        ostiumSmoother.PassBand = 0.1
        ostiumSmoother.BoundarySmoothing = 0
        ostiumSmoother.Execute()

        return ostiumSmoother.Surface

    def _gen_ostium_normal_vector(self):
        """Calculate the normal vector to the aneurysm neck surface/plane.

        The outwards normal unit vector to the neck surface is computed by
        summing the normal vectors to each cell of the neck surface.
        Rigorously, the neck plane vector should be computed with the actual
        neck *plane*, however, there are other ways to compute the aneurysm
        neck which is not based on a plane surface. In this scenario, it is
        robust enough to employ the approach used here because it provides a
        'sense of normal direction' to the neck line, be it a 3D curved path
        in space.

        In any case, if an actual plane is passed, the function will work.
        """
        # Compute outwards normals
        normals = vtk.vtkPolyDataNormals()
        normals.SetInputData(self._cap_aneurysm())
        normals.ComputeCellNormalsOn()
        normals.ComputePointNormalsOff()
        normals.Update()

        # Get only ostium surface (id = 0)
        getNeckSurface = vtk.vtkThreshold()
        getNeckSurface.SetInputData(normals.GetOutput())
        getNeckSurface.SetInputArrayToProcess(0,0,0,1,_cellEntityIdsArrayName)
        getNeckSurface.ThresholdBetween(self._neck_index, self._neck_index)
        getNeckSurface.Update()

        # Converts vtkUnstructuredGrid -> vtkPolyData
        gridToSurfaceFilter = vtk.vtkGeometryFilter()
        gridToSurfaceFilter.SetInputData(getNeckSurface.GetOutput())
        gridToSurfaceFilter.Update()

        ostiumSurface = gridToSurfaceFilter.GetOutput()

        # Convert to points
        cellCenter = vtk.vtkCellCenters()
        cellCenter.SetInputData(ostiumSurface)
        cellCenter.Update()

        # Use Numpy
        npSurface = dsa.WrapDataObject(cellCenter.GetOutput())

        neckNormalsVector = npSurface.GetPointData().GetArray("Normals").sum(axis=0)
        neckNormalsVector /= np.linalg.norm(neckNormalsVector)

        return tuple(neckNormalsVector)

    def _max_normal_height_vector(self):
        """Compute verctor with maximum normal height.

        Function to compute the vector from the neck contour barycenter and the
        fartest point on the aneurysm surface that have the maximum normal
        distance from the neck plane.
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

    # Compute 1D Size Indices
    def _compute_neck_diameter(self):
        """Computes neck diameter.

        Compute neck diameter, defined as the hydraulic diameter of the neck
        plane section:

            Dn = 4*An/pn

        where An is the aneurysm neck section area and pn is its perimeter.
        """
        ostiumPerimeter = geo.ContourPerimeter(self._neck_contour())

        if ostiumPerimeter == 0.0:
            sys.exit("Ostium perimeter is zero")

        return const.four*self._ostium_area/ostiumPerimeter

    def _compute_max_normal_height(self):

        vecMaxHeight = self._max_normal_height_vector()
        vecNormal = self._ostium_normal_vector

        return abs(vtk.vtkMath.Dot(vecMaxHeight, vecNormal))

    def _compute_max_diameter(self):
        """Finds the maximum diameter of parallel neck sections and its
        location.

        Computation of the maximum section diameter of the aneurysm, defined as
        the maximum diameter of the aneurysm cross sections that are parallel
        to the neck plane/surface, i.e. along the neck normal vector. Also
        returns the bulge height, i.e. the distance between the neck center
        and the location of the largest section, along a normal line to the
        ostium surface.
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
    def GetSurface(self):
        return self._aneurysm_surface

    def GetHullSurface(self):
        return self._hull_surface

    def GetOstiumSurface(self):
        return self._ostium_surface

    def GetAneurysmSurfaceArea(self):
        return self._surface_area

    def GetOstiumArea(self):
        return self._ostium_area

    def GetAneurysmVolume(self):
        return self._volume

    def GetHullSurfaceArea(self):
        return self._hull_surface_area

    def GetHullVolume(self):
        return self._hull_volume

    def GetNeckDiameter(self):
        """Return aneurysm neck diameter.

        The neck diameter is defined as the the hydraulic diameter of the
        ostium surface:

            Dn = 4*An/pn

        where An is the aneurysm ostium surface area and pn is its perimeter.
        The ideal computation would be based on a plane ostium section, but
        it will compute even for a curved ostium path.
        """

        return self._neck_diameter

    def GetMaximumNormalHeight(self):
        """Return maximum normal height.

        The maximum normal aneurysm height is defined as the
        maximum distance between the neck barycenter and the aneurysm surface.
        """

        return self._max_normal_height

    def GetMaximumDiameter(self):
        """Return the maximum aneurysm diameter."""

        return self._max_diameter

    # 2D Shape indices
    def GetAspectRatio(self):
        """Return the aspect ratio.

        Computes the aneurysm aspect ratio, defined as the ratio between the
        maximum perpendicular height and the neck diameter.
        """

        return self._max_normal_height/self._neck_diameter

    def GetBottleneckFactor(self):
        """Return the non-sphericity index.

        Computes the bottleneck factor, defined as the ratio between the
        maximum diameter and the neck diameter. This index represents the level
        to which the neck acts as a bottleneck to entry of blood during normal
        physiological function and to coils during endovascular procedures.
        """

        return self._max_diameter/self._neck_diameter

    def GetConicityParameter(self):
        """Return the conicity parameter.

        The conicity parameter was defined by Raghavan et al. (2005) as a shape
        metric for saccular IAs and it measures how far is the 'bulge' of the
        aneurysm, i.e. the section of largest section, from the aneurysm ostium
        surface or neck surface. In the way it was defined, it can vary from
        -0.5 (the bulge is at the dome) to 0.5 (bulge closer to neck).  CP =
        0.0 occurs when the bulge is at the midway from neck to the maximum
        normal height.
        """

        return 0.5 - self._bulge_height/self._max_normal_height

    # 3D Shape indices
    def GetNonSphericityIndex(self):
        """Return the non-sphericity index.

        Computes the non-sphericity index of an aneurysm surface, given by:

            NSI = 1 - (18pi)^(1/3) * Va^(2/3)/Sa

        where Va and Sa are the volume and surface area of the aneurysm.
        """
        factor = (18*const.pi)**(1./3.)

        area = self._surface_area
        volume = self._volume

        return const.one - (factor/area)*(volume**(2./3.))

    def GetEllipticityIndex(self):
        """Return ellipticity index.

        Computes the ellipiticity index of an aneurysm surface, given by:

            EI = 1 - (18pi)^(1/3) * Vch^(2/3)/Sch

        where Vch and Sch are the volume and surface area of the aneurysm
        convex hull.
        """

        factor = (18*const.pi)**(1./3.)

        area = self._hull_surface_area
        volume = self._hull_volume

        return const.one - (factor/area)*(volume**(2./3.))

    def GetUndulationIndex(self):
        """Return undulation index.

        Computes the undulation index of an aneurysm, defined as:

            UI = 1 - Va/Vch

        where Va is the aneurysm volume and Vch the volume of its convex hull.
        """
        return 1.0 - self._volume/self._hull_volume

    def GetCurvatureMetrics(self):
        """Compute curvature metrics.

        Based on local mean and Gaussian curvatures, compute their
        area-averaged values (MAA and GAA, respectively) and their L2-norm (MLN
        and GLN), as shown in

            Ma et al. (2004).
            Three-dimensional geometrical characterization
            of cerebral aneurysms.

        Return a dict with the metrics in the order (MAA, GAA, MLN, GLN).
        Assumes that both curvature arrays are defined on the aneurysm surface
        for a more accurate calculation avoiding border effects.
        """
        # Get arrays on the aneurysm surface
        nArrays = self._aneurysm_surface.GetCellData().GetNumberOfArrays()
        aneurysmCellData = self._aneurysm_surface.GetCellData()

        arrayNames = [aneurysmCellData.GetArray(array_id).GetName()
                      for array_id in range(nArrays)]

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

        aneurysmCellData = curvatureSurface.GetCellData()

        # TODO: improve this patch with vtkIntegrateAttributes I don't know why
        # the vtkIntegrateAttributes was not available in the python interface,
        # so I had to improvise (this version was the most efficient that I got
        # with pure python -- I tried others more numpythonic as well)

        # Helper functions
        getArea = lambda id_: curvatureSurface.GetCell(id_).ComputeArea()
        getValue = lambda id_, array: aneurysmCellData.GetArray(array).GetValue(id_)

        def GetCellCurvature(id_):
            cellArea = getArea(id_)
            GaussCurvature = getValue(id_, curvatureArrays.get('Gauss'))
            MeanCurvature  = getValue(id_, curvatureArrays.get('Mean'))

            return cellArea, MeanCurvature, GaussCurvature

        integralSquareGaussCurvature = 0.0
        integralSquareMeanCurvature  = 0.0
        integralGaussCurvature = 0.0
        integralMeanCurvature  = 0.0

        cellIds = range(curvatureSurface.GetNumberOfCells())

        # Map function to cell ids
        for area, meanCurv, GaussCurv in map(GetCellCurvature, cellIds):
            integralGaussCurvature += area*GaussCurv
            integralMeanCurvature  += area*meanCurv

            integralSquareGaussCurvature += area*(GaussCurv**2)
            integralSquareMeanCurvature  += area*(meanCurv**2)

        # Compute L2-norm of Gauss and mean curvature (GLN and MLN)
        # and their area averaged values (GAA and MAA)
        MAA = integralMeanCurvature/self._surface_area
        GAA = integralGaussCurvature/self._surface_area

        MLN = np.sqrt(integralMeanCurvature)/(4.0*const.pi)
        GLN = np.sqrt(integralGaussCurvature*self._surface_area)/(4.0*const.pi)

        return {"MAA": MAA, "GAA": GAA, "MLN": MLN, "GLN": GLN}
