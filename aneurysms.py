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
from .lib import names
from .lib import constants as const
from .lib import polydatatools as tools
from .lib import polydatageometry as geo
from .lib import polydatamath as pmath

_cellEntityIdsArrayName = "CellEntityIds"

# Field names

# Name of the field defined on a vascular surface that identifies the aneurysm
# with zero values and one the rest of the surface. The value 0.5, hence,
# identifies the aneurysm neck path (see 'NeckIsoValue')
AneurysmNeckArrayName = 'AneurysmNeckContourArray'
NeckIsoValue = 0.5

# Name of the field defined on a vascular surface that identifies the parent
# artery with zero values and one the rest of the surface.
ParentArteryArrayName = 'ParentArteryContourArray'

def SelectAneurysm(
        surface: names.polyDataType
    )   -> names.polyDataType:
    """Compute array marking the aneurysm neck.

    Given a vasculature with an aneurysm, prompt the user to draw the aneurysm
    neck on the surface. An array (field) is then defined on the surface with
    value 0 on the aneurysm and 1 out of the aneurysm. Return a copy of the
    vascular surface with 'AneurysmNeckContourArray' field defined on it.

    .. warning::
        VMTK uses its length dimensions in millimeters. Since this function is
        intended to operate on surfaces that were used in an OpenFOAM
        simulation, it must be already in meters. So we scaled it to
        millimeters here so the smoothing algorithm works as intended.

    .. warning::
        The smoothing array script works better on good quality triangle
        surfaces, hence the function operates on a remeshed surface with good
        quality triangles and map the results back to the original surface.
    """

    # Keep reference to surface, because the region drawing script triangulates
    # the output
    originalSurface = surface

    scaledSurface = tools.ScaleVtkObject(surface, const.millimeterToMeterFactor)

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
    rescaledSurface = tools.ScaleVtkObject(smoother.Surface,
                                         1.0/const.millimeterToMeterFactor)

    # Map the field back to the original surface
    surfaceProjection = vtkvmtk.vtkvmtkSurfaceProjection()
    surfaceProjection.SetInputData(originalSurface)
    surfaceProjection.SetReferenceSurface(rescaledSurface)
    surfaceProjection.Update()

    return surfaceProjection.GetOutput()

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
        self._ostium_surface = self._gen_ostium_surface()
        self._ostium_normal_vector = self._gen_ostium_normal_vector()

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
        """Cap aneurysm neck with triangles.

        Return the aneurysm surface 'capped', i.e. with a surface covering the
        neck region. The surface is created with the vtkvmtkCapPolyData()
        filter and build this neck or ostium surface by joining the neck
        vertices with its barycenter with triangles. The original aneurysm
        surface and the neck one are defined by a CellEntityIds array defined
        on them, with zero values on the ostium surface.
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
        capper.SetCellEntityIdOffset(-1) # The ostium surface will be 0
        capper.Update()

        return capper.GetOutput()

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

    def _neck_contour(self):
        """Return boundary of aneurysm surface (== neck contour)"""
        boundaryExtractor = vtkvmtk.vtkvmtkPolyDataBoundaryExtractor()
        boundaryExtractor.SetInputData(self._aneurysm_surface)
        boundaryExtractor.Update()

        return boundaryExtractor.GetOutput()

    def _neck_barycenter(self):
        """Return the neck contour barycenter as a Numpy array."""

        # Get neck contour
        neckContour = self._neck_contour()

        return geo.ContourBarycenter(neckContour)

    def _gen_ostium_surface(self):
        """Generate aneurysm' ostium surface."""

        # Use thrshold filter to get neck plane
        # Return a vtkUnstructuredGrid -> needs conversion to vtkPolyData
        getNeckSurface = vtk.vtkThreshold()
        getNeckSurface.SetInputData(self._cap_aneurysm())
        getNeckSurface.SetInputArrayToProcess(0,0,0,1,_cellEntityIdsArrayName)
        getNeckSurface.ThresholdBetween(self._neck_index, self._neck_index)
        getNeckSurface.Update()

        # Converts vtkUnstructuredGrid -> vtkPolyData
        neckSurface = tools.UnsGridToPolyData(getNeckSurface.GetOutput())

        ostiumRemesher = vmtkscripts.vmtkSurfaceRemeshing()
        ostiumRemesher.Surface = tools.Cleaner(neckSurface)
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
        """Calculate the normal vector to the aneurysm ostium surface/plane.

        The outwards normal unit vector to the ostium surface is computed by
        summing the normal vectors to each cell of the ostium surface.
        Rigorously, the neck plane vector should be computed with the actual
        neck *plane*, however, there are other ways to compute the aneurysm
        neck which is not based on a plane surface. In this scenario, it is
        robust enough to employ the approach used here because it provides a
        'sense of normal direction' to the neck line, be it a 3D curved path in
        space.

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
        ostiumSurface = tools.UnsGridToPolyData(getNeckSurface.GetOutput())

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

        Compute neck diameter, defined as the hydraulic diameter of the ostium
        surface or plane:

        .. math::
            D_n = 4A_n/p_n

        where :math:`A_n` is the ostium surface area and :math:`p_n` is its
        perimeter.
        """
        ostiumPerimeter = geo.ContourPerimeter(self._neck_contour())

        if ostiumPerimeter == 0.0:
            sys.exit("Ostium perimeter is zero")

        return const.four*self._ostium_area/ostiumPerimeter

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
