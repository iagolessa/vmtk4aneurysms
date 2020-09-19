"""Module defining the Aneurysm class."""

import sys
import vtk
import numpy as np

from vmtk import vtkvmtk
from scipy.spatial import ConvexHull

# Local modules
from constants import *
import polydatatools as tools
import polydatageometry as geo


class Aneurysm:
    """Representation for saccular intracranial aneurysms.

    Given a saccular aneurysm surface as a vtkPolyData object, creates a
    representation of the aneurysm with its geometrical parameters. The  input
    aneurysm surface must be open for correct computations. Note that the
    calculations of aneurysm parameters performed here are intended for a plane
    aneurysm neck. However, the computations will still occur for a generic
    neck contour. 
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

        # Triangulate vtkPolyData surface
        # (input is cleaned surface)
        # cleanedSurface = tools.Cleaner(surface)
        self._aneurysm_surface = tools.Cleaner(surface)

        # triangulate = vtk.vtkTriangleFilter()
        # triangulate.SetInputData(cleanedSurface)
        # triangulate.Update()

        # self._aneurysm_surface = triangulate.GetOutput()

        # Compute neck surface area
        # Compute areas...
        self._surface_area = geo.SurfaceArea(self._aneurysm_surface)
        self._neck_plane_area = geo.SurfaceArea(self._neck_surface())

        # ... and volume
        self._volume = geo.SurfaceVolume(self._cap_aneurysm())

        # Computing hull properties
        self._hull_surface_area = 0.0
        self._hull_volume = 0.0
        self._hull_surface = self._aneurysm_convex_hull()

    def _cap_aneurysm(self):
        """Cap aneurysm neck with triangles. 

        Returns aneurysm surface capped with a plane of triangles. Uses VMTK's
        script 'vmtksurfacecapper'. 
        """

        # TODO: I noticed that sometimes (yes, this is subjective) the cap
        # algorithm does not generate correct array values for each cap
        # Investigate that
        cellEntityIdsArrayName = "CellEntityIds"

        capper = vtkvmtk.vtkvmtkCapPolyData()
        capper.SetInputData(self._aneurysm_surface)
        capper.SetDisplacement(intZero)
        capper.SetInPlaneDisplacement(intZero)
        capper.SetCellEntityIdsArrayName(cellEntityIdsArrayName)
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
        self._hull_surface_area = aneurysmHull.area - self._neck_plane_area

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


    def _neck_surface(self):
        """Generate aneurysm neck plane."""

        neckIndex = intZero
        CellEntityIdsArrayName = "CellEntityIds"

        # Use thrshold filter to get neck plane
        # Return a vtkUnstructuredGrid -> needs conversion to vtkPolyData
        getNeckPlane = vtk.vtkThreshold()
        getNeckPlane.SetInputData(self._cap_aneurysm())
        getNeckPlane.SetInputArrayToProcess(0, 0, 0, 1, CellEntityIdsArrayName)
        getNeckPlane.ThresholdBetween(neckIndex, neckIndex)
        getNeckPlane.Update()

        # Converts vtkUnstructuredGrid -> vtkPolyData
        gridToSurfaceFilter = vtk.vtkGeometryFilter()
        gridToSurfaceFilter.SetInputData(getNeckPlane.GetOutput())
        gridToSurfaceFilter.Update()

        return gridToSurfaceFilter.GetOutput()

    def _max_height_vector(self):
        """Compute maximum height vector.

        Function to compute the vector from the neck contour barycenter and the
        fartest point on the aneurysm surface.
        """

        neckContour = self._neck_contour()
        barycenter = self._neck_barycenter()

        # Get point in which distance to neck line baricenter is maximum
        maxDistance = float(intZero)
        maxVertex = None

        nVertices = self._aneurysm_surface.GetPoints().GetNumberOfPoints()

        for index in range(nVertices):
            vertex = self._aneurysm_surface.GetPoint(index)

            distance = geo.Distance(barycenter, vertex)

            if distance > maxDistance:
                maxDistance = distance
                maxVertex = vertex

        return tuple(np.subtract(maxVertex, barycenter))


    def _neck_plane_normal_vector(self):
        """Calculate the normal vector to the aneurysm neck plane."""

        # Get neck plane surface
        neckPlaneSurface = self._neck_surface()

        # Compute neck plane normal
        normals = vtk.vtkPolyDataNormals()
        normals.SetInputData(neckPlaneSurface)
        normals.ComputeCellNormalsOn()
        normals.ComputePointNormalsOff()
        normals.Update()

        xNormal = normals.GetOutput().GetCellData().GetNormals().GetRange(0)[0]
        yNormal = normals.GetOutput().GetCellData().GetNormals().GetRange(1)[0]
        zNormal = normals.GetOutput().GetCellData().GetNormals().GetRange(2)[0]

        return (xNormal, yNormal, zNormal)

    # Public interface
    def GetSurface(self):
        return self._aneurysm_surface

    def GetHullSurface(self):
        return self._hull_surface

    def GetAneurysmSurfaceArea(self):
        return self._surface_area

    def GetNeckPlaneArea(self):
        return self._neck_plane_area

    def GetAneurysmVolume(self):
        return self._volume

    def GetHullSurfaceArea(self):
        return self._hull_surface_area

    def GetHullVolume(self):
        return self._hull_volume

    # 1D Size Indices
    def GetNeckDiameter(self):
        """Return aneurysm neck diameter.

        Compute neck diameter, defined as the hydraulic diameter of the neck
        plane section:

            Dn = 4*An/pn

        where An is the aneurysm neck section area and pn is its perimeter.
        """

        neckContour = self._neck_contour()

        # Compute perimeter
        neckPerimeter = geo.ContourPerimeter(neckContour)

        return intFour*self._neck_plane_area/neckPerimeter

    def GetMaximumHeight(self):
        """Return maximum height.

        Aneurysm maximum aneurysm height is defined as the maximum distance
        between the neck barycenter and the aneurysm surface.
        """
        # Get neck contour
        vec = self._max_height_vector()
        return np.linalg.norm(vec)

    def GetMaximumNormalHeight(self):
        """Return maximum normal height.

        Computation of the maximum NORMAL aneurysm height, defined as the
        maximum distance between the neck barycenter and the aneurysm surface.
        """
        # Get max height vector and neck plane normal vector
        vecMaxHeight = self._max_height_vector()
        vecNormal = self._neck_plane_normal_vector()

        return abs(vtk.vtkMath.Dot(vecMaxHeight, vecNormal))

    def GetMaximumDiameter(self):
        """Return maximum aneurysm diameter.

        Computation of the maximum section diameter of the aneurysm, defined as
        the maximum diameter of the aneurysm cross sections that are parallel
        to the neck plane.
        """
        # Compute neck contour barycenter and normal vector
        normal = self._neck_plane_normal_vector()
        barycenter = self._neck_barycenter()

        # Get maximum normal height
        Hnmax = self.GetMaximumNormalHeight()

        # Form points of perpendicular line to neck plane
        nPoints = intThree * intTen
        dimensions = intThree

        t = np.linspace(0, Hnmax, nPoints)
        parameters = np.array([t]*dimensions).T

        points = np.array(barycenter) + parameters * np.array(normal)

        # Computes minimum hydraulic diameter
        maxDiameter = 0.0

        for center in points:
            plane = vtk.vtkPlane()
            plane.SetOrigin(center)
            plane.SetNormal(normal)

            # Cut initial aneurysm surface with create plane
            cutWithPlane = vtk.vtkCutter()
            cutWithPlane.SetInputData(self._aneurysm_surface)
            cutWithPlane.SetCutFunction(plane)
            cutWithPlane.Update()

            nVertices = cutWithPlane.GetOutput().GetNumberOfPoints()

            # Compute diamenetr if contour is not empty
            if nVertices > intZero:

                # Compute hydraulic diameter of cut line
                # TODO: will the error to compute the cut surface area due 
                # to open contour work here
                hydraulicDiameter = geo.ContourHydraulicDiameter(
                    cutWithPlane.GetOutput()
                )

                # Update minmum area
                if hydraulicDiameter > maxDiameter:
                    maxDiameter = hydraulicDiameter

        return maxDiameter

    # 2D Shape indices
    def GetAspectRatio(self):
        """Return the aspect ratio.

        Computes the aneurysm aspect ratio, defined as the ratio between the
        maximum perpendicular height and the neck diameter. 
        """

        return self.GetMaximumNormalHeight()/self.GetNeckDiameter()

    def GetBottleneckFactor(self):
        """Return the non-sphericity index.

        Computes the bottleneck factor, defined as the ratio between the
        maximum diameter and the neck diameter. This index represents the level
        to which the neck acts as a bottleneck to entry of blood during normal
        physiological function and to coils during endovascular procedures. 
        """

        return self.GetMaximumDiameter()/self.GetNeckDiameter()

    # 3D Shape indices
    def GetNonSphericityIndex(self):
        """Return the non-sphericity index.

        Computes the non-sphericity index of an aneurysm surface, given by:

            NSI = 1 - (18pi)^(1/3) * Va^(2/3)/Sa

        where Va and Sa are the volume and surface area of the aneurysm.
        """
        factor = (18*np.pi)**(1./3.)

        area = self._surface_area
        volume = self._volume

        return intOne - (factor/area)*(volume**(2./3.))

    def GetEllipticityIndex(self):
        """Return ellipticity index.

        Computes the ellipiticity index of an aneurysm surface, given by:

            EI = 1 - (18pi)^(1/3) * Vch^(2/3)/Sch

        where Vch and Sch are the volume and surface area of the aneurysm
        convex hull.
        """

        factor = (18*np.pi)**(1./3.)

        area = self._hull_surface_area
        volume = self._hull_volume

        return intOne - (factor/area)*(volume**(2./3.))

    def GetUndulationIndex(self):
        """Return undulation index.

        Computes the undulation index of an aneurysm, defined as:

            UI = 1 - Va/Vch

        where Va is the aneurysm volume and Vch the volume of its convex hull.
        """
        aneurysmVolume = self._volume

    def GetCurvatureMetrics(self):
        """Compute curvature metrics.

        Based on local mean and Gaussian curvatures, compute their
        area-averaged values (MAA and GAA, respectively) and their L2-norm (MLN
        and GLN), as shown in

            Ma et al. (2004).
            Three-dimensional geometrical characterization
            of cerebral aneurysms.

        Return a tuple with the metrics in the order (MAA, GAA, MLN, GLN).
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
            curvatureSurface = geo.SurfaceCurvature(self._aneurysm_surface)
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

        MLN = np.sqrt(integralMeanCurvature)/(4.0*Pi)
        GLN = np.sqrt(integralGaussCurvature*self._surface_area)/(4.0*Pi)

        return (MAA, GAA, MLN, GLN)
