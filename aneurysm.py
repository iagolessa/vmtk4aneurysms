"""Module defining the Aneurysm class."""

import sys
import vtk
import numpy as np

from vmtk import vtkvmtk
from scipy.spatial import ConvexHull

# Local modules
from constants import *
import polydatatools as tools 


class Aneurysm:  
    """Class representing saccular intracranial aneurysms.
        
    Made internal use of VTK and VMTK's classes for 
    vtkPolyData manipulations. Its main input is the
    aneurysm sac as a vtkPolyData object.
        
    The surface must be open for correct computation of
    its surface area.
    """
    
    # Constructor
    def __init__(self, surface, typeOfAneurysm, status, label):
        self.aneurysmType   = typeOfAneurysm
        self.aneurysmLabel  = label
        self.aneurysmStatus = status
        
        # Triangulate vtkPolyData surface
        triangulate = vtk.vtkTriangleFilter()
        triangulate.SetInputData(surface)
        triangulate.Update()
        
        self.aneurysmSurface = triangulate.GetOutput()
        
        # Cap aneurysm surface
        # (needed for correct calculation of aneurysm volume)
        cappedSurface = self._cap_aneurysm()
        
        # Compute attributes (3D size indices)
        # Surface area is computed with the open surface
        self.surfaceArea = self._surface_area(self.aneurysmSurface)
        self.volume      = self._surface_volume(cappedSurface)
        
        # Compute neck surface area
        self.neckPlaneArea = self._surface_area(self._neck_surface())
        
        # Computing hull properties
        self.hullSurface = self._aneurysm_convex_hull()
        
    
    def _cap_aneurysm(self):
        """Cap aneurysm neck with triangles. 
        
        Returns aneurysm surface capped with a plane
        of triangles. Uses VMTK's script 'vmtksurfacecapper'. 
        """
        
        cellEntityIdsArrayName = "CellEntityIds"
        
        capper = vtkvmtk.vtkvmtkCapPolyData()
        capper.SetInputData(self.aneurysmSurface)
        capper.SetDisplacement(intZero)
        capper.SetInPlaneDisplacement(intZero)
        capper.SetCellEntityIdsArrayName(cellEntityIdsArrayName)
        capper.SetCellEntityIdOffset(intZero)
        capper.Update()
        
        return capper.GetOutput()
    

    def _surface_area(self, surf):
        """Compute the surface area of an input surface."""

        surface_area = vtk.vtkMassProperties()
        surface_area.SetInputData(surf)
        surface_area.Update()
        
        return surface_area.GetSurfaceArea()

    
    def _surface_volume(self, surf):
        """Compute voluem of closed surface.
        
        Computes the volume of an assumed orientable 
        surface. Works internally with VTK, so it 
        assumes that the surface is closed. 
        """

        volume = vtk.vtkMassProperties()
        volume.SetInputData(surf)
        volume.Update()
        
        return volume.GetVolume()

    
    def _make_vtk_id_list(self,it):
        vil = vtk.vtkIdList()
        for i in it:
            vil.InsertNextId(int(i))
        return vil    
    
    def _aneurysm_convex_hull(self):
        """Compute convex hull of closed surface.
            
        This function computes the convex hull set of an
        aneurysm surface provided as a polyData set of VTK.
        It uses internally the scipy.spatial package.
        """

        # Convert surface points to numpy array
        nPoints = self.aneurysmSurface.GetNumberOfPoints()
        vertices  = list()
        
        for index in range(nPoints):
            vertex = self.aneurysmSurface.GetPoint(index)
            vertices.append(list(vertex))

        vertices = np.array(vertices)

        # Compute convex hull of points
        aneurysmHull = ConvexHull(vertices)

        # Get hull properties
        self.hullVolume = aneurysmHull.volume
        
        # Need to subtract neck area to 
        # compute correct hull surface area
        self.hullArea   = aneurysmHull.area - self.neckPlaneArea

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
        boundaryExtractor.SetInputData(self.aneurysmSurface)
        boundaryExtractor.Update()

        return boundaryExtractor.GetOutput()


    def _neck_barycenter(self):
        """Computes and return the neck line barycenter as a Numpy array."""
        # Get neck contour
        neckContour = self._neck_contour()
        neckPoints  = neckContour.GetPoints()

        barycenter  = np.zeros(intThree)
        vtkvmtk.vtkvmtkBoundaryReferenceSystems.ComputeBoundaryBarycenter(
            neckPoints,
            barycenter
        )

        return barycenter


    def _neck_surface(self):
        """Compute aneurysm neck plane."""
        
        neckIndex = intOne
        CellEntityIdsArrayName = "CellEntityIds"

        # Use thrshold filter to get neck plane
        # Return a vtkUnstructuredGrid -> needs conversion to vtkPolyData
        getNeckPlane = vtk.vtkThreshold()
        getNeckPlane.SetInputData(self._cap_aneurysm())
        getNeckPlane.SetInputArrayToProcess(0, 0, 0, 1, CellEntityIdsArrayName) 
        getNeckPlane.ThresholdBetween(neckIndex, neckIndex)
        getNeckPlane.Update()

        gridToSurfaceFilter = vtk.vtkGeometryFilter()
        gridToSurfaceFilter.SetInputData(getNeckPlane.GetOutput())
        gridToSurfaceFilter.Update()

        return gridToSurfaceFilter.GetOutput()
        

    def _max_height_vector(self):
        """Compute maximum height vector.

        Function to compute the vector from the neck 
        contour barycenter and the fartest point
        on the aneurysm surface
        """

        neckContour = self._neck_contour()
        barycenter  = self._neck_barycenter()

        # Get point in which distance to neck line baricenter is maximum
        maxDistance = float(intZero)
        maxVertex   = None

        nVertices   = self.aneurysmSurface.GetPoints().GetNumberOfPoints()

        for index in range(nVertices):
            vertex = self.aneurysmSurface.GetPoint(index)

            # Compute distance between point and neck barycenter
            distanceSquared = vtk.vtkMath.Distance2BetweenPoints(barycenter, vertex)
            distance = np.sqrt(distanceSquared)

            if distance > maxDistance: 
                maxDistance = distance
                maxVertex = vertex

        return np.array(maxVertex) - barycenter


    def _contour_perimeter(self,contour):
        """Coompute perimeter of a contour defined in 3D space."""

        nContourVertices = contour.GetNumberOfPoints()

        # Compute neck perimeter
        perimeter = intZero
        previous  = contour.GetPoint(intZero)

        for index in range(nContourVertices):
            if index > intZero:
                previous = contour.GetPoint(index - intOne)

            vertex = contour.GetPoint(index)

            # Compute distance between two consecutive points
            distanceSquared = vtk.vtkMath.Distance2BetweenPoints(previous, vertex)
            increment = np.sqrt(distanceSquared)    
            
            perimeter += increment
    
        return perimeter
    
    
    def _contour_hydraulic_diameter(self,contour):
        """Provided a poly line contour, compute its perimeter."""
        
        contourPerimeter = self._contour_perimeter(contour)

        # Compute contour fill area
        fillContour = vtk.vtkContourTriangulator()
        fillContour.SetInputData(contour)
        fillContour.Update()

        # Convert vtkUnstructuredData to vtkPolyData
        meshToSurfaceFilter = vtk.vtkGeometryFilter()
        meshToSurfaceFilter.SetInputData(fillContour.GetOutput())
        meshToSurfaceFilter.Update()

        computeArea = vtk.vtkMassProperties()
        computeArea.SetInputData(meshToSurfaceFilter.GetOutput())
        computeArea.Update()
        
        contourArea = computeArea.GetSurfaceArea()

        # Compute hydraulic diameter of neck
        return intFour*contourArea/contourPerimeter

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

        return np.array([xNormal, yNormal, zNormal])

    # 1D Size Indices
    def neckDiameter(self):
        """
            Compute neck diameter, defined as the hydraulic diameter
            of the neck plane section:

                Dn = 4*An/pn

            where An is the aneurysm neck section area and pn is its
            perimeter.
         """   

        # Get lenght of boundary neck (validate in ParaView)
        neckContour   = self._neck_contour()
        neckPerimeter = self._contour_perimeter(neckContour)
        
        # Compute hydraulic diameter of neck
        return intFour * self.neckPlaneArea/neckPerimeter


    def maximumHeight(self):
        """ 
            Computation of the maximum aneurysm height, 
            defined as the maximum distance between the 
            neck barycenter and the aneurysm surface.
        """
        # Get neck contour
        vec = self._max_height_vector()
        return np.linalg.norm(vec)


    def maximumNormalHeight(self):
        """ 
            Computation of the maximum NORMAL aneurysm 
            height, defined as the maximum distance between 
            the neck barycenter and the aneurysm surface.
        """

        # Get max height vector and neck plane normal vector 
        vecMaxHeight = self._max_height_vector()
        vecNormal    = self._neck_plane_normal_vector()

        return abs(vtk.vtkMath.Dot(vecMaxHeight,vecNormal))


    def maximumDiameter(self):
        """
            Computattion of the maximum section diameter of the aneurysm,
            defined as the maximum diameter of the aneurysm cross sections
            that are parallel to the neck plane.
        """
        # Compute neck contour barycenter and normal vector
        normal     = self._neck_plane_normal_vector()
        barycenter = self._neck_barycenter()

        
        # Get maximum normal height
        Hnmax = self.maximumNormalHeight()

        # Form points of perpendicular line to neck plane
        nPoints    = intThree * intTen
        dimensions = intThree

        t = np.linspace(0, Hnmax, nPoints)
        parameters = np.array([t]*dimensions).T

        points = barycenter + parameters * normal

        # Computes minimum hydraulic diameter
        maxDiameter = 0.0

        for center in points:
            plane = vtk.vtkPlane()
            plane.SetOrigin(center)
            plane.SetNormal(normal)

            # Cut initial aneurysm surface with create plane
            cutWithPlane = vtk.vtkCutter()
            cutWithPlane.SetInputData(self.aneurysmSurface)
            cutWithPlane.SetCutFunction(plane)
            cutWithPlane.Update()

            nVertices = cutWithPlane.GetOutput().GetNumberOfPoints()

            # Compute diamenetr if contour is not empty
            if nVertices > intZero:

                # Compute hydraulic diameter of cut line
                hydraulicDiameter = self._contour_hydraulic_diameter(
                                        cutWithPlane.GetOutput()
                                    )

                # Update minmum area 
                if hydraulicDiameter > maxDiameter: 
                    maxDiameter = hydraulicDiameter

        return maxDiameter    
    
    
    # 2D Shape indices
    def aspectRatio(self):
        """
            Computes the aneurysm aspect ratio, defined as the 
            ratio between the maximum perpendicular height and
            the neck diameter. 

        """
        
        return self.maximumNormalHeight()/self.neckDiameter()
    
    
    def bottleneckFactor(self):
        """
            Computes the bottleneck factor, defined as the 
            ratio between the maximum diameter and the neck
            diameter. This index represents the level 
            to which the neck acts as a bottleneck to entry of 
            blood during normal physiological function and to 
            coils during endovascular procedures. 
        """
        
        return self.maximumDiameter()/self.neckDiameter()
    
    
    
    # 3D Shape indices
    def nonSphericityIndex(self):
        """ Computes the non-sphericity index of an aneurysm 
            surface, given by:

                NSI = 1 - (18pi)^(1/3) * Va^(2/3)/Sa

            where Va and Sa are the volume and surface area of the
            aneurysm.
        """
        factor = (18*np.pi)**(1./3.)
        
        return 1 - factor/self.surfaceArea*(self.volume**(2./3.))
    
    
    def ellipticityIndex(self):
        """ Computes the ellipiticity index of an aneurysm 
            surface, given by:

                EI = 1 - (18pi)^(1/3) * Vch^(2/3)/Sch

            where Vch and Sch are the volume and surface area 
            of the aneurysm convex hull.
        """
        
        
        factor = (18*np.pi)**(1./3.)
        
        return intOne - factor/self.hullArea*(self.hullVolume**(2./3.))
    
    def undulationIndex(self):
        """
            Computes the undulation index of an aneurysm,
            defined as:
                
                UI = 1 - Va/Vch
            
            where Va is the aneurysm volume and Vch the
            volume of its convex hull.
        """
        
        return intOne - self.volume/self.hullVolume
#
# if __name__ == '__main__':
