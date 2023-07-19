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

"""Collection of tools to compute geometric properties of VTK objects."""

import vtk
import math
import numpy as np
from typing import Union
from itertools import combinations
from scipy.spatial import ConvexHull

from vmtk import vtkvmtk
from numpy import array, multiply, zeros, where
from vtk.numpy_interface import dataset_adapter as dsa

from . import names
from . import constants as const
from . import polydatatools as tools
from . import polydatamath as pmath

def ComputeCellBarycenter(
        cell: vtk.vtkTriangle
    )   -> array:

    return array(
               [cell.GetPoints().GetPoint(idx)
                for idx in range(cell.GetNumberOfPoints())]
           ).mean(axis=0)

def Distance(
        point1: tuple,
        point2: tuple
    )   -> float:
    """Compute distance between two points."""
    sqrDistance = vtk.vtkMath.Distance2BetweenPoints(
        point1,
        point2
    )

    return math.sqrt(sqrDistance)

def SpatialGradient(
        vtk_object: Union[names.polyDataType, names.unstructuredGridType],
        field_name: str
    )   -> Union[names.polyDataType, names.unstructuredGridType]:
    """Compute gradient of cell or point field on a VTK object."""


    # Create gradient filter
    gradient = vtk.vtkGradientFilter()
    gradient.SetInputData(vtk_object)

    # Get the field associations CELLS or POINTS
    if field_name in tools.GetPointArrays(vtk_object):
        # 0 -> POINT field
        gradient.SetInputScalars(0, field_name)

    elif field_name in tools.GetCellArrays(vtk_object):
        # 1 -> CELL field
        gradient.SetInputScalars(1, field_name)

    else:
        raise ValueError("{} not in input VTK object.".format(field_name))

    gradient.SetResultArrayName(field_name + names.grad)
    gradient.ComputeDivergenceOff()
    gradient.ComputeQCriterionOff()
    gradient.ComputeVorticityOff()
    gradient.Update()

    return gradient.GetOutput()

def Divergence(
        vtk_object: Union[names.polyDataType, names.unstructuredGridType],
        field_name: str
    )   -> Union[names.polyDataType, names.unstructuredGridType]:
    """Compute the divergence of cell or point field on a VTK object."""

    divWssFieldName  = field_name + names.div
    gradWssFieldName = field_name + names.grad

    # Create gradient filter
    gradient = vtk.vtkGradientFilter()
    gradient.SetInputData(vtk_object)

    # Get the field associations CELLS or POINTS
    if field_name in tools.GetPointArrays(vtk_object):
        # 0 -> POINT field
        gradient.SetInputScalars(0, field_name)

        # Set flag so later in the function we know the
        # type of field to be removed
        isPointField = True

    elif field_name in tools.GetCellArrays(vtk_object):
        # 1 -> CELL field
        gradient.SetInputScalars(1, field_name)
        isPointField = False

    else:
        raise ValueError("{} not in input VTK object.".format(field_name))

    gradient.SetResultArrayName(gradWssFieldName)
    gradient.SetDivergenceArrayName(divWssFieldName)

    # Grad needs to be computed too apparently
    gradient.ComputeDivergenceOn()
    gradient.ComputeGradientOn()
    gradient.ComputeQCriterionOff()
    gradient.ComputeVorticityOff()
    gradient.Update()

    divSurface = gradient.GetOutput()

    # Delete gradient field
    if isPointField:
        divSurface.GetPointData().RemoveArray(gradWssFieldName)

    else:
        divSurface.GetCellData().RemoveArray(gradWssFieldName)

    return divSurface

def SurfaceGradient(
        surface: names.polyDataType,
        field_name: str,
        add_normal_gradient: bool=False
    )   -> names.polyDataType:
    """Compute surface gradient of field on a surface.

    Given the surface (vtkPolyData) and the scalar field name, compute the
    tangential or surface gradient of it on the surface.
    """

    cellData = surface.GetCellData()
    nArrays  = cellData.GetNumberOfArrays()

    # Compute normals, if necessary
    arrays = (cellData.GetArray(id_).GetName()
              for id_ in range(nArrays))

    if names.normals not in arrays:
        surface = Surface.Normals(surface)

    # Compute spatial gradient (adds field)
    surfaceWithGradient = SpatialGradient(surface, field_name)

    # GetArrays
    npSurface = dsa.WrapDataObject(surfaceWithGradient)
    getArray  = npSurface.GetCellData().GetArray

    normalsArray  = getArray(names.normals)
    gradientArray = getArray(field_name + names.grad)

    # Compute the normal gradient = vec(n) dot grad(field)
    normalGradient = multiply(normalsArray, gradientArray).sum(axis=1)

    # Compute the surface gradient
    surfaceGrad = gradientArray - normalGradient*normalsArray

    npSurface.CellData.append(
        surfaceGrad,
        field_name + names.sgrad
    )

    if add_normal_gradient:
        npSurface.CellData.append(
            normalGradient,
            field_name + names.ngrad
        )

    # Clean up
    npSurface.GetCellData().RemoveArray(field_name + names.grad)

    return npSurface.VTKObject

def ContourPerimeter(
        contour: names.polyDataType
    )   -> float:
    """Compute the perimeter of a contour defined in 3D space."""

    perimeter = 0.0

    # The approach employed here was chosen to keep a consistency
    # for the periemeter calculation of contours
    # obtained with both vtkvmtkBoundaryExtractor (single cell)
    # and vtkCutter.
    # TODO: The ideal alternative for this function is to use the
    # vtkIntegrateAttributes filter with VTK > 8.2
    #
    # Indicates that it was obtained with vtkvmtkBoundaryExtractor
    if contour.GetNumberOfCells() == 1:
        # Then use vtk-numpy interface
        npContour = dsa.WrapDataObject(contour)

        points = npContour.GetPoints()

        rotatedPoints = np.concatenate(
                            (np.array(points[1:]),
                             np.array([points[0]]))
                        )

        perimeter = sum([Distance(point1, point2)
                         for point1, point2 in zip(points, rotatedPoints)])

    else:
        # This one for some reason, does not work with the numpy interface
        perimeter = sum([math.sqrt(contour.GetCell(cell_id).GetLength2())
                         for cell_id in range(contour.GetNumberOfCells())])

    return perimeter

def ContourBarycenter(
        contour: names.polyDataType
    )   -> tuple:
    """Return contour barycenter."""

    # For the barycenter, the contour can be open
    contourPoints = contour.GetPoints()

    barycenter = [0.0, 0.0, 0.0]
    vtkvmtk.vtkvmtkBoundaryReferenceSystems.ComputeBoundaryBarycenter(
        contourPoints,
        barycenter
    )

    return tuple(barycenter)

def ContourPlaneArea(
        contour: names.polyDataType
    )   -> float:
    """Compute plane surface area enclosed by a 3D contour path."""

    # Fill contour
    # TODO: This algorithms seems to fail to triangulate the contour if
    # it is not closed. Need to find alternatives
    fillContour = vtk.vtkContourTriangulator()
    fillContour.SetInputData(contour)
    fillContour.Update()

    # Convert vtkUnstructuredData to vtkPolyData and compute area
    computeArea = vtk.vtkMassProperties()
    computeArea.SetInputData(
        tools.UnsGridToPolyData(fillContour.GetOutput())
    )
    computeArea.Update()

    return computeArea.GetSurfaceArea()

    # if computeArea.GetSurfaceArea() == 0.0:
    # raise ValueError
    # else:
    # return computeArea.GetSurfaceArea()
#
# except ValueError:
    # # Alternative procedure based on closing the aneurysm
    # pass

def ContourHydraulicDiameter(
        contour: names.polyDataType
    )   -> float:
    """Compute hydraulic diameter of a plane contour."""

    perimeter = ContourPerimeter(contour)
    area = ContourPlaneArea(contour)

    # Return hydraulic diameter of neck
    return const.four*area/perimeter

def ContourAverageDiameter(
        contour: names.polyDataType
    )   -> float:
    """Compute the averaged diameter of a 3D contour.

    Given a 3D contour in space, computes the average of the distance between
    its barycenter and the points on the contour. Returns the double of it,
    i.e. a measure of its diameter.
    """

    # Get contour barycenter
    contourBarycenter = ContourBarycenter(contour)

    # Compute distance between barycenter and contour points
    npContour = dsa.WrapDataObject(contour)
    contourPoints = npContour.GetPoints()

    contourRadius = np.array(
                        [Distance(point, contourBarycenter)
                         for point in contourPoints]
                    )

    # Return hydraulic diameter of neck
    return 2.0*contourRadius.mean()

# TODO: review this function to check closed contour
def ContourIsClosed(
        contour: names.polyDataType
    )   -> bool:
    """Check if contour (vtkPolyData) is closed."""
    nVertices = contour.GetNumberOfPoints()
    nEdges = contour.GetNumberOfCells()

    return nVertices == nEdges

def WarpVtkObject(
        vtk_object: Union[names.polyDataType, names.unstructuredGridType],
        field_name: str
    )   -> Union[names.polyDataType, names.unstructuredGridType]:
    """Warp (or deform) a VTK object with a vector field (cell or point)."""

    # First check whether the field is on the object
    # and convert it to point field if necessary
    if field_name not in tools.GetPointArrays(vtk_object):

        if field_name in tools.GetCellArrays(vtk_object):

            # Convert field to point data
            print(
                "Found " + field_name + " in cell data. Converting to point field."
            )

            vtk_object = tools.CellFieldToPointField(
                             vtk_object,
                             field_name
                         )

        else:
            raise ValueError(field_name + "not found in object.")

    else:
        pass

    # Set active field
    displData = vtk_object.GetPointData().GetArray(field_name)
    vtk_object.GetPointData().SetActiveVectors(displData.GetName())

    # Warp the surfaces at the two instants
    warpObject = vtk.vtkWarpVector()
    warpObject.SetInputData(vtk_object)
    warpObject.SetScaleFactor(1.0)
    warpObject.Update()

    return warpObject.GetOutput()

def WarpPolydata(
        polydata: names.polyDataType,
        field_name: str
    )   -> names.polyDataType:
    """Given a vtkPolyData with a field, warp it by the field."""

    # For backward compatibility
    return WarpVtkObject(polydata, field_name)

def SurfaceEuclideanDistanceToContour(
        surface: names.polyDataType,
        id_list: vtk.vtkCommonCorePython.vtkIdList,
        distance_array_name: str=names.EuclideanDistanceArrayName
    )   -> names.polyDataType:
    """Add the Euclidean distance from a loop of points defined on
    the surface.

    Given the list of ids of the points selected interactively by the user,
    compute the Euclidean distance on the surface to the contour formed by
    the points.  The set of points defined on the surface and the array
    name must be passed.
    """

    # Convert ids to points
    points = vtk.vtkPoints()
    points.SetNumberOfPoints(id_list.GetNumberOfIds())

    # Get points in surface
    for i in range(id_list.GetNumberOfIds()):
        pointId = id_list.GetId(i)

        point = surface.GetPoint(pointId)
        points.SetPoint(i, point)

    selectionFilter = vtk.vtkSelectPolyData()
    selectionFilter.SetInputData(surface)
    selectionFilter.SetLoop(points)
    selectionFilter.GenerateSelectionScalarsOn()
    selectionFilter.SetSelectionModeToSmallestRegion()
    selectionFilter.Update()

    # Change name of scalars
    selectionFilter.GetOutput().GetPointData().GetScalars().SetName(
        distance_array_name
    )

    # Add a little bit of smoothing
    surface = tools.SmoothSurfacePointField(
                  selectionFilter.GetOutput(),
                  distance_array_name
              )

    return surface

def SurfaceGeodesicDistanceToContour(
        surface: names.polyDataType,
        id_list: vtk.vtkCommonCorePython.vtkIdList,
        gdistance_array_name: str=names.GeodesicDistanceArrayName
    )   -> names.polyDataType:
    """Compute the geodesic distance from a contour on a surface.

    Given a surface and a list of point IDs that define a closed contour
    on the surface, compute the geodesic distance from this contour. Returns
    the input surface with the array of the geodesic distance defined on it.

    Once the contour must be closed, the algorithm will compute negative
    distances on the smallest region defined by the contour and positive
    on the rest.
    """

    selectionScalarsName = "Scalars"

    # Get points set on the surface
    seedPoints = vtk.vtkPoints()

    for idx in range(id_list.GetNumberOfIds()):

        # Get and store point
        point = surface.GetPoint(id_list.GetId(idx))
        seedPoints.InsertNextPoint(point)

    # Select the region inside the neck contour
    # (the aneurysm)
    selectionFilter = vtk.vtkSelectPolyData()
    selectionFilter.SetInputData(surface)
    selectionFilter.SetLoop(seedPoints)
    selectionFilter.GenerateSelectionScalarsOn()
    selectionFilter.SetSelectionModeToSmallestRegion()
    selectionFilter.Update()

    selectionFilter.GetOutput().GetPointData().GetScalars().SetName(
        selectionScalarsName
    )

    # Compute geodesic distance
    geodesicFastMarching = vtkvmtk.vtkvmtkNonManifoldFastMarching()
    geodesicFastMarching.SetInputData(selectionFilter.GetOutput())

    # Set F(x) == 1 to obtain the geodesic distance
    geodesicFastMarching.UnitSpeedOn()
    geodesicFastMarching.SetSolutionArrayName(gdistance_array_name)
    geodesicFastMarching.SetInitializeFromScalars(0)
    geodesicFastMarching.SeedsBoundaryConditionsOn()
    geodesicFastMarching.SetSeeds(id_list)
    geodesicFastMarching.PolyDataBoundaryConditionsOff()
    geodesicFastMarching.Update()

    # Use the numpy interface to change sign of distance array
    npSurface = dsa.WrapDataObject(geodesicFastMarching.GetOutput())

    # Get selection scalars
    selectionScalars = npSurface.PointData.GetArray(selectionScalarsName)
    gdistanceArray   = npSurface.PointData.GetArray(gdistance_array_name)

    # Where selection value is < 0.0, for this case, invert sign of
    # geodesic distance (inside the contour, in this case)
    updatedGdistanceArray = np.where(
                                selectionScalars < 0.0,
                                -1.0*gdistanceArray,
                                gdistanceArray
                            )

    npSurface.PointData.append(
        updatedGdistanceArray,
        gdistance_array_name
    )

    npSurface.GetPointData().RemoveArray(selectionScalarsName)

    return npSurface.VTKObject

# TODO: I have to optimized this combination
def _vec_even_bi_combination(values: list) -> list:

    # Make the combinations
    fieldVecComb = list(
                        combinations(
                            values, 2
                        )
                    )

    # Change order to the combination be the same as the theorem (even):
    # (tau_i,tau_j), (tau_j,tau_k), (tau_k,tau_i)
    # TODO: there might be a better and more elegant way to do this
    return [fieldVecComb[0],
            fieldVecComb[2],
            list(reversed(fieldVecComb[1]))]

# This function is the bottleneck, but this per-cell version was much more
# efficient combined with list comprehension below then to computing everything
# in a single function
def _get_cell_Poincare_matrices(
        tri_surface: names.polyDataType,
        vector_field_name: str,
        cell_id: int
    )   -> int:
    """Given the three IDs of the cell vertices, compute the determinant of the
    Poincaré matrices.

    The vector field must be a point field and the cell normal array must also
    be present. The surface must be comprised of triangles."
    """
    nCellPoints = tri_surface.GetCell(cell_id).GetNumberOfPoints()
    vectorField = tri_surface.GetPointData().GetArray(vector_field_name)

    cellNormal = list(
                    tri_surface.GetCellData().GetArray(
                        names.normals
                    ).GetTuple(cell_id)
                 )

    # Get the cell connectivity ids as list
    cellConnecIds = [tri_surface.GetCell(cell_id).GetPointIds().GetId(idx)
                     for idx in range(nCellPoints)]

    # Get the vector field in each vertice of the cell
    vecInCellVertices = map(
                            lambda idx: list(vectorField.GetTuple(idx)),
                            cellConnecIds
                        )

    # Make the combinations
    fieldVecComb = _vec_even_bi_combination(vecInCellVertices)

    # Add the cell normal to build the matrices
    cellMatrices = [list(lst) + [cellNormal]
                    for lst in fieldVecComb]

    return cellMatrices

def SurfaceConvexHull(
        surface: names.polyDataType,
        compute_normals: bool=False
    )   -> names.polyDataType:
    """Computes the convex hull of surface.

    Given an open or closed surface, compute its convex hull set and returns a
    triangulated surface representation of it.  It uses internally the
    scipy.spatial package.

    .. warning::
        The cell connectivity orientation returned by the ConvexHull function
        do not necessarily orients the normals outwards, as expected since the
        hull computation returns a closed surface.  If using the Surface.Volume
        function to compute the hull volume, then it is recommended to compute
        its normal vectors explicitly first and using the 'auto-orient'
        functionality on (argument 'auto_orient_if_closed' with
        Surface.Normals). This can also be accomplished here by turning the
        argument 'compute_normals' to True.
    """

    # Get vertices only
    vertices = np.array(
                   [surface.GetPoint(index)
                    for index in range(surface.GetNumberOfPoints())]
               )

    # Compute convex hull of points
    surfaceHull = ConvexHull(vertices)

    # Build poly data for convex hull
    hullSurface = tools.BuildPolyData(
                      surfaceHull.points,
                      surfaceHull.simplices
                  )

    # Compute normals before remove the constraint
    if compute_normals:
        # The hull is closed at this point
        # AFFECTS THE CONTOUR EXTRACTION!
        hullSurface = Surface.Normals(
                          hullSurface,
                          auto_orient_if_closed=True
                      )

    return hullSurface

def ComputeSurfaceVectorFixedPoints(
        surface: names.polyDataType,
        vec_field_name: str
    )   -> names.polyDataType:
    """Compute and characterize the fixed points of a surface vector field.

    Finds the fixed points by using the method used by:

        V. Mazzi et al., “A Eulerian method to analyze wall shear stress fixed
        points and manifolds in cardiovascular flows,” Biomechanics and
        Modeling in Mechanobiology, vol. 19, no. 5, pp. 1403–1423, Oct. 2020,
        doi: 10.1007/s10237-019-01278-3.
    """
    # Compue normals field
    if names.normals not in tools.GetCellArrays(surface):
        surface = Surface.Normals(surface)

    # Compute the gradient of the WSS too to be used later in the
    # characterization of the fixed points
    gradFieldName = vec_field_name + names.grad

    surface = SpatialGradient(
                  surface,
                  vec_field_name
              )

    # The theorem to compute the Poincaré index is valid for simplicial
    # surfaces ie for triangulated surfaces.  triangulate the surface first
    triagulation = vtk.vtkTriangleFilter()
    triagulation.SetInputData(surface)
    triagulation.Update()

    triSurface = triagulation.GetOutput()

    # get magnitude of WSS and convert it to point field
    triSurface = tools.CellFieldToPointField(
                    triSurface,
                    vec_field_name
                )

    PoincareDets = np.linalg.det(
                       [_get_cell_Poincare_matrices(
                           triSurface,
                           vec_field_name,
                           cell_id
                        ) for cell_id in range(triSurface.GetNumberOfCells())]
                   )

    # Compute sign of the dets and sum along axis
    # The fixed points are located where the sume is either 3 or -3
    sumDeterminants = np.sign(PoincareDets).sum(axis=1).astype(int)

    isFixedPoint = [val == -3 or val == 3
                    for val in sumDeterminants]

    PoincareIndex = np.where(
                        isFixedPoint,
                        sumDeterminants,
                        0
                    )

    # Now lets locate the cell ids with PoincareIndex != 0
    fixedPointCellIds = PoincareIndex.nonzero()[0]

    # Create different field defined on the fixed points
    npTriSurface = dsa.WrapDataObject(triSurface)
    gradField = npTriSurface.CellData.GetArray(gradFieldName)

    # Coordinates of points
    fpCoords = np.array(
                    [ComputeCellBarycenter(triSurface.GetCell(cellId))
                     for cellId in fixedPointCellIds]
                )

    # Poincare indices
    fpGradWssField = gradField[fixedPointCellIds]
    fpPoincareIndices = PoincareIndex[fixedPointCellIds]
    fpEigenValues = np.linalg.eig(fpGradWssField)[0]

    fpTypes = np.array(
                    [pmath.CharacterizeFixedPoint(eigenvalue)
                     for eigenvalue in fpEigenValues]
                )

    fpEigenVectors = np.real(np.linalg.eig(fpGradWssField)[1])

    fixedPointsData = tools.BuildPolyDataPoints(
                            fpCoords,
                            {"Poincareindex": fpPoincareIndices,
                             "EigenvectorsDir1": fpEigenVectors[:,:,0],
                             "EigenvectorsDir2": fpEigenVectors[:,:,1],
                             "EigenvectorsDir3": fpEigenVectors[:,:,2],
                             "FixedPointType": fpTypes}
                        )

    return fixedPointsData

class Surface():
    """Computational model of a three-dimensional surface."""

    def __init__(self, vtk_poly_data):
        """Build surface model from vtkPolyData.

        Given a vtkPolyData characterizing a surface in the 3D Euclidean space,
        automatically computes its outwards unit normal fiels, stored as
        'Normals' and its curvature type field based on the Gaussian and mean
        curvatures.
        """

        self._surface_object = Surface.Normals(vtk_poly_data)
        self._surface_object = Surface.Curvatures(self._surface_object)

    @classmethod
    def from_file(cls, file_name):
        """Build surface model from file."""

        return cls(tools.ReadSurface(file_name))

    @staticmethod
    def Area(surface_object: names.polyDataType) -> float:
        """Return the surface area in the units of the original data."""

        triangulate = vtk.vtkTriangleFilter()
        triangulate.SetInputData(surface_object)
        triangulate.Update()

        surface_area = vtk.vtkMassProperties()
        surface_area.SetInputData(triangulate.GetOutput())
        surface_area.Update()

        return surface_area.GetSurfaceArea()

    @staticmethod
    def Volume(surface_object: names.polyDataType) -> float:
        """Compute volume of closed surface.

        Computes the volume of an assumed orientable surface. Works internally
        with VTK, so it assumes that the surface is closed.
        """

        cleaner = vtk.vtkCleanPolyData()
        cleaner.SetInputData(surface_object)
        cleaner.Update()

        triangulate = vtk.vtkTriangleFilter()
        triangulate.SetInputConnection(cleaner.GetOutputPort())
        triangulate.Update()

        measureVolume = vtk.vtkMassProperties()
        measureVolume.SetInputConnection(triangulate.GetOutputPort())
        measureVolume.Update()

        # Check whether there was any error in the surface based on the
        # GeProjectedVolume output (according to the documentation of
        # vtkMassProperties)
        volTol = 1.0e-5 # according to the doc. should be 1e-5...

        volume = measureVolume.GetVolume()
        projectedVolume = measureVolume.GetVolumeProjected()

        normError = abs(volume - projectedVolume)/volume

        if normError > volTol:
            print(
                "Warning: the volume was returned, but check the surface input.\n"\
                "Either the polydata is not closed or the polydata \n"\
                "contains triangles that are flipped. Therefore, the volume \n"\
                "computation could be impaired."
            )

        # TODO for tomorrow.
        # for the hull, this will occurs because in the real aneurysms
        # cases the hull procedure may leave complete holes on the surface.
        # Ideally I should pass the volume compute but add a warning
        # message that the user should inspect the hull surface for the
        # hole and eventually compute the voluem better. I should then add
        # and exception here to be catched in the Aneurysm class, hull
        # computations.

        # This also brings me to the next correction I should make: how to
        # close this hull surface

        return measureVolume.GetVolume()

    @staticmethod
    def Volume2(surface_object: names.polyDataType) -> float:
        """Alternative explict computation of the volume."""

        cleaner = vtk.vtkCleanPolyData()
        cleaner.SetInputData(surface_object)
        cleaner.Update()

        triangulate = vtk.vtkTriangleFilter()
        triangulate.SetInputConnection(cleaner.GetOutputPort())
        triangulate.Update()

        surface = triangulate.GetOutput()

        if names.normals not in tools.GetPointArrays(surface):
            surface = Surface.Normals(
                          surface,
                          auto_orient_if_closed
                      )

            surface = tools.CellFieldToPointField(surface)

        cellCenters = vtk.vtkCellCenters()
        cellCenters.SetInputData(surface)
        cellCenters.Update()

        npSurface = dsa.WrapDataObject(cellCenters.GetOutput())

        normals = npSurface.GetPointData().GetArray(names.normals)
        centers = npSurface.GetPoints()

        cellAreas = dsa.VTKArray([
                        surface.GetCell(idx).ComputeArea()
                        for idx in range(surface.GetNumberOfCells())
                    ])

        cellVectors = normals*cellAreas.reshape((len(cellAreas), 1))

        return dsa.VTKArray([
                   np.dot(x, dA)
                   for x, dA in zip(centers, cellVectors)
               ]).sum()/3.0

    @staticmethod
    def Normals(
            surface_object: names.polyDataType,
            auto_orient_if_closed=False,
            flip=False
        )   -> names.polyDataType:
        """Compute (outward) surface normals.

        .. warning::
            Set 'auto_orient_if_closed' to True if the surface is closed.
        """

        normals = vtk.vtkPolyDataNormals()

        normals.SetInputData(surface_object)
        normals.ComputeCellNormalsOn()
        normals.ComputePointNormalsOn()

        if auto_orient_if_closed:
            normals.AutoOrientNormalsOn()

        else:
            normals.AutoOrientNormalsOff()

        if flip:
            normals.FlipNormalsOn()

        else:
            normals.FlipNormalsOff()

        normals.Update()

        return normals.GetOutput()

    @staticmethod
    def Curvatures(surface_object: names.polyDataType) -> names.polyDataType:
        """Compute curvature of surface.

        Uses VTK to compute the mean and Gauss curvature of a surface
        represented as a vtkPolydata. Also computes an integer array that
        identify the local shape of the surface, as presented by Ma et al.
        (2004) for intracranial aneurysms, if Kg and Km are the Gauss and mean
        curvature, we have:

        .. table:: Local shape characterization
           :widths: auto

           === ==== ================== =========
           Kg   Km  Local Shape        Int Label
           === ==== ================== =========
           > 0  > 0 Elliptical Convex  0
           > 0  < 0 Elliptical Concave 1
           > 0  = 0 Not possible       2
           < 0  > 0 Hyperbolic Convex  3
           < 0  < 0 Hyperbolic Concave 4
           < 0  = 0 Hyperbolic         5
           = 0  > 0 Cylidrical Convex  6
           = 0  < 0 Cylidrical Concave 7
           = 0  = 0 Planar             8
           === ==== ================== =========

        The name of the generated arrays are: "Mean_Curvature",
        "Gauss_Curvature", and "Local_Shape_Type".
        """
        # Compute mean curvature
        meanCurvature = vtk.vtkCurvatures()
        meanCurvature.SetInputData(surface_object)
        meanCurvature.SetCurvatureTypeToMean()
        meanCurvature.Update()

        # Compute Gaussian curvature
        gaussianCurvature = vtk.vtkCurvatures()
        gaussianCurvature.SetInputData(meanCurvature.GetOutput())
        gaussianCurvature.SetCurvatureTypeToGaussian()
        gaussianCurvature.Update()

        # Compute Min and Max curvature
        minCurvature = vtk.vtkCurvatures()
        minCurvature.SetInputData(gaussianCurvature.GetOutput())
        minCurvature.SetCurvatureTypeToMinimum()
        minCurvature.Update()

        maxCurvature = vtk.vtkCurvatures()
        maxCurvature.SetInputData(minCurvature.GetOutput())
        maxCurvature.SetCurvatureTypeToMaximum()
        maxCurvature.Update()

        cellCurvatures = vtk.vtkPointDataToCellData()
        cellCurvatures.SetInputData(maxCurvature.GetOutput())
        cellCurvatures.PassPointDataOn()
        cellCurvatures.Update()

        npCurvatures   = dsa.WrapDataObject(cellCurvatures.GetOutput())
        GaussCurvature = npCurvatures.GetCellData().GetArray('Gauss_Curvature')
        meanCurvature  = npCurvatures.GetCellData().GetArray('Mean_Curvature')

        surfaceLocalShapes = {
            'ellipticalConvex' :
                {'condition': (GaussCurvature >  0.0) & (meanCurvature >  0.0),
                 'id': 0},
            'ellipticalConcave':
                {'condition': (GaussCurvature >  0.0) & (meanCurvature <  0.0),
                 'id': 1},
            'elliptical'       : # apparently, not possible
                {'condition': (GaussCurvature >  0.0) & (meanCurvature == 0.0),
                 'id': 2},
            'hyperbolicConvex' :
                {'condition': (GaussCurvature <  0.0) & (meanCurvature >  0.0),
                 'id': 3},
            'hyperboliConcave' :
                {'condition': (GaussCurvature <  0.0) & (meanCurvature <  0.0),
                 'id': 4},
            'hyperbolic'       :
                {'condition': (GaussCurvature <  0.0) & (meanCurvature == 0.0),
                 'id': 5},
            'cylindricConvex'  :
                {'condition': (GaussCurvature == 0.0) & (meanCurvature >  0.0),
                 'id': 6},
            'cylindricConcave' :
                {'condition': (GaussCurvature == 0.0) & (meanCurvature <  0.0),
                 'id': 7},
            'planar'           :
                {'condition': (GaussCurvature == 0.0) & (meanCurvature == 0.0),
                 'id': 8}
        }

        LocalShapeArray = zeros(shape=len(meanCurvature), dtype=int)

        for shape in surfaceLocalShapes.values():
            LocalShapeArray += where(shape.get('condition'), shape.get('id'), 0)

        npCurvatures.CellData.append(LocalShapeArray, 'Local_Shape_Type')

        return npCurvatures.VTKObject

    def GetSurfaceObject(self):
        """Return the surface vtkPolyData object."""
        return self._surface_object

    def GetSurfaceArea(self):
        """Return the surface total area."""
        return Surface.Area(self._surface_object)

    def GetSurfaceVolume(self):
        """Return the surface total enclosed volume."""
        return Surface.Volume(self._surface_object)

    def GetCellArrays(self):
        """Return the names of arrays for a vtkPolyData."""
        return tools.GetCellArrays(self._surface_object)

    def GetPointArrays(self):
        """Return the names of point arrays for a vtkPolyData."""
        return tools.GetPointArrays(self._surface_object)
