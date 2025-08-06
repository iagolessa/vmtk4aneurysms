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

"""Classes of vascular models and related ones."""

import vtk
import numpy as np
from scipy.signal import find_peaks_cwt

from vmtk import vtkvmtk
from vmtk import vmtkscripts
from vtkmodules.numpy_interface import dataset_adapter as dsa

from vmtk4aneurysms.lib import names
from vmtk4aneurysms.lib import constants as const
from vmtk4aneurysms.lib import polydatatools as tools
from vmtk4aneurysms.lib import polydatageometry as geo

class Branch():
    """Branch segment representation."""

    def __init__(self, centerline, surface):
        """Initialize from branch vtkPolyData."""

        self._branch_centerline = centerline
        self._branch_surface = surface

        self._branch_length = self._compute_length()
        self._branch_area = geo.Surface.Area(self._branch_surface)

    def GetCenterline(self):
        """Return branch vtkPolyData."""
        return self._branch_centerline

    def _compute_length(self):
        # Get arrays
        pointArrays = tools.GetPointArrays(self._branch_centerline)
        abscissasArray = "Abscissas"

        if abscissasArray not in pointArrays:
            attributes = vmtkscripts.vmtkCenterlineAttributes()
            attributes.Centerlines = self._branch_centerline
            attributes.Execute()

            self._branch_centerline = attributes.Centerlines

        # Compute length by Abscissas array
        distanceRange = self._branch_centerline.GetPointData().GetArray(
                            abscissasArray
                        ).GetRange()

        return max(distanceRange) - min(distanceRange)

    def GetLength(self):
        """Return the length of branch."""

        return self._branch_length

    def GetSurface(self):
        """Return branch surface."""

        return self._branch_surface

    def GetSurfaceArea(self):
        """Return the branch surface area."""

        return self._branch_area

class Bifurcation:
    """Model of a bifurcation of a vascular centerline network.

    Based on the work by Piccinelli et al. (2009), who proposed a framework to
    identify and quantitatively analyze bifurcations in vascular models, this
    class implements some of their geometric definitions.  Its inputs are the
    bifurcation reference system (vtkPolyData) containing its center and
    normal, and the bifurcation vectors. Both can be computed with
    'vmtkbifurcationreferencesystem' and 'vmtkbifurcationvectors' scripts of
    the VMTK library.
    """

    def __init__(
            self,
            referenceSystem,
            vectors
        ):

        self._bif_reference_system = referenceSystem
        self._bif_vectors_object = vectors

        # Characterize bifurcation plane
        self._bif_center = referenceSystem.GetPoints().GetPoint(0)

        self._bif_plane_normal = referenceSystem.GetPointData().GetArray(
                                     'Normal'
                                 ).GetTuple(0)


        self._bif_plane_normal = referenceSystem.GetPointData().GetArray(
                                     'UpNormal'
                                 ).GetTuple(0)

        # Characterize branches directions in bifurcation
        npBifVectors = dsa.WrapDataObject(vectors)
        bifPointData = npBifVectors.GetPointData()

        self._nbranches = vectors.GetPoints().GetNumberOfPoints()
        self._branches_ids = bifPointData.GetArray(
                                 "GroupIds"
                             )

        # Collect points and vectors (in plane and out plane)
        self._bif_points  = npBifVectors.GetPoints()

        self._bif_vectors = bifPointData.GetArray(
                                "BifurcationVectors"
                            )

        # In plane vector are the bif vectors projected in the bif plane
        self._inplane_bif_vectors = bifPointData.GetArray(
                                        "InPlaneBifurcationVectors"
                                    )
        self._outplane_bif_vectors = bifPointData.GetArray(
                                        "OutOfPlaneBifurcationVectors"
                                    )

        # In plane vector are the bif vectors projected in the bif plane
        self._inplane_bif_angles = bifPointData.GetArray(
                                        "InPlaneBifurcationVectorAngles"
                                    )

        self._outplane_bif_angles = bifPointData.GetArray(
                                        "OutOfPlaneBifurcationVectorAngles"
                                    )

    def GetBifurcationReferenceSystem(self):
        """Get the bifurcation reference system."""

        return self._bif_reference_system

    def GetBifurcationVectorsObject(self):
        """Get the bifurcation vector object."""

        return self._bif_vectors_object

    def GetBifurcationPlaneAngles(self):
        """Get the bifurcation vectors."""

        return self._inplane_bif_angles

    def GetBifurcationVectors(self):
        """Get the bifurcation vectors."""

        return self._bif_vectors

    def GetBifurcationPlane(self):
        """Get the bifurcation plane."""

        bifPlane = vtk.vtkPlane()
        bifPlane.SetOrigin(self._bif_center)
        bifPlane.SetNormal(self._bif_plane_normal)

        return bifPlane

    def GetDaugtherBranchesAngle(self) -> tuple:
        """Return the angle between daughter branches (for two-branched
        bifurcations)."""

        # Get the largest values of the GroupIds array, that identifies
        # the branches departing from a bifurcation
        daughterIds = np.where(
                          self._branches_ids != min(self._branches_ids)
                      )[0]

        if self._nbranches == int(const.three):

            return const.radToDeg*vtk.vtkMath.AngleBetweenVectors(
                       self._inplane_bif_vectors[daughterIds][0],
                       self._inplane_bif_vectors[daughterIds][1]
                   ),


        else:
            angles = [const.radToDeg*vtk.vtkMath.AngleBetweenVectors(
                            self._inplane_bif_vectors[daughterIds][i],
                            self._inplane_bif_vectors[daughterIds][i + 1]
                        ) for i in range(self._nbranches - 2)]

            return tuple(angles)

class VascularCenterline:
    """A class to encapsulate operations and properties related to vascular
    centerlines.

    Attributes:
        data (names.polyDataType): The vtkPolyData object representing the
                                   centerline.
    """

    def __init__(
            self,
            centerline_data: names.polyDataType
        ):
        """Initializes a Centerline object with vtkPolyData.

        Args:
            centerline_data (names.polyDataType): A vtkPolyData object
                                                  representing the centerline.
        Raises:
            TypeError: If centerline_data is not a vtkPolyData object.
        """
        if not isinstance(centerline_data, vtk.vtkPolyData):
            raise TypeError("centerline_data must be a vtkPolyData object.")

        self._centerline_data = centerline_data

        # Compute centerline geometry arrays
        self.compute_geometry()

        # Split centerline into branches
        self.compute_branching()

        # Compute the centerlines' bifurcations ref. systems
        self._bifurcation_ref_systems = self.compute_reference_systems()

        # Collect bifurcations and their geometry
        self._bifurcations = []
        self._nbranching_points = int(const.zero)
        self._compute_bifurcations_geometry()

        # Split centerline into its constituents
        self._individual_centerlines = self.split_centerline_object()

        self._bifurcating_centerlines = {}

    @classmethod
    def from_file(
            cls,
            file_name: str
        ):
        """Generates a Centerline object from a centerlines file.

        Returns:
            Centerline: A new Centerline object.
        """
        return cls(tools.ReadSurface(file_name))

    @classmethod
    def from_vascular_surface(
            cls,
            surface: names.polyDataType,
            source_points: list=None,
            target_points: list=None,
            append_end_points: bool=True
        ):
        """Generates a Centerline object from a vascular surface.

        Returns:
            Centerline: A new Centerline object.
        """
        centerlines = VascularCenterline.GenerateCenterlines(
                          surface,
                          source_points,
                          target_points,
                          append_end_points
                      )

        return cls(centerlines)

    @classmethod
    def from_vascular_surface_file(
            cls,
            file_name: str,
            source_points: list=None,
            target_points: list=None,
            append_end_points: bool=True
        ):
        """Generates a Centerline object from a vascular surface file.

        Returns:
            Centerline: A new Centerline object.
        """
        return cls.from_vascular_surface(
            tools.ReadSurface(file_name),
            source_points,
            target_points,
            append_end_points
        )

    # Code of this functions was based on the vmtkcenterlines.py script of the
    # VMTK library: https://github.com/vmtk/vmtk
    @staticmethod
    def GenerateCenterlines(
            surface: names.polyDataType,
            source_points: list=None,
            target_points: list=None,
            append_end_points: bool=True
        )   -> names.polyDataType:
        """ Generates a centerline from a vascular surface.

        This is a class method that acts as an alternative constructor,
        encapsulating the logic of the original 'GenerateCenterlines' function.

        Args:
            surface (names.polyDataType): The input vascular surface.
            source_points (list, optional): List of source points.
                                            Defaults to None.
            target_points (list, optional): List of target points.
                                            Defaults to None.
            append_end_points (bool, optional): Whether to append end points to
                                                centerlines. Defaults to True.

        Returns:
            centerlines(names.polyDataType): the vascular centerline.
        """

        noEndPoints = source_points is None and target_points is None

        if noEndPoints:
            inletRefs, outletRefs = VascularSurface.ComputeOpenCenters(
                                        surface
                                    )

            source_points = list(inletRefs.keys())
            target_points = list(outletRefs.keys())

        # Internal parameters of the original GenerateCenterlines function
        capDisplacement = 0.0
        flipNormals = 0
        costFunction = '1/R'
        appendEndPointsToCenterlines = append_end_points
        # check_non_manifold = 0 # Not used in original function, removed from params

        resampling = 1
        resamplingStepLength = 0.1
        simplifyVoronoi = 0

        # Cleaning and triangulation
        surface = tools.Cleaner(surface)

        surfaceTriangulator = vtk.vtkTriangleFilter()
        surfaceTriangulator.SetInputData(surface)
        surfaceTriangulator.PassLinesOff()
        surfaceTriangulator.PassVertsOff()
        surfaceTriangulator.Update()

        # Capping the surface
        surfaceCapper = vtkvmtk.vtkvmtkCapPolyData()
        surfaceCapper.SetInputConnection(surfaceTriangulator.GetOutputPort())
        surfaceCapper.SetDisplacement(capDisplacement)
        surfaceCapper.SetInPlaneDisplacement(capDisplacement)
        surfaceCapper.Update()

        centerlineInputSurface = surfaceCapper.GetOutput()

        # Get IDs of the closest source and target points
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

        centerlineFilter.SetRadiusArrayName(names.VascularRadiusArrayName)
        centerlineFilter.SetCostFunction(costFunction)
        centerlineFilter.SetFlipNormals(flipNormals)
        centerlineFilter.SetAppendEndPointsToCenterlines(
            appendEndPointsToCenterlines
        )
        centerlineFilter.SetSimplifyVoronoi(simplifyVoronoi)

        centerlineFilter.SetCenterlineResampling(resampling)
        centerlineFilter.SetResamplingStepLength(resamplingStepLength)
        centerlineFilter.Update()

        return centerlineFilter.GetOutput()

    def compute_geometry(self):
        """Computes the sections and geometry of the centerline."""

        calcGeometry = vmtkscripts.vmtkCenterlineGeometry()
        calcGeometry.Centerlines = self._centerline_data
        calcGeometry.Execute()

        calcAttributes = vmtkscripts.vmtkCenterlineAttributes()
        calcAttributes.Centerlines = calcGeometry.Centerlines
        calcAttributes.Execute()

        self._centerline_data = calcAttributes.Centerlines

    def get_max_length(self) -> float:
        """Calculates the maximum centerline length of the vascular tree.

        Returns:
            float: The maximum length.
        """
        abscissasRange = self._centerline_data.GetPointData().GetArray(
                             names.vmtkAbscissasArrayName
                         ).GetRange()

        return abs(max(abscissasRange) - min(abscissasRange))

    def compute_branching(self):
        """Split the vasculature centerlines into branches and creates the
        centerline's branching fields.

        The branching fields are the 'GroupIds', 'CenterlineIds', 'TractIds',
        and 'Blanking' arrays, which are used to identify the branches and
        their IDs in the vascular tree. The original explanation for the
        meaning of these arrays can be found in the VMTK documentation.
        """

        branches = vmtkscripts.vmtkBranchExtractor()
        branches.Centerlines = self._centerline_data

        # Use v4a default names
        branches.RadiusArrayName = names.VascularRadiusArrayName
        branches.BlankingArrayName = names.vmtkBlankingArrayName
        branches.TractIdsArrayName = names.vmtkTractIdsArrayName
        branches.GroupIdsArrayName = names.vmtkGroupIdsArrayName
        branches.CenterlineIdsArrayName = names.vmtkCenterlineIdsArrayName
        branches.Execute()

        self._centerline_data = branches.Centerlines

    def compute_reference_systems(self) -> names.polyDataType:
        """Computes the reference systems of centerlines bifurcations.

        Returns:
            names.polyDataType: A new vtkPolyData object representing the
                                reference systems.
        """
        bifsRefSystem = vmtkscripts.vmtkBifurcationReferenceSystems()

        bifsRefSystem.Centerlines = self._centerline_data
        bifsRefSystem.RadiusArrayName = names.VascularRadiusArrayName
        bifsRefSystem.GroupIdsArrayName = names.vmtkGroupIdsArrayName
        bifsRefSystem.ReferenceSystemsNormalArrayName = \
            names.vmtkReferenceSystemsNormalArrayName
        bifsRefSystem.Execute()

        return bifsRefSystem.ReferenceSystems

    def _robust_offset_centerline(
            self,
            ref_systems: names.polyDataType,
            bif_group_id: int
        )   -> names.polyDataType:
        """Private helper method to robustly offset a centerline.
        Adaptation of the original '_robust_offset_centerline' function.

        Args:
            ref_systems (names.polyDataType): Centerline reference systems.
            bif_group_id (int): Reference bifurcation group ID.

        Returns:
            names.polyDataType: A new vtkPolyData object representing the
                                offset centerlines.
        """
        maxLength = self.get_max_length() # Calls a method of the class itself
        centerlines = tools.CopyVtkObject(self._centerline_data)

        # TODO: How to better handle this?
        # Iterate to avoid spourious errors in offsert computation
        for _ in range(0, 1000):
            offsetFilter = \
                vtkvmtk.vtkvmtkCenterlineReferenceSystemAttributesOffset()

            offsetFilter.SetInputData(centerlines)

            offsetFilter.SetReferenceSystems(ref_systems)
            offsetFilter.SetAbscissasArrayName(names.vmtkAbscissasArrayName)
            offsetFilter.SetNormalsArrayName(names.vmtkParallelTransportArrayName)
            offsetFilter.SetOffsetAbscissasArrayName(
                names.vmtkAbscissasArrayName
            )
            offsetFilter.SetOffsetNormalsArrayName(
                names.vmtkParallelTransportArrayName
            )
            offsetFilter.SetGroupIdsArrayName(names.vmtkGroupIdsArrayName)
            offsetFilter.SetCenterlineIdsArrayName(
                names.vmtkCenterlineIdsArrayName
            )
            offsetFilter.SetReferenceSystemsNormalArrayName(
                names.vmtkReferenceSystemsNormalArrayName
            )
            offsetFilter.SetReferenceSystemsGroupIdsArrayName(
                names.vmtkGroupIdsArrayName
            )
            offsetFilter.SetReferenceGroupId(bif_group_id)
            offsetFilter.Update()

            offsetCenterlines = offsetFilter.GetOutput()

            # Computes new max length
            abscissasRange = offsetCenterlines.GetPointData().GetArray(
                                 names.vmtkAbscissasArrayName
                             ).GetRange()

            newMaxLength = abs(max(abscissasRange) - min(abscissasRange))

            if not np.abs(newMaxLength - maxLength) > 1.0:
                break

        return offsetCenterlines

    def split_centerline_object(self) -> dict:
        """Splits the tree centerline into a dictionary of its components.

        The dictionary stores the centerline object, its length, and the group
        ID in the following format:
        {group_id: {"object": centerline, "length": length}}

        Returns:
            dict: A dictionary containing the centerline components.
        """
        centerlines = self._centerline_data
        npCenterlines = dsa.WrapDataObject(centerlines)

        centerlineIds = list(
                            set(
                                npCenterlines.CellData.GetArray(
                                    names.vmtkCenterlineIdsArrayName
                                )
                            )
                        )

        individualCenterlines = {}

        for cl_id in centerlineIds:
            individualCenterlines[cl_id] = {}

            clPortion = tools.CellFieldToPointField(
                tools.ExtractPortion(
                    centerlines,
                    names.vmtkCenterlineIdsArrayName,
                    cl_id
                ),
                cell_field_name=names.vmtkGroupIdsArrayName
            )

            individualCenterlines[cl_id].update(
                {"object": clPortion}
            )

            individualCenterlines[cl_id].update({
                "length": max(
                    clPortion.GetCellData().GetArray(
                        names.vmtkLengthArrayName
                    ).GetRange()
                )
            })

        return individualCenterlines

    def _get_longest_centerline(self) -> names.polyDataType:

        idLongestCenterline = max(
            self._individual_centerlines,
            key=lambda idx: self._individual_centerlines[idx]["length"]
        )

        return self._individual_centerlines[idLongestCenterline]["object"]

    def _compute_bifurcations_geometry(self):
        """Collect centerline bifurcations and computes their geometry.

        Identifies the bifurcations of the input centerline model and gather
        their information in a list of bifurcations.
        """

        # Get bifuraction list
        referenceSystems = self._bifurcation_ref_systems

        # get number of branching points (bi or trifurcations)
        self._nbranching_points = referenceSystems.GetPoints().GetNumberOfPoints()

        branchPtsIdsArray = referenceSystems.GetPointData().GetArray(
                               names.vmtkGroupIdsArrayName
                           )

        branchPointsIds = [branchPtsIdsArray.GetValue(index)
                           for index in range(self._nbranching_points)]

        if self._nbranching_points > const.zero:
            # Compute bifurcation
            bifVectors = vmtkscripts.vmtkBifurcationVectors()

            bifVectors.ReferenceSystems  = self._bifurcation_ref_systems
            bifVectors.Centerlines       = self._centerline_data

            bifVectors.RadiusArrayName   = names.VascularRadiusArrayName
            bifVectors.GroupIdsArrayName = names.vmtkGroupIdsArrayName
            bifVectors.TractIdsArrayName = names.vmtkTractIdsArrayName
            bifVectors.BlankingArrayName = names.vmtkBlankingArrayName
            bifVectors.CenterlineIdsArrayName = names.vmtkCenterlineIdsArrayName

            bifVectors.ReferenceSystemsNormalArrayName   = \
                    names.vmtkReferenceSystemsNormalArrayName

            bifVectors.ReferenceSystemsUpNormalArrayName = \
                    names.vmtkReferenceSystemsUpNormalArrayName

            bifVectors.NormalizeBifurcationVectors = True
            bifVectors.Execute()

            for index in branchPointsIds:
                # Filter bifurcation reference system to get only one
                # bifurcation
                self._bifurcation_ref_systems.GetPointData().SetActiveScalars(
                    names.vmtkGroupIdsArrayName
                )

                bifurcationSystem = vtk.vtkThresholdPoints()
                bifurcationSystem.SetInputData(self._bifurcation_ref_systems)
                bifurcationSystem.SetLowerThreshold(index)
                bifurcationSystem.SetUpperThreshold(index)
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

                # Collect bifurcation with Bifurcation object
                self._bifurcations.append(
                    Bifurcation(system, vectors)
                )

    def _compute_bifurcating_centerlines(self) -> dict:
        """Given a centerline of a vascular tree, builds the connectivity
        between bifurcation GroupIds and its daughter GroupIds.

        This effectively allows for finding the two individual centerlines that
        make up a particular bifurcation in the vascular tree.

        The algorithm is as follows:
        1) Identifies the bifurcation group ids
        2) Compute the TractId of the daughter branches
           that is equal to the Bif. Tract + 1
        3) Identifies the individual centerlines that has
           the bifurcation
        4) Get the group ids of the two branches.

        Returns:
            dict: A dictionary containing the bifurcating centerlines with the
            keys the bifurcation GroupId.
        """

        bifurcationsPortion = tools.ExtractPortion(
                                  self._centerline_data,
                                  names.vmtkBlankingArrayName,
                                  const.one
                              )

        bifurcationGroupIds = self.GetBifurcationGroupIds()

        for bif_id in bifurcationGroupIds:

            # Get TractId of bifurcation
            bifTract = tools.ExtractPortion(
                            bifurcationsPortion,
                            names.vmtkGroupIdsArrayName,
                            bif_id
                        )

            tractId = int(set(
                            bifTract.GetCellData().GetArray(
                                names.vmtkTractIdsArrayName
                            ).GetRange()
                      ).pop())

            clsWithBifId = {}

            # Check each centerline and get the ones
            # that have the particular bifurcation
            for cl_id, cl_data in self._individual_centerlines.items():
                cl = cl_data["object"]

                npCl = dsa.WrapDataObject(cl)

                # Get group ids on individual centerline
                clGroupIds = set(
                                npCl.GetCellData().GetArray(
                                    names.vmtkGroupIdsArrayName
                                )
                            )

                if bif_id in clGroupIds:
                    clsWithBifId.update({cl_id: cl})

            # Now, for each of the found centerlines, get the
            # 2 only that have different GroupId at the
            # branch where TractId == daughterTractId
            groupIds = {}
            for cl_id, cl in clsWithBifId.items():

                # Get the group id of the portion of the daughter tract id
                clTract = tools.ExtractPortion(
                                cl,
                                names.vmtkTractIdsArrayName,
                                tractId + 1
                            )

                # Using a dict allows to keep a single centerline for each
                # group id found.
                # Hence, this will leave only two centerlines per bifurcation,
                # as it doesnt matter which centerline is as long as its group
                # id is the correct one
                groupIds.update({
                    int(set(
                        clTract.GetCellData().GetArray(
                            names.vmtkGroupIdsArrayName
                        ).GetRange()
                    ).pop()): cl
                })

            appendPolyData = vtk.vtkAppendPolyData()

            for _, cl in groupIds.items():
                appendPolyData.AddInputData(cl)

            appendPolyData.Update()

            self._bifurcating_centerlines.update({
                bif_id: appendPolyData.GetOutput()
            })

    def ChangePropertiesOffBifurcation(
            self,
            bif_point: tuple
        ):
        """Changes centerline geometry and branching properties from a
        reference bifurcation.

        Args:
            bif_point (tuple): A point near the reference bifurcation.
        """
        # Get the reference systems of the bifurcation closest to the point
        bifGroupId = int(
                         tools.GetFieldValueAtClosestPoint(
                             self._bifurcation_ref_systems,
                             bif_point,
                             names.vmtkGroupIdsArrayName
                         )
                     )

        # Calls the private helper method for the offset
        self._centerline_data = self._robust_offset_centerline(
                                      self._bifurcation_ref_systems,
                                      bifGroupId
                                  )

        # Updates splitted centerline dictionary
        self._individual_centerlines = self.split_centerline_object()

    # Public Interface
    def GetCenterline(self):
        """Gets the centerline object with field properties.

        Returns:
            names.polyDataType: The vtkPolyData object representing the
                                centerline.
        """
        return self._centerline_data

    def GetIndividualCenterlines(self) -> dict:
        """Gets the individual centerlines of the vascular tree.

        Returns:
            dict: A dictionary containing the individual centerlines.
        """
        return self._individual_centerlines

    def GetBifurcationReferenceSystem(self) -> names.polyDataType:
        """Gets the bifurcation reference systems of the vascular tree.

        Returns:
            names.polyDataType: The vtkPolyData object representing the
                                bifurcation reference systems.
        """
        return self._bifurcation_ref_systems

    def GetLongestCenterline(self) -> names.polyDataType:
        """Gets the longest centerline of the vascular tree.

        Returns:
            names.polyDataType: The vtkPolyData object representing the
                                longest centerline.
        """
        return self._get_longest_centerline()

    def GetBifurcations(self):
        """Return the centerline's bifurcations list."""
        return self._bifurcations

    def GetNumberOfBifurcations(self):
        """Return the number of bifurcations."""
        return self._nbranching_points

    def GetBifurcationGroupIds(self) -> list:
        """Return the bifurcation group ids.

        Returns:
            list: A list of bifurcation group ids.
        """
        if not self._bifurcations:
            self._compute_bifurcations_geometry()

        return [
            bif.GetBifurcationReferenceSystem().GetPointData().GetArray(
                names.vmtkGroupIdsArrayName
            ).GetValue(0) for bif in self._bifurcations
        ]

    def GetBifurcatingCenterline(
            self,
            bifurcation_id: int
        ) -> names.polyDataType:
        """Gets a bifurcating centerline of the vascular tree.

        Returns:
            names.polyDataType: The vtkPolyData object representing the
                                bifurcating centerline.
        """
        if not self._bifurcating_centerlines:
            self._compute_bifurcating_centerlines()

        return self._bifurcating_centerlines.get(bifurcation_id, None)

    def GetPointFields(self):
        """Return the number of bifurcations."""
        return tools.GetPointArrays(self._centerline_data)

class InternalCarotidCenterline(VascularCenterline):
    """Class to represent the centerline of the internal carotid artery.

    A specialized class for handling the internal carotid artery (ICA) segment
    of a vascular centerline. Inherits from the Centerline class and provides
    methods to compute bends of the ICA segment based on its abscissas,
    torsion, and curvature.
    """
    def __init__(
            self,
            centerline_data: names.polyDataType,
            ica_bifurcation_point: tuple
        ):
        """Initializes the IcaCenterline with the given centerline data.

        Args:
            centerline_data (names.polyDataType): A vtkPolyData object
                representing the ICA segment.

            ica_bifurcation_point (tuple, optional): A point near the ICA
                bifurcation.
        """
        super().__init__(centerline_data)

        self._ica_bifurcation_point = ica_bifurcation_point

        # Updates centerline to bifurcation point
        self.ChangePropertiesOffBifurcation(ica_bifurcation_point)

    @classmethod
    def from_file(
            cls,
            surface: names.polyDataType,
            source_points: list=None,
            target_points: list=None,
            append_end_points: bool=True
        ):
        """Generates an ICA Centerline object from a vascular surface.

        Returns:
            Centerline: A new Centerline object.
        """
        centerlines = VascularCenterline.GenerateCenterlines(
                          surface,
                          source_points,
                          target_points,
                          append_end_points
                      )

        return cls(centerlines)

    def compute_ica_bends_limits(
            self,
            abscissas_peak_widths: np.ndarray=np.arange(30, 40)
        )   -> list:
        """Compute bends of internal carotid artery (ICA) segment.

        Given the vascular tree centerline containing the ICA segment with
        Abscissas defined from its bifurcation, compute the intervals of its
        bends.

        A precise definition of the bends of the internal carotide artery (ICA)
        was provided by the work:

            M. Piccinelli et al., “Geometry of the Internal Carotid Artery and
            Recurrent Patterns in Location, Orientation, and Rupture Status of
            Lateral Aneurysms: An Image-Based Computational Study”,
            Neurosurgery, vol. 68, nº 5, p. 1270–1285, maio 2011, doi:
            10.1227/NEU.0b013e31820b5242.

        which subdivides the ICA intro bends defined by torsion and curvature
        peaks. This functions implements it based on the procedure proposed in
        the paper. However, the procedure is sensitive to some arguments. For
        example, its is recommended that the passed centerline be smoothed with
        the centerline smoothing procedure in VMTK (a function that
        encapsulates the procedure with suitable arguments tuned for this
        subdivision of the ICA is in the lib/centerline.py module, see
        'SmoothCenterline').

        Also, the subdivision depends on the identification of peaks of torsion
        and curvature of the centerline that is normally a relatively noisy
        field for discretized centerline (hence the recommendation to smooth
        it). In this case the peaks are found with the
        scipy.signal.find_peaks_cwt function which depends on the 'width'
        argument passed to the wavelets functions. These widths are a list of
        possible widths between peaks of the torsion and curvature signal,
        which depends on each vascular case. These arguments can be passed to
        this function in the 'abscissas_peak_widths' argument. If None is
        passed, then the default is between 30 and 40, which is pretty
        arbitrary, but were found based on testing with the aneurisk repository
        cases and yield the best results.

        .. warning::
            It is highly recommended to smooth the centerline prior to passing
            it to this function.

        Args:
            abscissas_peak_widths (np.ndarray, optional): Abscissa peak widths.
                Defaults to np.arange(30, 40).

        Returns:
            list: A list of tuples, where each tuple represents the
                  (upstream, downstream) limits of a bend.
        """
        longestCenterline = self._get_longest_centerline()

        # Get the minimum abscissa value from the longest centerline
        # In this case it is the negative of the ICA length
        minAbscissas, _ = longestCenterline.GetPointData().GetArray(
                              names.vmtkAbscissasArrayName
                          ).GetRange()

        icaCenterline = tools.ClipWithScalar(
                            longestCenterline,
                            names.vmtkAbscissasArrayName,
                            const.zero
                        )

        npIcaCenterline = dsa.WrapDataObject(icaCenterline)

        icaTorsionField = npIcaCenterline.GetPointData().GetArray(
            names.TorsionArrayName
        )

        icaCurvatureField = npIcaCenterline.GetPointData().GetArray(
            names.CurvatureArrayName
        )

        icaAbscissasField = npIcaCenterline.GetPointData().GetArray(
            names.vmtkAbscissasArrayName
        )

        torsionPeaksIds = find_peaks_cwt(
            abs(icaTorsionField),
            widths=abscissas_peak_widths
        )

        curvaturePeaksIds = find_peaks_cwt(
            icaCurvatureField,
            widths=abscissas_peak_widths
        )

        torsionPeaksAbscissas = sorted(
            np.append(
                icaAbscissasField[torsionPeaksIds],
                [const.zero, minAbscissas]
            )
        )

        curvaturePeaksAbscissas = sorted(
            np.append(
                icaAbscissasField[curvaturePeaksIds],
                [const.zero, minAbscissas]
            )
        )

        torsionPeaksAbscissas = np.array(
            list(reversed(torsionPeaksAbscissas))
        )
        curvaturePeaksAbscissas = np.array(
            list(reversed(curvaturePeaksAbscissas))
        )

        bendLimits = []
        saveValueForNext = []

        # The original iteration was (max_abs, centre, min_abs)
        # meaning the zip needs 3 arrays.
        # curvature_peaks_abscissas[2:] ensures there will be at least 3
        # elements to iterate.
        curvatureIntervals = zip(
            curvaturePeaksAbscissas,
            curvaturePeaksAbscissas[1:],
            curvaturePeaksAbscissas[2:]
        )

        for maxAbs, centre, minAbs in curvatureIntervals:
            upstreamTorsionPeaks = torsionPeaksAbscissas[
                (torsionPeaksAbscissas <= maxAbs) &
                (torsionPeaksAbscissas >= centre)
            ]

            downstreamTorsionPeaks = torsionPeaksAbscissas[
                (torsionPeaksAbscissas <= centre) &
                (torsionPeaksAbscissas >= minAbs)
            ]

            if downstreamTorsionPeaks.size == 0:
                saveValueForNext.append(
                    upstreamTorsionPeaks[
                        abs(upstreamTorsionPeaks - centre).argmin()
                    ]
                )
                continue

            elif upstreamTorsionPeaks.size == 0:
                # This case occurs when a downstream was empty and the value
                # was saved in `saveValueForNext`.
                upstreamLimit = saveValueForNext[0]
                # Clears the saved value after use to prevent improper reuse
                saveValueForNext.clear()
                downstreamLimit = downstreamTorsionPeaks[
                    abs(downstreamTorsionPeaks - centre).argmax()
                ]

            else:
                upstreamLimit = upstreamTorsionPeaks[
                    abs(upstreamTorsionPeaks - centre).argmin()
                ]
                downstreamLimit = downstreamTorsionPeaks[
                    abs(downstreamTorsionPeaks - centre).argmax()
                ]

            bendLimits.append((upstreamLimit, downstreamLimit))

        return bendLimits

    def SplitICACenterlineIntoBends(
            centerlines: names.polyDataType,
            bif_point: tuple
        )   -> names.polyDataType:
        """Split ICA centerline into bends based on curvature and torsion."""

        # TODO: Where to smooth the centerlines?
        # smoothedCenterlines = cl.ComputeCenterlinePropertiesOffBifurcation(
        #                           centerlines,
        #                           bif_point
        #                       )

        # Compute bends limits
        bendLimits = self.compute_ica_bends_limits()

        longestCenterline = self._get_longest_centerline()
        npCenterlines = dsa.WrapDataObject(longestCenterline)

        bendIdsField = -dsa.VTKArray(
                            np.ones(
                                shape=longestCenterline.GetNumberOfPoints(),
                                dtype=int
                            )
                        )

        abscissasField = npCenterlines.PointData.GetArray(
                            names.vmtkAbscissasArrayName
                        )

        for bend_id, (min_, max_) in enumerate(bendLimits):

            bendIdsField[
                (abscissasField <= min_) &
                (abscissasField >  max_)
            ] = bend_id

        # When adding to the Numpy wrap object, it automatically
        # adds to the underlying VTK object
        npCenterlines.PointData.append(
            bendIdsField,
            names.BendsIdsFieldName
        )

        return npCenterlines.VTKObject

class VascularSurface(geo.Surface):
    """Computational model of a vascular surface.

    A vascular surface is a particular type of computational surface with a
    tubular structure containing a few number of bifurcations and open
    boundaries. It is used to represent the surface geometry of blood vessels,
    for example. The surface model must contain the open boundaries of the
    domain, i.e.  its inlets and outlets of blood flow as perpendicular open
    sections, as traditionally used for CFD. So far, it handles only
    vasculatures with a single inlet, defined as the one with largest radius.

    At construction, the class defines the following fields on the surface:

    .. table:: Default Fields on Vascular Surface
       :widths: auto

       ==================
       Field Name
       ==================
       Gauss_Curvature
       Mean_Curvature
       Local_Shape_Type
       Normals
       ==================
    """

    def __init__(self, vtk_poly_data: names.polyDataType):
        """Build vascular surface model from vtkPolyData.

        Initializes the base Surface object and provides specific functionality
        for vascular surfaces. Adds surface curvature and normal vector fields.
        """
        super().__init__(vtk_poly_data)

        # Add curvature fields to the surface
        self._surface_object = super().Curvatures(self._surface_object)

        # Set also the centers and normals of the inlets and outlets
        self._inlet_ref_systems, self._outlet_ref_systems = VascularSurface.ComputeOpenCenters(
                                                                self._surface_object
                                                            )

        self._inlet_centers = list(self._inlet_ref_systems.keys())
        self._outlet_centers = list(self._outlet_ref_systems.keys())

        self._inlet_normals = list(self._inlet_ref_systems.values())
        self._outlet_normals = list(self._outlet_ref_systems.values())

    @classmethod
    def from_file(cls, file_name):
        """Build vascular surface model from file."""

        return cls(tools.ReadSurface(file_name))

    @staticmethod
    def ComputeOpenCenters(
            vascular_surface,
            interactive: bool=False
        ):
        """Computes barycenters and outwards normals of inlets and outlets.

        Computes the geometric center and outward normal of each open boundary
        of the model. Computes two dictionaries with the centers as keys
        (tuples) and the normals as values, one for the inlets and another for
        the outlets. Both normals and centers are given as tuples:

        Dict inlet: {(x1, y1, z1): (nx1, ny1, nz1)}

        Dict outlet: {(x1, y1, z1): (nx1, ny1, nz1),
                      (x2, y2, z2): (nx2, ny2, nz2),
                      ...,
                      (xN, yN, zN): (nxN, nyN, nzN)}

        for a model with a single inlet and n outlets. The magnitude of the
        normals is the radius of the open profile. The inlet is defined as the
        open boundary with largest radius.

        The resulting dicts are stored as attributes of the instance.

        Returns:
            tuple: A tuple containing two dictionaries:
                - inletRefSystems: Dictionary with inlet centers and normals.
                - outletRefSystems: Dictionary with outlet centers and normals.
        """
        # I noticed some weird behavior of the vtkvmtkBoundaryReferenceSystems
        # when using it with a surface that has passed through the
        # vtkPolyDataNormals filter. i couldn't solve the problem, so I am
        # putting a clean-up and copy of the input surface to avoid any
        # problems for safety but keep in mind that this did not solved the
        # problem.
        # Update: the Cleaner function seemed to solve the problem

        # Clean up any arrays in surface and make copy of surface
        newSurface = tools.CopyVtkObject(vascular_surface)
        newSurface = tools.CleanupArrays(newSurface)
        newSurface = tools.Cleaner(newSurface)

        # Get complete ref systems
        pointArrays = ['Point1', 'Point2']
        boundaryRadiusArrayName = 'Radius'
        boundaryNormalsArrayName = 'BoundaryNormals'

        boundarySystems = vtkvmtk.vtkvmtkBoundaryReferenceSystems()
        boundarySystems.SetInputData(newSurface)
        boundarySystems.SetBoundaryRadiusArrayName(boundaryRadiusArrayName)
        boundarySystems.SetBoundaryNormalsArrayName(boundaryNormalsArrayName)
        boundarySystems.SetPoint1ArrayName(pointArrays[0])
        boundarySystems.SetPoint2ArrayName(pointArrays[1])
        boundarySystems.Update()

        referenceSystems = boundarySystems.GetOutput()

        npEndPoints = dsa.WrapDataObject(referenceSystems)
        endCenters = npEndPoints.GetPoints()
        radiusArray = npEndPoints.PointData.GetArray(boundaryRadiusArrayName)

        # The normal are outward the vascular domain
        outNormalsArray = npEndPoints.PointData.GetArray(boundaryNormalsArrayName)

        # Get inlet and outlet ids
        if interactive:
            capper = vtkvmtk.vtkvmtkCapPolyData()
            capper.SetInputData(newSurface)
            capper.SetDisplacement(0.0)
            capper.SetInPlaneDisplacement(0.0)
            capper.SetCellEntityIdsArrayName(names.CellEntityIdsArrayName)
            capper.Update()

            cappedSurface = capper.GetOutput()

            # Select the inlet point
            inletPickPoint = tools.PickPointSeedSelector()
            inletPickPoint.SetSurface(cappedSurface)
            inletPickPoint.InputInfo("Select a point on the inlet\n")
            inletPickPoint.Execute()

            outletPickPoint = tools.PickPointSeedSelector()
            outletPickPoint.SetSurface(cappedSurface)
            outletPickPoint.InputInfo(
                "Select the aneurysm branch outlets (if branching > 3 branches)\n"
            )
            outletPickPoint.Execute()

            inletSeeds = inletPickPoint.PickedSeeds
            outletSeeds = outletPickPoint.PickedSeeds

            # Locate selected inlet and outlets ref. systems
            locator = vtk.vtkPointLocator()
            locator.SetDataSet(referenceSystems)
            locator.BuildLocator()

            inletIds = [locator.FindClosestPoint(inletSeeds.GetPoint(pid))
                        for pid in range(inletSeeds.GetNumberOfPoints())]

            outletIds = [locator.FindClosestPoint(outletSeeds.GetPoint(pid))
                         for pid in range(outletSeeds.GetNumberOfPoints())]

        else:
            # Select as inlet the profile with largest section area
            inletIds = [radiusArray.argmax()]
            outletIds = [idx
                         for idx in range(referenceSystems.GetNumberOfPoints())
                         if idx not in inletIds]

        _inlet_ref_systems = {
            tuple(c): tuple(r * n) for c, r, n in zip(
                                                   endCenters[inletIds],
                                                   radiusArray[inletIds],
                                                   outNormalsArray[inletIds]
                                               )
        }

        _outlet_ref_systems = {
            tuple(c): tuple(r * n) for c, r, n in zip(
                                                    endCenters[outletIds],
                                                    radiusArray[outletIds],
                                                    outNormalsArray[outletIds]
                                                )
        }

        return _inlet_ref_systems, _outlet_ref_systems

    def GetOpenBoundariesRefSystems(
            self,
            interactive: bool=False
        )   -> tuple:
        """Get barycenters and outwards normals of inlets and outlets.

        Computes the geometric center and outward normal of each open boundary
        of the model. Computes two dictionaries with the centers as keys
        (tuples) and the normals as values, one for the inlets and another for
        the outlets. Both normals and centers are given as tuples:

        Dict inlet: {(x1, y1, z1): (nx1, ny1, nz1)}

        Dict outlet: {(x1, y1, z1): (nx1, ny1, nz1),
                      (x2, y2, z2): (nx2, ny2, nz2),
                      ...,
                      (xN, yN, zN): (nxN, nyN, nzN)}

        for a model with a single inlet and n outlets. The magnitude of the
        normals is the radius of the open profile. The inlet is defined as the
        open boundary with largest radius.

        The resulting dicts are stored as attributes of the instance.

        Returns:
            tuple: A tuple containing two dictionaries:
                - inletRefSystems: Dictionary with inlet centers and normals.
                - outletRefSystems: Dictionary with outlet centers and normals.
        """
        return self._inlet_ref_systems, self._outlet_ref_systems

    def GetInletCenters(self) -> list:
        """Get inlet centers of the vascular surface.

        Returns:
            list: List of tuples representing the centers of the inlets.
        """
        return self._inlet_centers

    def GetOutletCenters(self) -> list:
        """Get outlet centers of the vascular surface.

        Returns:
            list: List of tuples representing the centers of the outlets.
        """
        return self._outlet_centers

    def GetInletNormals(self) -> list:
        """Get inlet normals of the vascular surface.

        Returns:
            list: List of tuples representing the normals of the inlets.
        """
        return self._inlet_normals

    def GetOutletNormals(self) -> list:
        """Get outlet normals of the vascular surface.

        Returns:
            list: List of tuples representing the normals of the outlets.
        """
        return self._outlet_normals

    def GetVolume(self):
        """Return the vascular surface total enclosed volume."""

        # Close the surface at the open boundaries
        capper = vtkvmtk.vtkvmtkCapPolyData()
        capper.SetInputData(self._surface_object)
        capper.SetDisplacement(0.0)
        capper.SetInPlaneDisplacement(0.0)
        capper.SetCellEntityIdsArrayName(names.CellEntityIdsArrayName)
        capper.Update()

        return geo.Surface.Volume(capper.GetOutput())

class VascularTree:
    """Representation of a vascular network tree model.

    At construction, it automatically computes the centerline and the
    morphology of the vasculature. Internally, it uses VMTK to compute all the
    geometrical features of the centerline path to fully characterize the
    vasculature topology. Furthermore, the surface curvature characterization
    is added as arrays to the surface: mean and Gaussian curvatures with an
    array defining the local curvature type.
    """

    def __init__(
            self,
            vtk_poly_data: names.polyDataType,
            centerlines_data: names.polyDataType=None
        ):
        """Initiate vascular model.

        Arguments:
        vtk_poly_data -- the vtkPolyData vascular model (default None)
        """

        vascular_surface = tools.Cleaner(vtk_poly_data)

        # Initiate vascular surface object
        self._vasc_surface_obj = VascularSurface(vascular_surface)

        # Generate centerline object
        if centerlines_data is None:
            self._vasc_centerline_obj = VascularCenterline.from_vascular_surface(
                                            vascular_surface
                                        )

        else:
            # If centerline data is provided, use it
            self._vasc_centerline_obj = VascularCenterline(
                                            centerlines_data
                                        )

        self._vascular_surface = self._vasc_surface_obj.GetSurface()
        self._centerlines = self._vasc_centerline_obj.GetCenterline()

        self._Voronoi_diagram = None

        # Branches attributes
        self._branched_surface = None
        self._branches = {}
        self._thickness_computed = False

    @classmethod
    def from_file(
            cls,
            file_name
        ):
        """Initialize vasculature object from vascular tree surface file."""

        return cls(tools.ReadSurface(file_name))

    def _compute_branched_surface(self):
        """Split the vascular tree surface into branches.

        This method literally splits the vascular tree surface into its
        constituent branches, generating a branched surface object that
        has a field that identifies the branches and bifurcations.
        """
        if self._branched_surface is None:
            # Using vtkvtmk lib class for better configuration
            # This scripts preserves all the fields on the input surface
            # for this class
            clipper = vtkvmtk.vtkvmtkPolyDataCenterlineGroupsClipper()
            clipper.SetInputData(self._vascular_surface)
            clipper.SetCenterlines(self._centerlines)
            clipper.SetCenterlineGroupIdsArrayName(names.vmtkGroupIdsArrayName)
            clipper.SetGroupIdsArrayName(names.vmtkGroupIdsArrayName)
            clipper.SetCenterlineRadiusArrayName(names.VascularRadiusArrayName)
            clipper.SetBlankingArrayName(names.vmtkBlankingArrayName)
            clipper.SetCutoffRadiusFactor(1e16)
            clipper.SetUseRadiusInformation(1)

            # TODO: clip the surface branches a scalar off the bifurcation to
            # get only the tubular structure? Before doing that I would have to
            # change the Blaking array computation of the ref. points
            # definition inside the VascularCenterline class.
            clipper.SetClipValue(0.0)
            clipper.ClipAllCenterlineGroupIdsOn()
            clipper.GenerateClippedOutputOff()
            clipper.Update()

            # Get the clipped output
            self._branched_surface = clipper.GetOutput()

    def _compute_local_wlr(self, diameter):
        if diameter > const.VesselLargeDiameter:
            return const.WlrLarge

        elif diameter < const.VesselMediumDiameter:
            return const.WlrMedium

        else:
            # Linear threshold
            deltaWlr = const.WlrLarge - const.WlrMedium
            deltaDiameter = const.VesselLargeDiameter - const.VesselMediumDiameter
            angCoeff = deltaWlr/deltaDiameter

            return const.WlrMedium + angCoeff*(diameter - const.VesselMediumDiameter)

    def _compute_vascular_thickness_internal(
            self,
            set_uniform_wlr: bool=False,
            uniform_wlr_value: float=const.WlrMedium
        ):
        """
        Internal method to compute thickness of the vascular surface
        and add it as a point array. Modifies self._vascular_surface in place.
        """
        # Compute distance to centerlines
        distanceToCenterlines = vtkvmtk.vtkvmtkPolyDataDistanceToCenterlines()
        distanceToCenterlines.SetInputData(self._vascular_surface)
        distanceToCenterlines.SetCenterlines(self._centerlines)

        distanceToCenterlines.SetUseRadiusInformation(True)
        distanceToCenterlines.SetEvaluateCenterlineRadius(True)
        distanceToCenterlines.SetEvaluateTubeFunction(False)
        distanceToCenterlines.SetProjectPointArrays(False)

        distanceToCenterlines.SetDistanceToCenterlinesArrayName(
            names.ThicknessArrayName
        )

        distanceToCenterlines.SetCenterlineRadiusArrayName(
            names.VascularRadiusArrayName
        )

        distanceToCenterlines.Update()

        # Update the internal _vascular_surface with the output of this filter
        self._vascular_surface = distanceToCenterlines.GetOutput()

        # Use numpy interface with VTK
        npSurface = dsa.WrapDataObject(self._vascular_surface)

        distanceArray = npSurface.GetPointData().GetArray(
                            names.ThicknessArrayName
                        )

        radiusArray   = npSurface.GetPointData().GetArray(
                            names.VascularRadiusArrayName
                        )

        # This portion evaluates if distance is much higher
        # than the actual radius array
        # This necessarily will need some smoothing

        # Set high and low threshold factors
        # Are they arbitrary?
        highRadiusThresholdFactor = 1.4
        lowRadiusThresholdFactor  = 0.9

        npMaxRadiusLim = highRadiusThresholdFactor*radiusArray
        npMinRadiusLim = lowRadiusThresholdFactor*radiusArray

        distanceArray = np.where(
                            distanceArray > npMaxRadiusLim,
                            npMaxRadiusLim,
                            distanceArray
                        )

        distanceArray = np.where(
                            distanceArray < npMinRadiusLim,
                            radiusArray,
                            distanceArray
                        )

        # Smooth the distance to centerline array to avoid sudden changes of
        # thickness in certain regions
        self._vascular_surface = tools.SmoothSurfacePointField(
                                    npSurface.VTKObject,
                                    names.ThicknessArrayName,
                                    niterations=5
                                )

        npSurface = dsa.WrapDataObject(self._vascular_surface)

        # Multiply by WLR to have a prelimimar thickness array
        # I assume that the WLR is the same for medium sized arteries
        # but I can change this in a point-wise manner based on
        # the local radius array by using the algorithm contained
        # in the vmtksurfacearrayoperation script
        distanceArray = npSurface.GetPointData().GetArray(
                            names.ThicknessArrayName
                        )

        radiusArray   = npSurface.GetPointData().GetArray(
                            names.VascularRadiusArrayName
                        )

        if set_uniform_wlr:

            npSurface.PointData.append(
                dsa.VTKArray([
                    uniform_wlr_value*(2.0*r)
                    for r in distanceArray
                ]),
                names.ThicknessArrayName
            )

        else:
            # Compute are store local WLR for debug
            localWLRArray = dsa.VTKArray([
                                self._compute_local_wlr(2.0*r)
                                for r in distanceArray
                            ])

            npSurface.PointData.append(
                localWLRArray,
                "LocalWLR"
            )

            # Compute thickness array and replace the thickness array
            # (originally stored as the distance to neck array with a new one)
            npSurface.PointData.append(
                localWLRArray*(2.0*distanceArray),
                names.ThicknessArrayName
            )

        # Remove the radius array from the vascular surface
        vascular_surface = npSurface.VTKObject
        vascular_surface.GetPointData().RemoveArray(
            names.VascularRadiusArrayName
        )

        # Updates vascular surface OBJECT
        self._vasc_surface_obj = VascularSurface(vascular_surface)
        self._vascular_surface = self._vasc_surface_obj.GetSurface()

    def ComputeVascularWallThickness(
            self,
            set_uniform_wlr: bool=False,
            uniform_wlr_value: float=const.WlrMedium
        ):
        """Computes the vascular wall thickness and adds it as a point data
        array to the internal vascular surface.

        Given input surface with the radius array, computes the thickness by
        multiplying by the wall-to-lumen ration. This method modifies the
        internal _vascular_surface attribute in-place. It will only compute the
        thickness once unless explicitly reset.

        Arguments:
            set_uniform_wlr (bool): If True, use a uniform wall-to-lumen ratio.

            uniform_wlr_value (float): The uniform wall-to-lumen ratio to use.
        """
        if not self._thickness_computed:
            self._compute_vascular_thickness_internal(
                set_uniform_wlr=set_uniform_wlr,
                uniform_wlr_value=uniform_wlr_value
            )

            self._thickness_computed = True

        else:
            print("Thickness field already computed. Skipping re-computation.")

    def ComputeVoronoiDiagram(self) -> names.polyDataType:
        """Compute Voronoi diagram of a vascular surface."""

        if self._Voronoi_diagram is None:
            voronoiDiagram = vmtkscripts.vmtkDelaunayVoronoi()
            voronoiDiagram.Surface = self._vascular_surface
            voronoiDiagram.CheckNonManifold = True
            voronoiDiagram.Execute()

            self._Voronoi_diagram = voronoiDiagram.Surface

        return self._Voronoi_diagram

    def GetBranches(self) -> dict:
        """Split vascular tree into branch objects.

        Given the vasculature centerlines, slits it into its constituent
        branches. Generates a list of branch objects.

        Returns:
            dict: A dict of {GroupId: Branch...} objects representing the
                branches of the vascular model.
        """
        if not self._branches:

            # Compute branched surface if not already done
            self._compute_branched_surface()

            # Extract only the branches portion
            # (the blanking array separates the branches from the bifurcations
            # branches are identified with the value 1)
            branchesId = 0
            centerlineBranches = tools.ExtractPortion(
                                     self._centerlines,
                                     names.vmtkBlankingArrayName,
                                     branchesId
                                 )

            # Get only the branch group ids
            npBranches = dsa.WrapDataObject(centerlineBranches)
            branchesIds = set(
                              npBranches.GetCellData().GetArray(
                                  names.vmtkGroupIdsArrayName
                              )
                          )

            for branchId in branchesIds:
                # try:
                branch = tools.ExtractPortion(
                             centerlineBranches,
                             names.vmtkGroupIdsArrayName,
                             branchId
                         )

                # The surface branches do not have a "bifurcation patch"
                # Hence can use directly the branches id obtained above
                surfaceBranch = tools.ExtractPortion(
                                    self._branched_surface,
                                    names.vmtkGroupIdsArrayName,
                                    branchId
                                )

                self._branches.update({
                    branchId: Branch(
                                  branch,
                                  surfaceBranch
                              )
                })

                # except(ValueError):
                #     pass

        return self._branches

    # TODO: add automatic computation of vascular thickness here?
    def GetCenterlinesObject(self):
        """Return the vasculature's centerlines."""
        return self._vasc_centerline_obj

    def GetVascularSurfaceObject(self):
        """Return the vasculature's centerlines."""
        return self._vasc_surface_obj

    def GetVascularSurface(self):
        """Return the vascular surface."""
        return self._vasc_surface_obj.GetSurface()

    # def GetBranchedSurface(self):
    #     """Return the vascular surface."""
    #     return self._branched_surface

    def GetCenterlines(self):
        """Return the vasculature's centerlines."""
        return self._vasc_centerline_obj.GetCenterline()

    def GetInletCenters(self):
        """Return the inlet center."""
        return self._vasc_surface_obj.GetInletCenters()

    def GetOutletCenters(self):
        """Return the outlet center(s)."""
        return self._vasc_surface_obj.GetOutletCenters()

    def GetBifurcations(self):
        """Return the vascular model bifurcations list."""
        return self._vasc_centerline_obj.GetBifurcations()

    def GetNumberOfBifurcations(self):
        """Return the number of bifurcations."""
        return self._vasc_centerline_obj.GetNumberOfBifurcations()
