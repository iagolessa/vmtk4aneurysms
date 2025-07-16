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

"""Collection of tools that operate on or are related to centerlines."""

import vtk
import numpy as np
from morphman.manipulate_curvature import extract_single_line
from vtkmodules.numpy_interface import dataset_adapter as dsa

from vmtk import vtkvmtk
from vmtk import vmtkscripts

# Assuming these local modules are available in PYTHONPATH
# or in the same directory.
from . import names
from . import constants as const
from . import polydatageometry as geo
from . import polydatatools as tools

def SmoothCenterline(
        centerlines: names.polyDataType,
        niterations: int=100,
        smoothing_factor: float=1.25
    )   -> names.polyDataType:
    """Smooth the 3D contour of centerline."""

    smoother = vmtkscripts.vmtkCenterlineSmoothing()

    smoother.Centerlines = centerlines
    smoother.NumberOfSmoothingIterations = niterations
    smoother.SmoothingFactor = smoothing_factor
    smoother.Execute()

    return smoother.Centerlines

# This function was adapted from the Morphman library,
# available at https://github.com/KVSlab/morphMan
def GetDivergingPoint(
        centerline: names.polyDataType,
        tolerance: float
    )   -> tuple:
    """Get diverging point on centerline bifurcation.

    Args:
        centerline (vtkPolyData): centerline of a bifurcation.
        tolerance (float): tolerance.
    Returns:
        point (tuple): diverging point.
    """
    line0 = extract_single_line(centerline, 0)
    line1 = extract_single_line(centerline, 1)

    nPoints = min(line0.GetNumberOfPoints(), line1.GetNumberOfPoints())

    bifPointIndex = None
    getPoint0 = line0.GetPoints().GetPoint
    getPoint1 = line1.GetPoints().GetPoint

    for index in range(0, nPoints):
        distance = geo.Distance(getPoint0(index), getPoint1(index))

        if distance > tolerance:
            bifPointIndex = index
            break

    return getPoint0(bifPointIndex)

# This function was adapted from the Morphman library,
# available at https://github.com/KVSlab/morphMan
def ComputeClPatchEndPointParameters(
        patch_centerlines: names.polyDataType,
        patch_id: int
    )   -> tuple:
    """Compute the tangent, point and radius at the ends of a centerline.

    The result depend on the id of the centerline patch: it returns the
    end points closest to the bifurcation (or patched region) of the original
    centerline. The tangent direction is always towards the path of the
    centerline.
    """

    # Set cell to a vtk cell
    cell = vtk.vtkGenericCell()
    patch_centerlines.GetCell(patch_id, cell)

    if (patch_id == 0):
        # Then get the last point
        lastPointId       = cell.GetNumberOfPoints() - 1
        beforeLastpointId = cell.GetNumberOfPoints() - 2

        point0 = np.array(cell.GetPoints().GetPoint(lastPointId))
        point1 = np.array(cell.GetPoints().GetPoint(beforeLastpointId))

        radius0 = patch_centerlines.GetPointData().GetArray(
                      names.VascularRadiusArrayName
                  ).GetTuple1(cell.GetPointId(lastPointId))

        tan = point1 - point0
        vtk.vtkMath.Normalize(tan)

    else:
        # then get the first point
        point0 = np.array(cell.GetPoints().GetPoint(0))
        point1 = np.array(cell.GetPoints().GetPoint(1))
        radius0 = patch_centerlines.GetPointData().GetArray(
                      names.VascularRadiusArrayName
                  ).GetTuple1(cell.GetPointId(0))

        tan = point1 - point0
        vtk.vtkMath.Normalize(tan)

    return tan, point0, radius0

def ComputeTubeSurface(
        centerline: names.polyDataType,
        smooth: bool = True
    )   -> names.polyDataType:
    """Reconstruct tube surface of a given vascular surface.

    The tube surface is the maximum tubular structure inscribed in the
    vasculature.

    Arguments:
    centerline -- the centerline to compute the tube surface with the radius
    array.

    Keyword arguments:
    smooth -- to smooth tube surface (default True)
    """


    # Get bounds of model
    centerlineBounds  = centerline.GetBounds()
    radiusArrayBounds = centerline.GetPointData().GetArray(names.VascularRadiusArrayName).GetValueRange()
    maxSphereRadius   = radiusArrayBounds[1]

    # To enlarge the box: could be a fraction of maxSphereRadius
    # tests show that the whole radius is appropriate
    enlargeBoxBounds  = maxSphereRadius

    modelBounds = np.array(centerlineBounds) + \
                  np.array(const.nSpatialDimensions*[-enlargeBoxBounds, enlargeBoxBounds])

    # Extract image with tube function from model
    modeller = vtkvmtk.vtkvmtkPolyBallModeller()
    modeller.SetInputData(centerline)
    modeller.SetRadiusArrayName(names.VascularRadiusArrayName)

    # This needs to be 'on' for centerline
    modeller.UsePolyBallLineOn()

    modeller.SetModelBounds(list(modelBounds))
    modeller.SetNegateFunction(0)
    modeller.Update()

    tubeImage = modeller.GetOutput()

    # Convert tube function to surface
    tubeSurface = vmtkscripts.vmtkMarchingCubes()
    tubeSurface.Image = tubeImage
    tubeSurface.Execute()

    tube = tools.ExtractConnectedRegion(tubeSurface.Surface, 'largest')

    if smooth:
        return tools.SmoothSurface(tube)
    else:
        return tube

def ComputeVoronoiDiagram(
        vascular_surface: names.polyDataType
    )   -> names.polyDataType:
    """Compute Voronoi diagram of a vascular surface."""

    voronoiDiagram = vmtkscripts.vmtkDelaunayVoronoi()
    voronoiDiagram.Surface = vascular_surface
    voronoiDiagram.CheckNonManifold = True
    voronoiDiagram.Execute()

    return voronoiDiagram.Surface

def ComputeVoronoiEnvelope(
        voronoi_surface: names.polyDataType,
        smooth: bool=True
    )   -> names.polyDataType:
    """Compute the envelope surface of a Voronoi diagram."""

    VoronoiBounds     = voronoi_surface.GetBounds()
    radiusArrayBounds = voronoi_surface.GetPointData().GetArray(names.VascularRadiusArrayName).GetValueRange()
    maxSphereRadius   = radiusArrayBounds[1]
    enlargeBoxBounds  = maxSphereRadius

    modelBounds = np.array(VoronoiBounds) + \
                  np.array(const.nSpatialDimensions*[-enlargeBoxBounds, enlargeBoxBounds])

    # Building the envelope image function
    modeller = vtkvmtk.vtkvmtkPolyBallModeller()
    modeller.SetInputData(voronoi_surface)
    modeller.SetRadiusArrayName(names.VascularRadiusArrayName)

    # This needs to be off for surfaces
    modeller.UsePolyBallLineOff()

    modeller.SetModelBounds(list(modelBounds))
    modeller.SetNegateFunction(0)
    modeller.Update()

    envelopeImage = modeller.GetOutput()

    # Get level zero surface
    envelopeSurface = vmtkscripts.vmtkMarchingCubes()
    envelopeSurface.Image = envelopeImage
    envelopeSurface.Execute()

    envelope = tools.ExtractConnectedRegion(
                    envelopeSurface.Surface,
                    'largest'
                )

    if smooth:
        return tools.SmoothSurface(envelope)
    else:
        return envelope
