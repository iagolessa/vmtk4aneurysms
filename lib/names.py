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

"""Definitions of names and other variables."""

import vtk
from vtk.numpy_interface import dataset_adapter as dsa

# Attribute array names
polyDataType = vtk.vtkCommonDataModelPython.vtkPolyData
unstructuredGridType = vtk.vtkCommonDataModelPython.vtkUnstructuredGrid
multiBlockType = vtk.vtkCommonDataModelPython.vtkMultiBlockDataSet
foamReaderType = vtk.vtkIOParallelPython.vtkPOpenFOAMReader
idList = vtk.vtkCommonCorePython.vtkIdList

planeType = vtk.vtkCommonDataModelPython.vtkPlane
vtkArrayType = dsa.VTKArray

# Field-type labels
scalarFieldLabel = "scalarField"
vectorFieldLabel = "vectorField"
tensor2SymmFieldLabel = "tensor2SymmField"
tensor2FieldLabel = "tensor2Field"

# Field suffixes
spaceDimensionality = 3 # 3D Euclidean space only, for now
xAxisSufx = "X"
yAxisSufx = "Y"
zAxisSufx = "Z"

avg = '_average'
mag = '_magnitude'
grad = '_gradient'
div  = '_div'
norm  = '_normalized'
ngrad = '_ngradient'
sgrad = '_sgradient'
min_ = '_minimum'
max_ = '_maximum'

# important fields names
AneurysmalRegionArrayName = "AneurysmalRegionArray"
GeodesicDistanceArrayName = "GeodesicDistance"
WallTypeArrayName         = "WallType"
DistanceToNeckArrayName   = 'DistanceToNeck'

# this one is deprecated (using the distance to neck instead)
# only here for backward compatibility
AneurysmNeckArrayName     = DistanceToNeckArrayName # old 'AneurysmNeckContourArray'

# Attribute array names
WSS = 'WSS'
OSI = 'OSI'
RRT = 'RRT'
AFI = 'AFI'
GON = 'GON'
TAWSS = 'TAWSS'
WSSPI = 'WSSPI'
WSSTG = 'WSSTG'
TAWSSG = 'TAWSSG'
transWSS = 'transWSS'

WSSmag = WSS + mag
peakSystoleWSS = 'PSWSS'
lowDiastoleWSS = 'LDWSS'

WSSSG = 'WSSSG' + avg
WSSSGmag = 'WSSSG' + mag + avg
WSSDotP = 'WSSDotP'
WSSDotQ = 'WSSDotQ'

# Local coordinate system
pHat = 'pHat'
qHat = 'qHat'
normals = 'Normals'

# Other attributes
foamWSS = 'wallShearComponent'
wallPatchName = 'wall'
