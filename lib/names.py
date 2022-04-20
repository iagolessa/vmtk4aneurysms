"""Definitions of names and other variables."""

import vtk
from vtk.numpy_interface import dataset_adapter as dsa

# Attribute array names
polyDataType = vtk.vtkCommonDataModelPython.vtkPolyData
unstructuredGridType = vtk.vtkCommonDataModelPython.vtkUnstructuredGrid
multiBlockType = vtk.vtkCommonDataModelPython.vtkMultiBlockDataSet
foamReaderType = vtk.vtkIOParallelPython.vtkPOpenFOAMReader

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
ngrad = '_ngradient'
sgrad = '_sgradient'
min_ = '_minimum'
max_ = '_maximum'

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
