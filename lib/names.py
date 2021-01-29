import vtk

# Attribute array names
polyDataType = vtk.vtkCommonDataModelPython.vtkPolyData
multiBlockType = vtk.vtkCommonDataModelPython.vtkMultiBlockDataSet

# Field suffixes
xAxisSufx = "X"
yAxisSufx = "Y"
zAxisSufx = "Z"

avg = '_average'
mag = '_magnitude'
grad = '_gradient'
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
