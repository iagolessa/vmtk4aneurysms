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
from numpy import array
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
CellEntityIdsArrayName     = "CellEntityIds"
AneurysmalRegionArrayName  = "AneurysmalRegionArray"
GeodesicDistanceArrayName  = "GeodesicDistance"
EuclideanDistanceArrayName = "EuclideanDistance"
WallTypeArrayName          = "WallType"
DistanceToNeckArrayName    = 'DistanceToNeck'
ThicknessArrayName         = 'Thickness'
VascularRadiusArrayName    = "MaximumInscribedSphereRadius"
AbnormalFactorArrayName    = "AbnormalFactorArray"
ElasticityArrayName        = "E"
ParentArteryArrayName      = "ParentArteryContourArray"
GaussCurvatureArrayName    = "Gauss_Curvature"
MeanCurvatureArrayName     = "Mean_Curvature"
MaxCurvatureArrayName      = "Maximum_Curvature"
MinCurvatureArrayName      = "Minimum_Curvature"
SeamScalarsArrayName       = "SeamScalars"
TorsionArrayName           = "Torsion"
CurvatureArrayName         = "Curvature"

SqrGaussCurvatureArrayName = "SqrGaussCurvature"
SqrMeanCurvatureArrayName  = "SqrMeanCurvature"

# VMTK's array names
vmtkCenterlineIdsArrayName  = "CenterlineIds"
vmtkGroupIdsArrayName       = "GroupIds"
vmtkBlankingArrayName       = "Blanking"
vmtkLengthArrayName         = "Length"
vmtkAbscissasArrayName      = "Abscissas"
vmtkFrenetTangentArrayName  = "FrenetTangent"
vmtkFrenetNormalArrayName   = "FrenetNormal"
vmtkFrenetBinormalArrayName = "FrenetBinormal"
vmtkParallelTransportArrayName = "ParallelTransportNormals"
vmtkReferenceSystemsNormalArrayName = "Normal"

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

# List containing the main hemodynamic wall parameters (HWP)
hwpList = [TAWSS, OSI,
           RRT, AFI,
           GON, WSSPI,
           WSSTG, transWSS,
           peakSystoleWSS,
           lowDiastoleWSS]

areaAvgGaussCurvature = "GAA"
l2NormGaussCurvature  = "GLN"
areaAvgMeanCurvature  = "MAA"
l2NormMeanCurvature   = "MLN"

curvMetricsList = [areaAvgGaussCurvature,
                   areaAvgMeanCurvature,
                   l2NormGaussCurvature,
                   l2NormMeanCurvature]

# Normalized profiles data as published by Hoi et al. (2010) and Ford et al.
# (2005) (see docstring of function hemodynamics.generateBloodFlorProfile)
# The coordinates were extracted directly from the data plots provided by the
# study
def GetHoiICAProfile():
    HoiICAProfile = [
        [-0.06, 0.55],
        [-0.04, 0.62],
        [-0.02, 0.84],
        [-0.01, 1.00],
        [0.00, 1.12],
        [0.01, 1.25],
        [0.02, 1.41],
        [0.03, 1.54],
        [0.04, 1.61],
        [0.06, 1.67],
        [0.07, 1.64],
        [0.08, 1.59],
        [0.09, 1.54],
        [0.11, 1.51],
        [0.12, 1.52],
        [0.13, 1.54],
        [0.16, 1.60],
        [0.19, 1.60],
        [0.20, 1.57],
        [0.21, 1.53],
        [0.22, 1.49],
        [0.27, 1.21],
        [0.28, 1.15],
        [0.29, 1.08],
        [0.30, 1.02],
        [0.33, 0.96],
        [0.34, 0.98],
        [0.36, 1.04],
        [0.39, 1.07],
        [0.41, 1.05],
        [0.45, 0.98],
        [0.47, 0.93],
        [0.50, 0.84],
        [0.52, 0.80],
        [0.54, 0.78],
        [0.57, 0.76],
        [0.71, 0.70],
        [0.74, 0.68],
        [0.76, 0.66],
        [0.81, 0.60],
        [0.86, 0.56],
        [0.89, 0.55]
    ]

    npHoiICAProfile = array(HoiICAProfile)
    npHoiICAProfile[:, 0] += -npHoiICAProfile[0,0]

    return npHoiICAProfile

def GetFordICAProfile():

    FordICAProfile = [
        [-0.06000, 0.67371],
        [-0.03229, 0.79655],
        [-0.01973, 0.91293],
        [-0.00807, 1.05733],
        [0.00179, 1.17586],
        [0.01166, 1.32457],
        [0.02063, 1.43664],
        [0.02780, 1.53362],
        [0.03587, 1.60259],
        [0.04753, 1.65216],
        [0.06188, 1.61121],
        [0.07175, 1.54224],
        [0.07623, 1.49914],
        [0.08520, 1.41940],
        [0.10045, 1.31595],
        [0.11390, 1.25129],
        [0.14350, 1.19526],
        [0.16413, 1.20603],
        [0.18834, 1.22112],
        [0.20807, 1.19310],
        [0.22960, 1.12845],
        [0.24305, 1.07241],
        [0.25919, 0.99914],
        [0.28161, 0.93879],
        [0.29865, 0.96681],
        [0.31390, 1.01422],
        [0.33274, 1.06595],
        [0.35157, 1.08534],
        [0.37309, 1.08534],
        [0.39372, 1.06810],
        [0.42332, 1.02284],
        [0.44843, 0.97112],
        [0.46996, 0.93233],
        [0.49417, 0.89569],
        [0.52197, 0.86767],
        [0.54260, 0.84828],
        [0.58565, 0.82457],
        [0.62422, 0.80517],
        [0.65740, 0.78578],
        [0.70404, 0.75560],
        [0.74529, 0.71897],
        [0.78027, 0.68879],
        [0.82000, 0.67371]
    ]
    npFordICAProfile = array(FordICAProfile)
    npFordICAProfile[:, 0] += -npFordICAProfile[0,0]

    return npFordICAProfile

def GetFordVAProfile():
    FordVAProfile = [
        [-0.064,0.636],
        [-0.049,0.665],
        [-0.031,0.785],
        [-0.022,0.905],
        [-0.011,1.079],
        [0.004,1.281],
        [0.010,1.384],
        [0.018,1.508],
        [0.022,1.616],
        [0.027,1.674],
        [0.032,1.727],
        [0.036,1.756],
        [0.040,1.785],
        [0.048,1.769],
        [0.054,1.731],
        [0.065,1.624],
        [0.070,1.541],
        [0.086,1.405],
        [0.095,1.339],
        [0.107,1.273],
        [0.117,1.244],
        [0.125,1.244],
        [0.136,1.260],
        [0.161,1.318],
        [0.179,1.339],
        [0.193,1.326],
        [0.202,1.293],
        [0.219,1.231],
        [0.233,1.149],
        [0.244,1.074],
        [0.260,0.971],
        [0.266,0.934],
        [0.273,0.901],
        [0.282,0.884],
        [0.288,0.909],
        [0.295,0.938],
        [0.309,1.004],
        [0.328,1.074],
        [0.364,1.128],
        [0.394,1.095],
        [0.410,1.062],
        [0.424,1.033],
        [0.435,1.008],
        [0.448,0.975],
        [0.468,0.926],
        [0.484,0.901],
        [0.516,0.868],
        [0.544,0.839],
        [0.576,0.822],
        [0.606,0.802],
        [0.644,0.785],
        [0.685,0.756],
        [0.721,0.715],
        [0.758,0.669],
        [0.787,0.645],
        [0.826,0.632]
    ]

    npFordVAProfile = array(FordVAProfile)
    npFordVAProfile[:, 0] += -npFordVAProfile[0,0]

    return npFordVAProfile
