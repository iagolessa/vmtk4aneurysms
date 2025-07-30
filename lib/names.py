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

from numpy import array, ndarray
from vtk import (
    vtkPolyData,
    vtkUnstructuredGrid,
    vtkMultiBlockDataSet,
    vtkIdList,
    vtkPlane
)

from vtkmodules.vtkIOParallel import vtkPOpenFOAMReader
from vtkmodules.numpy_interface import dataset_adapter as dsa

# Attribute array names
idList = vtkIdList
polyDataType = vtkPolyData
unstructuredGridType = vtkUnstructuredGrid
multiBlockType = vtkMultiBlockDataSet
planeType = vtkPlane

foamReaderType = vtkPOpenFOAMReader
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
vec = '_vector'
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
SacRegionsArrayName        = "SacRegions"
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
LocalShapeTypeArrayName    = "Local_Shape_Type"
SeamScalarsArrayName       = "SeamScalars"
TorsionArrayName           = "Torsion"
CurvatureArrayName         = "Curvature"
BendsIdsFieldName          = "BendIds"

SqrGaussCurvatureArrayName = "SqrGaussCurvature"
SqrMeanCurvatureArrayName  = "SqrMeanCurvature"

# VMTK's array names
vmtkCenterlineIdsArrayName  = "CenterlineIds"
vmtkGroupIdsArrayName       = "GroupIds"
vmtkBlankingArrayName       = "Blanking"
vmtkTractIdsArrayName       = "TractIds"
vmtkLengthArrayName         = "Length"
vmtkAbscissasArrayName      = "Abscissas"
vmtkFrenetTangentArrayName  = "FrenetTangent"
vmtkFrenetNormalArrayName   = "FrenetNormal"
vmtkFrenetBinormalArrayName = "FrenetBinormal"
vmtkParallelTransportArrayName = "ParallelTransportNormals"
vmtkReferenceSystemsNormalArrayName = "Normal"
vmtkReferenceSystemsUpNormalArrayName = "UpNormal"

# this one is deprecated (using the distance to neck instead)
# only here for backward compatibility
AneurysmNeckArrayName     = DistanceToNeckArrayName # old 'AneurysmNeckContourArray'

# Attribute array names
# This is the WSS vector field
WSS = 'WSS' + vec

# Derived
OSI = 'OSI'
RRT = 'RRT'
AFI = 'AFI'
GON = 'GON'
TAWSS = 'TAWSS'
WSSPI = 'WSSPI'
WSSTG = 'WSSTG'
TAWSSG = 'TAWSSG'
transWSS = 'transWSS'
LowShearArea = 'LSA'

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
listHWP = [TAWSS, OSI,
           RRT, AFI,
           GON, WSSPI,
           WSSTG, transWSS,
           peakSystoleWSS,
           lowDiastoleWSS]

# Aneurysm metrics names
iaMetricSurfaceArea        = "AneurysmSurfaceArea"
iaMetricOstiumArea         = "OstiumArea"
iaMetricVolume             = "AneurysmVolume"
iaMetricSurfaceArea        = "AneurysmSurfaceArea"
iaMetricHullSurfaceArea    = "AneurysmHullSurfaceArea"
iaMetricHullVolume         = "AneurysmHullVolume"
iaMetricNeckDiameter       = "Dn"
iaMetricMaxNormalHeight    = "Hnmax"
iaMetricMaxDiameter        = "Dmax"
iaMetricAspectRatio        = "AR"
iaMetricBottleneckFactor   = "BF"
iaMetricConicityParameter  = "CP"
iaMetricNonsphericityIndex = "NSI"
iaMetricEllipticityIndex   = "EI"
iaMetricUndulationIndex    = "UI"

areaAvgGaussCurvature = "GAA"
l2NormGaussCurvature  = "GLN"
areaAvgMeanCurvature  = "MAA"
l2NormMeanCurvature   = "MLN"

curvMetricsList = [
    areaAvgGaussCurvature,
    areaAvgMeanCurvature,
    l2NormGaussCurvature,
    l2NormMeanCurvature
]

# Normalized profiles data as published by Hoi et al. (2010) and Ford et al.
# (2005) (see docstring of function hemodynamics.generateBloodFlorProfile)
# The coordinates were extracted directly from the data plots provided by the
# study
def GetHoiICAProfile():
    """Returns the normalized ICA profile as published by Hoi et al. (2010).

    The coordinates were extracted directly from the data plots provided by the
    study (see docstring of function hemodynamics.GenerateBloodFlowRateProfile).
    """

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
    """Returns the normalized ICA profile as published by Ford et al. (2005).

    The coordinates were extracted directly from the data plots provided by the
    study (see docstring of function hemodynamics.GenerateBloodFlowRateProfile).
    """

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
    """Returns the normalized VA profile as published by Ford et al. (2005).

    The coordinates were extracted directly from the data plots provided by the
    study (see docstring of function hemodynamics.GenerateBloodFlowRateProfile).
    """

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

def GetPoulinICAProfile(
        activity_conditions: bool=True
    )   -> ndarray:
    """Returns the normalized ICA profile as published by Poulin et al. (1999).

    The coordinates were extracted directly from the data plots provided by the
    study:

        M. J. Poulin, R. J. Syed, and P. A. Robbins, “Assessments of flow by
        transcranial Doppler ultrasound in the middle cerebral artery during
        exercise in humans,” Journal of Applied Physiology, vol. 86, no. 5, pp.
        1632–1637, May 1999, doi: 10.1152/jappl.1999.86.5.1632.

    Arguments:
        rest: If True, returns the profile at rest, otherwise returns the
            profile during exercise.

    Returns:
        np.ndarray: The normalized ICA profile as a 2D numpy array, where the
        first column is the normalized time and the second column is the
        normalized flow rate.
    """

    if not activity_conditions:

        # Normalized ICA profile at rest
        PoulinICAProfile = [
            [0.0377,1.1612e-06],
            [0.0554,1.5809e-06],
            [0.0664,2.1002e-06],
            [0.0709,2.4954e-06],
            [0.0753,3.0147e-06],
            [0.0797,3.5586e-06],
            [0.0842,4.1759e-06],
            [0.0908,4.6707e-06],
            [0.0930,4.9680e-06],
            [0.0997,5.3371e-06],
            [0.0975,5.7323e-06],
            [0.1019,6.1781e-06],
            [0.1063,7.1417e-06],
            [0.1107,7.6104e-06],
            [0.1174,8.0317e-06],
            [0.1262,8.4024e-06],
            [0.1351,8.6735e-06],
            [0.1506,8.2799e-06],
            [0.1595,7.8586e-06],
            [0.1639,7.4389e-06],
            [0.1750,7.0437e-06],
            [0.1816,6.6974e-06],
            [0.1905,6.3512e-06],
            [0.1971,5.9560e-06],
            [0.2104,5.6098e-06],
            [0.2237,5.3387e-06],
            [0.2458,5.1411e-06],
            [0.2680,4.9925e-06],
            [0.2857,4.7197e-06],
            [0.2990,4.3996e-06],
            [0.3123,4.1024e-06],
            [0.3256,3.9048e-06],
            [0.3455,3.9048e-06],
            [0.3676,4.1269e-06],
            [0.3898,4.2755e-06],
            [0.4141,4.3000e-06],
            [0.4341,4.1269e-06],
            [0.4496,3.8558e-06],
            [0.4673,3.6337e-06],
            [0.4828,3.4345e-06],
            [0.4983,3.1389e-06],
            [0.5205,2.8661e-06],
            [0.5470,2.6195e-06],
            [0.5758,2.3974e-06],
            [0.6002,2.3484e-06],
            [0.6290,2.2488e-06],
            [0.6533,2.1002e-06],
            [0.6799,1.9532e-06],
            [0.7065,1.8046e-06],
            [0.7330,1.7540e-06],
            [0.7596,1.5825e-06],
            [0.7973,1.5825e-06],
            [0.8261,1.4584e-06],
            [0.8526,1.2118e-06],
            [0.8881,1.0877e-06]
        ]

    else:
        # Normalized ICA profile at activity
        PoulinICAProfile = [
            [0.0360,9.2925e-07],
            [0.0444,1.5629e-06],
            [0.0503,2.4677e-06],
            [0.0539,3.0360e-06],
            [0.0599,3.9783e-06],
            [0.0623,4.5858e-06],
            [0.0647,5.4775e-06],
            [0.0683,6.0589e-06],
            [0.0695,6.9506e-06],
            [0.0707,7.5842e-06],
            [0.0755,8.5396e-06],
            [0.0779,9.0965e-06],
            [0.0839,1.0013e-05],
            [0.0875,1.0594e-05],
            [0.1043,1.1719e-05],
            [0.1223,1.0530e-05],
            [0.1271,9.9490e-06],
            [0.1391,9.0181e-06],
            [0.1474,8.4498e-06],
            [0.1666,7.5189e-06],
            [0.1798,7.0535e-06],
            [0.2038,6.1242e-06],
            [0.2158,5.6065e-06],
            [0.2361,4.6773e-06],
            [0.2457,4.1204e-06],
            [0.2613,3.2042e-06],
            [0.2733,2.5967e-06],
            [0.3021,1.9238e-06],
            [0.3392,2.2602e-06],
            [0.3740,1.8993e-06],
            [0.4136,1.1628e-06],
            [0.4423,7.7410e-07]
        ]

    npPoulinICAProfile = array(PoulinICAProfile)
    npPoulinICAProfile[:, 0] += -npPoulinICAProfile[0,0]

    return npPoulinICAProfile
