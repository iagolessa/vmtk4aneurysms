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
    vtkImageData,
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

vmtkBranchAngularMetricArrayName = "AngularMetric"
vmtkBranchStretchedMappingArrayName = "StretchedMapping"
vmtkBranchAbscissasMetricArrayName = "AbscissasMetric"

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

    The study provides the mean velocity of blood flow, whereas the code here
    provides the flow volumetric rate profile computed with the averaged
    area of the ICA of the Aneurisk repository.

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
            [0.0377,1.0547e-06],
            [0.0554,1.4360e-06],
            [0.0664,1.9077e-06],
            [0.0709,2.2667e-06],
            [0.0753,2.7384e-06],
            [0.0797,3.2324e-06],
            [0.0842,3.7932e-06],
            [0.0908,4.2426e-06],
            [0.0930,4.5126e-06],
            [0.0997,4.8479e-06],
            [0.0975,5.2069e-06],
            [0.1019,5.6118e-06],
            [0.1063,6.0168e-06],
            [0.1107,6.9128e-06],
            [0.1174,7.2955e-06],
            [0.1262,7.6323e-06],
            [0.1351,7.8785e-06],
            [0.1506,7.5210e-06],
            [0.1595,7.1383e-06],
            [0.1639,6.7571e-06],
            [0.1750,6.3981e-06],
            [0.1816,6.0836e-06],
            [0.1905,5.7691e-06],
            [0.1971,5.4101e-06],
            [0.2104,5.0956e-06],
            [0.2237,4.8494e-06],
            [0.2458,4.6699e-06],
            [0.2680,4.5349e-06],
            [0.2857,4.2871e-06],
            [0.2990,3.9964e-06],
            [0.3123,3.7264e-06],
            [0.3256,3.5469e-06],
            [0.3455,3.5469e-06],
            [0.3676,3.7486e-06],
            [0.3898,3.8836e-06],
            [0.4141,3.9059e-06],
            [0.4341,3.7486e-06],
            [0.4496,3.5024e-06],
            [0.4673,3.3006e-06],
            [0.4828,3.1197e-06],
            [0.4983,2.8512e-06],
            [0.5205,2.6034e-06],
            [0.5470,2.3794e-06],
            [0.5758,2.1777e-06],
            [0.6002,2.1332e-06],
            [0.6290,2.0427e-06],
            [0.6533,1.9077e-06],
            [0.6799,1.7742e-06],
            [0.7065,1.6392e-06],
            [0.7330,1.5932e-06],
            [0.7596,1.4375e-06],
            [0.7973,1.4375e-06],
            [0.8261,1.3247e-06],
            [0.8526,1.1007e-06],
            [0.8881,9.8797e-07]
        ]

    else:
        # Normalized ICA profile at activity
        PoulinICAProfile = [
            [0.0360,8.4408e-07],
            [0.0444,1.4196e-06],
            [0.0503,2.2415e-06],
            [0.0539,2.7577e-06],
            [0.0599,3.6137e-06],
            [0.0623,4.1655e-06],
            [0.0647,4.9755e-06],
            [0.0683,5.5036e-06],
            [0.0695,6.3135e-06],
            [0.0707,6.8891e-06],
            [0.0755,7.7569e-06],
            [0.0779,8.2627e-06],
            [0.0839,9.0950e-06],
            [0.0875,9.6231e-06],
            [0.1043,1.0645e-05],
            [0.1223,9.5652e-06],
            [0.1271,9.0371e-06],
            [0.1391,8.1915e-06],
            [0.1474,7.6753e-06],
            [0.1666,6.8297e-06],
            [0.1798,6.4070e-06],
            [0.2038,5.5629e-06],
            [0.2158,5.0926e-06],
            [0.2361,4.2486e-06],
            [0.2457,3.7427e-06],
            [0.2613,2.9105e-06],
            [0.2733,2.3587e-06],
            [0.3021,1.7475e-06],
            [0.3392,2.0531e-06],
            [0.3740,1.7252e-06],
            [0.4136,1.0562e-06],
            [0.4423,7.3875e-07]
        ]

    npPoulinICAProfile = array(PoulinICAProfile)
    npPoulinICAProfile[:, 0] += -npPoulinICAProfile[0,0]

    return npPoulinICAProfile
