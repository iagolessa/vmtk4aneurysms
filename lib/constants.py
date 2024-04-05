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

"""Mathematical and physical constants."""

import math
import vtk

# Constants
pi = vtk.vtkMath.Pi()

zero  = 0.0
one   = 1.0
two   = 2.0
three = 3.0
four  = 4.0
five  = 5.0
six   = 6.0
seven = 7.0
eight = 8.0
nine  = 9.0
ten   = 10.0

oneHalf = one/two
oneHundred = 100.0

degToRad = pi/180.0
radToDeg = 180.0/pi
millimeterToMeterFactor = 1000.0

mLMinToM3Sec = 1.0e-6/60.0 # m3/s
mmHgToPa     = 133.3223    # Pa/mmHg

nSpatialDimensions = int(three)

# Name of the field defined on a vascular surface that identifies the aneurysm
# with zero values and one the rest of the surface. The value 0.5, hence,
# identifies the aneurysm neck path (see 'NeckIsoValue')
# Here only for backward compatibility, as deprecated
NeckIsoValue = 0.0 # old 0.5

# Wall-to-lumen ration values
WlrMedium = 0.07
WlrLarge  = 0.088
VesselMediumDiameter = 2.0
VesselLargeDiameter  = 3.0

# Physical "constants"
lowWSS = 1.5 # Pa
bloodDensity = 1056.0 # kg/m3

# Structures holding constants

# Dictionary holding the IDs of each wall type
# See docstring of func vascular_operations.WallTypeClassification
IaWallTypes = {"RegularWall": 0,
               "AtheroscleroticWall": 1,
               "RedWall": 2}

# Average blood flow rates as measured by
# L. Zarrinkoob, K. Ambarki, A. Wåhlin, R. Birgander, A. Eklund, e J. Malm,
# “Blood flow distribution in cerebral arteries”, Journal of Cerebral Blood
# Flow and Metabolism, vol. 35, p. 648–654, 2015, doi: 10.1038/jcbfm.2014.241.
# Here only for the internal carotid artery and the basilar artery

# For the ICA, Ford and Hoi results are in agreement with measurements by
# Zarrinkoob. Hoi's study (didn't measured the VA flow rate). Also, the meand
# and sd. of Zarro=inkoobs subjects are within the mean and sd. of Ford and
# Hoi's subjects.
BfrAvgICAOlderAdults = 236.0*mLMinToM3Sec
BfrAvgICAYoungAdults = 276.0*mLMinToM3Sec
BfrAvgBAYoungAdults  = 162.0*mLMinToM3Sec
BfrAvgBAOlderAdults  = 128.0*mLMinToM3Sec
