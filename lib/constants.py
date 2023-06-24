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

oneHundred = 100.0

degToRad = pi/180.0
millimeterToMeterFactor = 1000.0
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

lowWSS = 1.5 # Pa

# Structures holding constants

# Dictionary holding the IDs of each wall type
# See docstring of func vascular_operations.WallTypeClassification
IaWallTypes = {"RegularWall": 0,
               "AtheroscleroticWall": 1,
               "RedWall": 2}
