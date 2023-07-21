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

"""Collection of vascular models."""

from numpy import array, sqrt
from vmtk4aneurysms.lib import names
from vmtk4aneurysms.lib import constants as const
from vmtk4aneurysms.lib import polydatageometry as geo

def SphereVolume(radius):
    """Return the volume of a sphere given its radius."""

    return (const.four/const.three)*const.pi*(radius**const.three)

def SphereSurfaceArea(radius):
    """Return the volume of a sphere given its radius."""

    return const.four*const.pi*(radius**const.two)


class HemisphereAneurysm:
    """Model of a saccular cerebral in the form of a hemisphere."""

    def __init__(self, radius, center):
        """Initiates hemisphere aneurysm model."""

        self._radius = radius
        self._center = center

        self._surface = geo.GenerateHemisphereSurface(
                            radius,
                            center
                        )

        self._surface_area = const.oneHalf*SphereSurfaceArea(radius)
        self._volume = const.oneHalf*SphereVolume(radius)

    # Public interface
    def GetDomeTipPoint(self) -> tuple:
        return tuple(array(self._center) + array([0,0,self._radius]))

    def GetSurface(self) -> names.polyDataType:
        return self._surface

    def GetHullSurface(self) -> names.polyDataType:
        return self.GetSurface()

    def GetAneurysmSurfaceArea(self) -> float:
        return self._surface_area

    def GetOstiumArea(self) -> float:
        return const.pi*(self._radius**const.two)

    def GetAneurysmVolume(self) -> float:
        return self._volume

    def GetHullSurfaceArea(self) -> float:
        return self.GetAneurysmSurfaceArea()

    def GetHullVolume(self) -> float:
        return self.GetAneurysmVolume()

    def GetNeckDiameter(self) -> float:
        return const.two*self._radius

    def GetMaximumNormalHeight(self) -> float:
        return self._radius

    def GetMaximumDiameter(self) -> float:
        return const.two*self._radius

    def GetAspectRatio(self) -> float:
        return const.oneHalf

    def GetBottleneckFactor(self) -> float:
        return const.one

    def GetConicityParameter(self) -> float:
        return const.one/const.two

    def GetNonSphericityIndex(self) -> float:
        return const.zero

    def GetEllipticityIndex(self) -> float:
        return const.zero

    def GetUndulationIndex(self) -> float:
        return const.zero

    def GetCurvatureMetrics(self) -> dict:
        return {"MAA": const.oneHalf/self.radius,
                "GAA": const.oneHalf/(self.radius**2),
                "MLN": sqrt(const.one/(const.eight*const.pi)),
                "GLN": sqrt(const.oneHalf),
                "HGLN": const.zero}
