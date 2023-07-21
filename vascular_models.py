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

from numpy import array, sqrt, arcsin
from vmtk4aneurysms.lib import names
from vmtk4aneurysms.lib import constants as const
from vmtk4aneurysms.lib import polydatageometry as geo
from vmtk4aneurysms.lib import polydatamath as pmath

def SphereVolume(radius):
    """Return the volume of a sphere given its radius."""

    return (const.four/const.three)*const.pi*(radius**const.three)

def SphereSurfaceArea(radius):
    """Return the surface area of a sphere given its radius."""

    return const.four*const.pi*(radius**const.two)

def EllipsoidVolume(
        minoraxis: float,
        majoraxis: float,
    )   -> float:
    """Return the volume of an ellipsoid given its axes."""

    return (const.four/const.three)*const.pi*(minoraxis**2)*majoraxis

def EllipsoidSurfaceArea(
        minoraxis: float,
        majoraxis: float,
    )   -> float:
    """Return the surface area of an ellipsoid given its axes."""

    # Surface computation is more complicated
    # I found on wikipedia
    axisRatio = minoraxis/majoraxis
    e = sqrt(const.one - axisRatio**const.two)

    factor = majoraxis/(minoraxis*e)

    return const.two*const.pi*(minoraxis**const.two)*(const.one + factor*arcsin(e))

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
        return const.oneHalf

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

class HemiEllipsoidAneurysm:
    """Model of a saccular cerebral in the form of a hemi-ellipsoid."""

    def __init__(self, minoraxis, majoraxis, center):
        """Initiates hemi-ellipsoid aneurysm model."""

        self._minoraxis = minoraxis
        self._majoraxis = majoraxis
        self._center = center

        self._surface = geo.GenerateHemiEllipsoid(
                            minoraxis,
                            majoraxis,
                            center
                        )

        self._surface_area = const.oneHalf*EllipsoidSurfaceArea(
                                               minoraxis,
                                               majoraxis
                                           )

        self._volume = const.oneHalf*EllipsoidVolume(
                                         minoraxis,
                                         majoraxis
                                     )

    # Public interface
    def GetDomeTipPoint(self) -> tuple:
        return tuple(array(self._center) + array([0,0,self._majoraxis]))

    def GetSurface(self) -> names.polyDataType:
        return self._surface

    def GetHullSurface(self) -> names.polyDataType:
        return self.GetSurface()

    def GetAneurysmSurfaceArea(self) -> float:
        return self._surface_area

    def GetOstiumArea(self) -> float:
        return const.pi*(self._minoraxis**const.two)

    def GetAneurysmVolume(self) -> float:
        return self._volume

    def GetHullSurfaceArea(self) -> float:
        return self.GetAneurysmSurfaceArea()

    def GetHullVolume(self) -> float:
        return self.GetAneurysmVolume()

    def GetNeckDiameter(self) -> float:
        return const.two*self._minoraxis

    def GetMaximumNormalHeight(self) -> float:
        return self._majoraxis

    def GetMaximumDiameter(self) -> float:
        return const.two*self._minoraxis

    def GetAspectRatio(self) -> float:
        return const.one

    def GetBottleneckFactor(self) -> float:
        return const.one

    def GetConicityParameter(self) -> float:
        return const.oneHalf

    def GetNonSphericityIndex(self) -> float:
        return const.one - pmath.SphericityIndex(
                               self.GetAneurysmSurfaceArea(),
                               self.GetAneurysmVolume()
                           )

    def GetEllipticityIndex(self) -> float:
        return const.one - pmath.SphericityIndex(
                               self.GetHullSurfaceArea(),
                               self.GetHullVolume()
                           )

    def GetUndulationIndex(self) -> float:
        return const.zero

    def GetCurvatureMetrics(self) -> dict:
        pass
        # TODO: Only possible to compute it numerically?
        # return {"MAA": const.oneHalf/self.radius,
        #         "GAA": const.oneHalf/(self.radius**2),
        #         "MLN": sqrt(const.one/(const.eight*const.pi)),
        #         "GLN": sqrt(const.oneHalf),
        #         "HGLN": const.zero}
