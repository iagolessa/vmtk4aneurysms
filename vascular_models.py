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

from numpy import array, sqrt, arcsin, sin
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

def _ellipsoid_cap_volume(
        a: float,
        b: float,
        c: float,
        cap_height: float
    )   -> float:
    """Return the volume of the cap of an ellipsoid given its axes.

    The cap height must be defined along the axis of the c lenght.
    """

    # Volume calculated based on the volume of an ellipsoid cap
    # https://keisan.casio.com/keisan/image/volume%20of%20an%20ellipsoidal%20cap.pdf

    # c is along the z axis, which is the case here
    a, b, c = sorted([a, b, c])

    h = cap_height

    return (
               (const.pi*a*b*(h**const.two)
           )/(
               const.three*c**const.two)
           )*(const.three*c - h)

def _ellipse_integral(
        e: float,
        theta: float
    )   -> float:

    aux = sqrt(const.one - (e*sin(theta))**2)

    return arcsin(e*sin(theta)) + e*sin(theta)*aux

def EllipsoidVolume(
        a: float,
        b: float,
        c: float
    )   -> float:
    """Return the volume of an ellipsoid given its axes."""

    # Get correct ordering (
    a, b, c = sorted([a, b, c])

    return (const.four/const.three)*const.pi*a*b*c

def EllipsoidSurfaceArea(
        a: float,
        c: float,
        theta1: float=-const.pi/2,
        theta2: float=const.pi/2
    )   -> float:
    """Return the surface area of an ellipsoid given its axes and angle limits.

    Considering only a ellipsoid that is a prolate spheroid, i.e. when the
    third axis (c) is longer than the other two that are equal, hence c > a =
    b, computes its surface area, by default. Optionally, the latitudinal angle
    limits can be passed to return the area of a sectior of such an ellipsoide.
    The complete ellipsoid corresponds to theta_1 = - pi/2 and theta_2 = pi/2
    (defined in radians).
    """

    # Get correct ordering (c > a here)
    minoraxis, majoraxis = sorted([a,c])

    # Compute eccentricity
    axisRatio = minoraxis/majoraxis
    e = sqrt(const.one - axisRatio**const.two)

    factor = const.pi*minoraxis*majoraxis/e

    return factor*(_ellipse_integral(e, theta2) - _ellipse_integral(e, theta1))

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

    def __init__(self, a, c, center):
        """Initiates hemi-ellipsoid aneurysm model."""

        self._minoraxis, self._majoraxis = sorted([a,c])
        self._center = center

        self._surface = geo.GenerateHemiEllipsoid(
                            self._minoraxis,
                            self._majoraxis,
                            center
                        )

        self._surface_area = EllipsoidSurfaceArea(
                                 self._minoraxis,
                                 self._majoraxis,
                                 theta1=0,
                                 theta2=const.pi/2
                             )

        self._volume = const.oneHalf*EllipsoidVolume(
                                         self._minoraxis,
                                         self._minoraxis,
                                         self._majoraxis,
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

class ThreeFourthEllipsoidAneurysm:
    """Model of a saccular cerebral in the form of a three-fourth-ellipsoid."""

    def __init__(self, a, c, center):
        """Initiates three-fourth-ellipsoid aneurysm model."""

        self._minoraxis, self._majoraxis = sorted([a,c])
        self._center = center

        self._surface = geo.GenerateThreeFourthEllipsoid(
                            self._minoraxis,
                            self._majoraxis,
                            center
                        )

        self._surface_area = EllipsoidSurfaceArea(
                                 self._minoraxis,
                                 self._majoraxis,
                                 theta1=-const.pi/2,
                                 theta2=const.pi/6
                             )

        # Height of cap in this case is half a semiaxis
        height = const.oneHalf*self._majoraxis

        completeVolume = EllipsoidVolume(
                             self._minoraxis,
                             self._minoraxis,
                             self._majoraxis
                         )

        capVolume = _ellipsoid_cap_volume(
                        self._minoraxis,
                        self._minoraxis,
                        self._majoraxis,
                        height
                    )

        self._volume = completeVolume - capVolume

        # Radius of the base (section over half a semi major axis)
        self._base_radius = self._minoraxis*sqrt(
                                const.one
                                -
                                (height/self._majoraxis)**const.two
                            )

    # Public interface
    def GetDomeTipPoint(self) -> tuple:
        return tuple(
                   array(self._center)
                   +
                   array(
                       [0, 0, 1.5*self._majoraxis]
                   )
               )

    def GetSurface(self) -> names.polyDataType:
        return self._surface

    def GetHullSurface(self) -> names.polyDataType:
        return self.GetSurface()

    def GetAneurysmSurfaceArea(self) -> float:
        return self._surface_area

    def GetOstiumArea(self) -> float:
        return const.pi*(self._base_radius**const.two)

    def GetAneurysmVolume(self) -> float:
        return self._volume

    def GetHullSurfaceArea(self) -> float:
        return self.GetAneurysmSurfaceArea()

    def GetHullVolume(self) -> float:
        return self.GetAneurysmVolume()

    def GetNeckDiameter(self) -> float:
        return const.two*self._base_radius

    def GetMaximumNormalHeight(self) -> float:
        return 1.5*self._majoraxis

    def GetMaximumDiameter(self) -> float:
        return const.two*self._minoraxis

    def GetAspectRatio(self) -> float:
        return const.three/sqrt(const.three)

    def GetBottleneckFactor(self) -> float:
        return const.two/sqrt(const.three)

    def GetConicityParameter(self) -> float:
        return const.one/const.six

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
