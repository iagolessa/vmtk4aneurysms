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

from vtk.numpy_interface import dataset_adapter as dsa

from numpy import (
        array, sqrt,
        arcsin, arccos, arctan,
        sin, cos, inf
    )

from vmtk4aneurysms.lib import names
from vmtk4aneurysms.lib import constants as const
from vmtk4aneurysms.lib import polydatatools as tools
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

def _sphere_curvatures(
        radius: float
    )   -> tuple:
    """Compute the Gauss and mean curvature of a point of a sphere."""

    K = const.one/(radius**const.two)
    H = const.one/radius

    return K, H

def _ellipsoid_curvatures(
        point: tuple,
        xsemiaxis: float,
        ysemiaxis: float,
        zsemiaxis: float
    )   -> tuple:
    """Compute the Gauss and mean curvature of a point of an ellipsoid."""

    # Equations found in https://mathworld.wolfram.com/Ellipsoid.html
    x, y, z = point
    a, b, c = xsemiaxis, ysemiaxis, zsemiaxis

    # Transform coordinates to parametric ones
    argU = (y/b)/(x/a) if x != const.zero else inf

    u = arctan(argU) # belongs to [-pi/2, pi/2]

    # Update in case u is negative to bring it to [0, 2pi[ interval
    u = const.two*const.pi + u if u < 0 else u

    v = arccos(z/c) # belongs to [0, pi] -> ok with numpy doc.

    sqrCosU = cos(u)**2
    sqrCosV = cos(v)**2
    sqrSinU = sin(u)**2
    sqrSinV = sin(v)**2

    sqrA = a**2
    sqrB = b**2
    sqrC = c**2

    denominator = (
                      sqrA*sqrB*sqrCosV
                      +
                      sqrC*sqrSinV*(
                          sqrB*sqrCosU
                          +
                          sqrA*sqrSinU
                      )
                  )

    # Gaussian curvature
    K = sqrA*sqrB*sqrC/(denominator**2)

    # Mean curvature
    H = (
            a*b*c*(
                3.0*(sqrA + sqrB)
                +
                2.0*sqrC
                +
                (sqrA + sqrB - 2.0*sqrC)*cos(2.0*v)
                -
                2.0*(sqrA - sqrB)*cos(2*u)*sqrSinV
            )
        )/(
            8*(denominator**(3.0/2.0))
        )

    return K, H

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

    # The equation is based on the integration over a rotation of an ellipse
    # around the major axis.
    return factor*(_ellipse_integral(e, theta2) - _ellipse_integral(e, theta1))

class HemisphereAneurysm:
    """Model of a saccular cerebral in the form of a hemisphere."""

    def __init__(self, radius, surface_resolution=100):
        """Initiates hemisphere aneurysm model."""

        # Set label for compatibility
        self._label = "hemisphere"

        self._radius = radius
        self._center = (0, 0, 0)

        self._surface = geo.GenerateSphereSurface(
                            radius,
                            max_phi=90,
                            resolution=surface_resolution
                        )

        self._surface_area = const.oneHalf*SphereSurfaceArea(radius)
        self._volume = const.oneHalf*SphereVolume(radius)

    # Public interface
    def GetLabel(self) -> str:
        return self._label

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
        return {"MAA": const.one/self._radius,
                "GAA": const.one/(self._radius**2),
                "MLN": sqrt(const.one/(const.eight*const.pi)),
                "GLN": const.oneHalf,
                "HGLN": const.zero}

class HemiEllipsoidAneurysm:
    """Model of a saccular cerebral in the form of a hemi-ellipsoid."""

    def __init__(self, a, c, surface_resolution=100):
        """Initiates hemi-ellipsoid aneurysm model."""

        # Set label for compatibility
        self._label = "hemi-ellipsoid"

        self._minoraxis, self._majoraxis = sorted([a,c])
        self._center = (0, 0, 0)

        self._surface = geo.GenerateEllipsoid(
                            self._minoraxis,
                            self._majoraxis,
                            max_phi=90,
                            resolution=surface_resolution
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
    def GetLabel(self) -> str:
        return self._label

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
        # USe numpy interface to compute array fields
        npSurface = dsa.WrapDataObject(self.GetSurface())

        curvatures = dsa.VTKArray([
                        list(
                              _ellipsoid_curvatures(
                                  point,
                                  self._minoraxis,
                                  self._minoraxis,
                                  self._majoraxis
                            )
                        ) for point in npSurface.Points
                    ])

        npSurface.PointData.append(
            curvatures[:, 0],
            names.GaussCurvatureArrayName
        )

        npSurface.PointData.append(
            curvatures[:, 1],
            names.MeanCurvatureArrayName
        )

        # Convert point data to cell data
        surface = tools.PointFieldToCellField(npSurface.VTKObject)

        # Compute squared values
        npSurface = dsa.WrapDataObject(surface)

        arrGaussCurv = npSurface.CellData.GetArray(names.GaussCurvatureArrayName)
        arrMeanCurv  = npSurface.CellData.GetArray(names.MeanCurvatureArrayName)

        nameSqrGaussCurv = "Squared_Gauss_Curvature"
        nameSqrMeanCurv  = "Squared_Mean_Curvature"

        npSurface.CellData.append(
            arrGaussCurv**2,
            nameSqrGaussCurv
        )

        npSurface.CellData.append(
            arrMeanCurv**2,
            nameSqrMeanCurv
        )

        curvatureSurface = npSurface.VTKObject
        surfaceArea = self.GetAneurysmSurfaceArea()

        GAA = pmath.SurfaceAverage(
                    curvatureSurface,
                    names.GaussCurvatureArrayName
                )

        MAA = pmath.SurfaceAverage(
                    curvatureSurface,
                    names.MeanCurvatureArrayName
                )

        surfIntSqrGaussCurv = surfaceArea*pmath.SurfaceAverage(
                                curvatureSurface,
                                nameSqrGaussCurv
                            )

        surfIntSqrMeanCurv = surfaceArea*pmath.SurfaceAverage(
                                curvatureSurface,
                                nameSqrMeanCurv
                            )

        GLN = sqrt(surfaceArea*surfIntSqrGaussCurv)/(4*const.pi)
        MLN = sqrt(surfIntSqrMeanCurv)/(4*const.pi)

        # Computing the hyperbolic L2-norm
        hyperbolicPatches = tools.ClipWithScalar(
                                curvatureSurface,
                                names.GaussCurvatureArrayName,
                                const.zero
                            )

        hyperbolicArea    = geo.Surface.Area(hyperbolicPatches)

        # Check if there is any hyperbolic areas
        if hyperbolicArea > 0.0:
            surfIntHypSqrGaussCurv = hyperbolicArea*pmath.SurfaceAverage(
                                                        hyperbolicPatches,
                                                        nameSqrGaussCurv
                                                    )

            HGLN = sqrt(hyperbolicArea*surfIntHypSqrGaussCurv)/(4*const.pi)

        else:
            HGLN = 0.0

        return {"MAA": MAA,
                "GAA": GAA,
                "MLN": MLN,
                "GLN": GLN,
                "HGLN": HGLN}

class ThreeFourthEllipsoidAneurysm:
    """Model of a saccular cerebral in the form of a three-fourth-ellipsoid."""

    def __init__(self, a, c, surface_resolution=100):
        """Initiates three-fourth-ellipsoid aneurysm model."""

        # Set label for compatibility
        self._label = "three-fourth-ellipsoid"

        self._minoraxis, self._majoraxis = sorted([a,c])
        self._center = (0, 0, 0)

        self._surface = geo.GenerateEllipsoid(
                            self._minoraxis,
                            self._majoraxis,
                            max_phi=120,
                            resolution=surface_resolution
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
    def GetLabel(self) -> str:
        return self._label

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
        # USe numpy interface to compute array fields
        npSurface = dsa.WrapDataObject(self.GetSurface())

        curvatures = dsa.VTKArray([
                        list(
                              _ellipsoid_curvatures(
                                  point,
                                  self._minoraxis,
                                  self._minoraxis,
                                  self._majoraxis
                            )
                        ) for point in npSurface.Points
                    ])

        npSurface.PointData.append(
            curvatures[:, 0],
            names.GaussCurvatureArrayName
        )

        npSurface.PointData.append(
            curvatures[:, 1],
            names.MeanCurvatureArrayName
        )

        # Convert point data to cell data
        surface = tools.PointFieldToCellField(npSurface.VTKObject)

        # Compute squared values
        npSurface = dsa.WrapDataObject(surface)

        arrGaussCurv = npSurface.CellData.GetArray(names.GaussCurvatureArrayName)
        arrMeanCurv  = npSurface.CellData.GetArray(names.MeanCurvatureArrayName)

        nameSqrGaussCurv = "Squared_Gauss_Curvature"
        nameSqrMeanCurv  = "Squared_Mean_Curvature"

        npSurface.CellData.append(
            arrGaussCurv**2,
            nameSqrGaussCurv
        )

        npSurface.CellData.append(
            arrMeanCurv**2,
            nameSqrMeanCurv
        )

        curvatureSurface = npSurface.VTKObject
        surfaceArea = self.GetAneurysmSurfaceArea()

        GAA = pmath.SurfaceAverage(
                    curvatureSurface,
                    names.GaussCurvatureArrayName
                )

        MAA = pmath.SurfaceAverage(
                    curvatureSurface,
                    names.MeanCurvatureArrayName
                )

        surfIntSqrGaussCurv = surfaceArea*pmath.SurfaceAverage(
                                curvatureSurface,
                                nameSqrGaussCurv
                            )

        surfIntSqrMeanCurv = surfaceArea*pmath.SurfaceAverage(
                                curvatureSurface,
                                nameSqrMeanCurv
                            )

        GLN = sqrt(surfaceArea*surfIntSqrGaussCurv)/(4*const.pi)
        MLN = sqrt(surfIntSqrMeanCurv)/(4*const.pi)

        # Computing the hyperbolic L2-norm
        hyperbolicPatches = tools.ClipWithScalar(
                                curvatureSurface,
                                names.GaussCurvatureArrayName,
                                const.zero
                            )

        hyperbolicArea    = geo.Surface.Area(hyperbolicPatches)

        # Check if there is any hyperbolic areas
        if hyperbolicArea > 0.0:
            surfIntHypSqrGaussCurv = hyperbolicArea*pmath.SurfaceAverage(
                                                        hyperbolicPatches,
                                                        nameSqrGaussCurv
                                                    )

            HGLN = sqrt(hyperbolicArea*surfIntHypSqrGaussCurv)/(4*const.pi)

        else:
            HGLN = 0.0

        return {"MAA": MAA,
                "GAA": GAA,
                "MLN": MLN,
                "GLN": GLN,
                "HGLN": HGLN}
