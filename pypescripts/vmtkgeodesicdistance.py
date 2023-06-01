#! /usr/bin/env python

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

import sys
import vtk

from vmtk import vtkvmtk
from vmtk import vmtkrenderer
from vmtk import pypes

from vmtk4aneurysms.lib import polydatatools as tools
from vmtk4aneurysms.lib import polydatageometry as geo

vmtkgeodesicdistance = 'vmtkGeodesicDistance'

class vmtkGeodesicDistance(pypes.pypeScript):

    # Constructor
    def __init__(self):
        pypes.pypeScript.__init__(self)

        self.Surface      = None
        self.GeodesicDistanceArrayName = "GeodesicDistance"

        self.SetScriptName('vmtkgeodesicdistance')
        self.SetScriptDoc(
            """compute the geodesic distance from a closed contour draw on
            a surface. The inner portion of the closed contour is marked with
            negative values."""
        )

        self.SetInputMembers([
            ['Surface','i', 'vtkPolyData', 1, '',
                'the input surface', 'vmtksurfacereader'],

            ['GeodesicDistanceArrayName','distancearrayname', 'str', 1, '',
                'name of the distance array']
        ])

        self.SetOutputMembers([
            ['Surface','o','vtkPolyData',1,'',
             'the input surface with geodesic distances', 'vmtksurfacewriter'],
        ])


    def Execute(self):
        if self.Surface == None:
            self.PrintError("Error: no Surface.")

        # Operate on a triangulated surface and map final result to orignal
        # surface
        cleaner = vtk.vtkCleanPolyData()
        cleaner.SetInputData(self.Surface)
        cleaner.Update()

        # Reference to original surface
        polygonalSurface = cleaner.GetOutput()

        # Will operate on the triangulated one
        triangulate = vtk.vtkTriangleFilter()
        triangulate.SetInputData(cleaner.GetOutput())
        triangulate.Update()

        self.Surface = triangulate.GetOutput()

        # Get ids of contour
        getContour = tools.SelectContourPointsIds()
        getContour.Surface = self.Surface
        getContour.Execute()

        # seedPoints = getContour.ContourPoints

        # Compute the geodesic distance  from the approximate neck contour
        self.Surface = geo.SurfaceGeodesicDistanceToContour(
                           self.Surface,
                           getContour.ContourIds,
                           gdistance_array_name=self.GeodesicDistanceArrayName
                       )

        # Map final geodesic distance field to original surface
        self.Surface = tools.ProjectPointArray(
                           polygonalSurface,
                           self.Surface,
                           self.GeodesicDistanceArrayName
                       )

if __name__ == '__main__':
    main = pypes.pypeMain()
    main.Arguments = sys.argv
    main.Execute()
