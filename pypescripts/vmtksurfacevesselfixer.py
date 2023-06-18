#!/usr/bin/env python

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


#NEEDS TO STAY AS TOP LEVEL MODULE FOR Py2-3 COMPATIBILITY
from __future__ import absolute_import
import vtk
import sys

from vmtk import vtkvmtk
from vmtk import vmtkscripts
from vmtk import vmtkrenderer
from vmtk import pypes

vmtksurfacevesselfixer = 'vmtkSurfaceVesselFixer'

class vmtkSurfaceVesselFixer(pypes.pypeScript):

    def __init__(self):

        pypes.pypeScript.__init__(self)

        self.Surface = None
        self.vmtkRenderer = None
        self.OwnRenderer  = 0
        self.Remesh = False
        self.Smooth = True
        self.Clip = True

        self.Actor = None
        self.ContourWidget = None
        self.Interpolator  = None
        self.ContourScalarsArrayName = 'ContourScalars'
        self.InsideValue = 0.0
        self.FillValue   = 1.0

        self.SetScriptName('vmtksurfacevesselfixer')
        self.SetScriptDoc("Function to interactively fix a surface of a"
                          "vessel segment with holes and other weird"
                          "artifacts. The function works internally with"
                          "the region drawing script and clipper with"
                          "array."
                          "Two forms of tools are available: one that directly"
                          "fix (=clip and cap) the surface by selecting a closed"
                          "region; and the other asks for the user to draw the"
                          "region to be removed by interactively filling it with"
                          "closed contours (this option is mainly designed to"
                          "fix joined regions of the vessels).")

        self.SetInputMembers([
            ['Surface', 'i', 'vtkPolyData', 1, '',
                'the input surface', 'vmtksurfacereader'],

            ['vmtkRenderer', 'renderer', 'vmtkRenderer', 1, '',
                'external renderer'],

            ['Remesh' , 'remesh', 'bool', 1, '',
                'to apply remeshing procedure after fixing it'],

            ['Smooth' , 'smooth','bool',1,'',
                'if surface must be smoothed before fixing'],

            ['Clip'   , 'clip'  ,'bool',1,'',
                'to clip surface with a box before fixing it'],
        ])

        self.SetOutputMembers([
            ['Surface', 'o', 'vtkPolyData', 1, '',
                'the output surface', 'vmtksurfacewriter']
        ])


    def __SmoothArray(self):
        print("Smoothing array...")
        arraySmoother = vmtkscripts.vmtkSurfaceArraySmoothing()
        arraySmoother.Surface = self.Surface
        arraySmoother.SurfaceArrayName = self.ContourScalarsArrayName
        arraySmoother.Iterations = 10
        arraySmoother.Relaxation = 1.0
        arraySmoother.Execute()

        self.Surface = arraySmoother.Surface

    def DeleteContourCallback(self, obj):
        self.ContourWidget.Initialize()

    def InteractCallback(self, obj):
        if self.ContourWidget.GetEnabled() == 1:
            self.ContourWidget.SetEnabled(0)
        else:
            self.ContourWidget.SetEnabled(1)

    def Display(self):
        self.vmtkRenderer.Render()

    def ScalarsCallback(self, obj):
        """Update the scalar contours on the surface for fixing."""
        rep = vtk.vtkOrientedGlyphContourRepresentation.SafeDownCast(
                    self.ContourWidget.GetRepresentation()
                )

        pointIds = vtk.vtkIdList()
        self.Interpolator.GetContourPointIds(rep, pointIds)

        points = vtk.vtkPoints()
        points.SetNumberOfPoints(pointIds.GetNumberOfIds())

        for i in range(pointIds.GetNumberOfIds()):
            pointId = pointIds.GetId(i)
            point = self.Surface.GetPoint(pointId)
            points.SetPoint(i,point)

        # Select the region inside or outside closed contour
        selectionFilter = vtk.vtkSelectPolyData()
        selectionFilter.SetInputData(self.Surface)
        selectionFilter.SetLoop(points)
        selectionFilter.GenerateSelectionScalarsOn()
        selectionFilter.SetSelectionModeToSmallestRegion()
        selectionFilter.Update()

        # Get scalars create by selection filter
        selectionScalars = selectionFilter.GetOutput().GetPointData().GetScalars()

        contourScalars = self.Surface.GetPointData().GetArray(
                            self.ContourScalarsArrayName
                        )

        # Update field on surface to include closed region with InsideValue
        for i in range(contourScalars.GetNumberOfTuples()):
            selectionValue = selectionScalars.GetTuple1(i)

            # If inside the closed contour
            if selectionValue < 0.0:
                contourScalars.SetTuple1(i, self.InsideValue)

        self.Actor.GetMapper().SetScalarRange(contourScalars.GetRange(0))
        self.Surface.Modified()
        self.ContourWidget.Initialize()


    def FixJoinedCallback(self, obj):
        """ Function to clip and fix a region from glued parts of the vessels."""

        # Clip surface on ContourScalars field
        self.clipper = vtk.vtkClipPolyData()
        self.clipper.SetInputData(self.Surface)
        self.clipper.GenerateClippedOutputOn()

        # Set active scalar to operate on by clipper
        self.Surface.GetPointData().SetActiveScalars(self.ContourScalarsArrayName)
        self.clipper.GenerateClipScalarsOff()

        # Clip value for generated field (mid value)
        clipValue = 0.5*(self.FillValue + self.InsideValue)
        self.clipper.SetValue(clipValue)
        self.clipper.Update()

        # Fill holes with capping smooth method
        # Smooth cap setup (note that this parameter is local
        # because is different fom the different types of fixes)
        ConstraintFactor = 0.65
        NumberOfRings = 20

        triangle = vtk.vtkTriangleFilter()
        triangle.SetInputData(self.clipper.GetOutput())
        triangle.PassLinesOff()
        triangle.PassVertsOff()
        triangle.Update()

        capper = vtkvmtk.vtkvmtkSmoothCapPolyData()
        capper.SetInputConnection(triangle.GetOutputPort())
        capper.SetConstraintFactor(ConstraintFactor)
        capper.SetNumberOfRings(NumberOfRings)
        capper.Update()

        # Update mapper
        self.vmtkRenderer.Renderer.RemoveActor(self.Actor)
        self.mapper.SetInputData(capper.GetOutput())
        self.mapper.ScalarVisibilityOn()
        self.mapper.Update()

        # Update scene
        self.Actor = vtk.vtkActor()
        self.Actor.SetMapper(self.mapper)
        self.Actor.GetMapper().SetScalarRange(-1.0, 0.0)
        self.Actor.Modified()

        # Get output
        self.Surface = capper.GetOutput()
        self.ContourWidget.Initialize()

        # Call Representation to initialize contour widget
        # on new clipped surface
        self.Representation()

    def FixMarkedRegion(self, obj):
        rep = vtk.vtkOrientedGlyphContourRepresentation.SafeDownCast(
                    self.ContourWidget.GetRepresentation()
                )

        # Get contour point of closed path
        pointIds = vtk.vtkIdList()
        self.Interpolator.GetContourPointIds(rep,pointIds)

        points = vtk.vtkPoints()
        points.SetNumberOfPoints(pointIds.GetNumberOfIds())

        for i in range(pointIds.GetNumberOfIds()):
            pointId = pointIds.GetId(i)
            point = self.Surface.GetPoint(pointId)
            points.SetPoint(i,point)

        # Create array on closed contour
        selectionFilter = vtk.vtkSelectPolyData()
        selectionFilter.SetInputData(self.Surface)
        selectionFilter.SetLoop(points)
        selectionFilter.GenerateSelectionScalarsOn()
        selectionFilter.SetSelectionModeToSmallestRegion() # AHA! smallest region!
        selectionFilter.Update()

        # Get scalars from selection filter
        selectionScalars = selectionFilter.GetOutput().GetPointData().GetScalars()

        # Get scalars defined on surface
        contourScalars = self.Surface.GetPointData().GetArray(self.ContourScalarsArrayName)

        # Update field on surface to include closed region with InsideValue
        for i in range(contourScalars.GetNumberOfTuples()):
            selectionValue = selectionScalars.GetTuple1(i)

            if selectionValue < 0.0:
                contourScalars.SetTuple1(i,self.InsideValue)

        self.Actor.GetMapper().SetScalarRange(contourScalars.GetRange(0))
        self.Surface.Modified()

        # Clip surface on ContourScalars field
        self.clipper = vtk.vtkClipPolyData()
        self.clipper.SetInputData(self.Surface)
        self.clipper.GenerateClippedOutputOn()

        self.Surface.GetPointData().SetActiveScalars(self.ContourScalarsArrayName)

        self.clipper.GenerateClipScalarsOff()

        # Clip value for generated field (mid value)
        clipValue = 0.5*(self.FillValue + self.InsideValue)
        self.clipper.SetValue(clipValue)
        self.clipper.Update()

        # Fill holes with capping smooth method
        # Smooth cap setup (note that this parameter is local
        # because is different fom the different types of fixes)
        ConstraintFactor = 1.2
        NumberOfRings = 20

        triangle = vtk.vtkTriangleFilter()
        triangle.SetInputData(self.clipper.GetOutput())
        triangle.PassLinesOff()
        triangle.PassVertsOff()
        triangle.Update()

        capper = vtkvmtk.vtkvmtkSmoothCapPolyData()
        capper.SetInputConnection(triangle.GetOutputPort())
        capper.SetConstraintFactor(ConstraintFactor)
        capper.SetNumberOfRings(NumberOfRings)
        capper.Update()

        # Update mapper
        self.vmtkRenderer.Renderer.RemoveActor(self.Actor)
        self.mapper.SetInputData(capper.GetOutput())
        self.mapper.ScalarVisibilityOn()
        self.mapper.Update()

        # Update scene
        self.Actor = vtk.vtkActor()
        self.Actor.SetMapper(self.mapper)
        self.Actor.GetMapper().SetScalarRange(-1.0, 0.0)
        self.Actor.Modified()

        # Get output
        self.Surface = capper.GetOutput()
        self.ContourWidget.Initialize()

        # Call Representation to initialize contour widget
        # on new clipped surface
        self.Representation()

    def Representation(self):
        # Define contour field on surface
        contourScalars = vtk.vtkDoubleArray()
        contourScalars.SetNumberOfComponents(1)
        contourScalars.SetNumberOfTuples(self.Surface.GetNumberOfPoints())
        contourScalars.SetName(self.ContourScalarsArrayName)
        contourScalars.FillComponent(0,self.FillValue)

        # Add array to surface
        self.Surface.GetPointData().AddArray(contourScalars)
        self.Surface.GetPointData().SetActiveScalars(
                self.ContourScalarsArrayName
            )

        # Create mapper and actor to scene
        self.mapper = vtk.vtkPolyDataMapper()
        self.mapper.SetInputData(self.Surface)
        self.mapper.ScalarVisibilityOn()

        self.Actor = vtk.vtkActor()
        self.Actor.SetMapper(self.mapper)
        self.Actor.GetMapper().SetScalarRange(-1.0,0.0)
        self.vmtkRenderer.Renderer.AddActor(self.Actor)

        # Create representation to draw contour
        self.ContourWidget = vtk.vtkContourWidget()
        self.ContourWidget.SetInteractor(
            self.vmtkRenderer.RenderWindowInteractor
        )

        rep = vtk.vtkOrientedGlyphContourRepresentation.SafeDownCast(
                    self.ContourWidget.GetRepresentation()
                )

        rep.GetLinesProperty().SetColor(1, 0.2, 0)
        rep.GetLinesProperty().SetLineWidth(3.0)

        pointPlacer = vtk.vtkPolygonalSurfacePointPlacer()
        pointPlacer.AddProp(self.Actor)
        pointPlacer.GetPolys().AddItem(self.Surface)
        rep.SetPointPlacer(pointPlacer)

        self.Interpolator = vtk.vtkPolygonalSurfaceContourLineInterpolator()
        self.Interpolator.GetPolys().AddItem(self.Surface)
        rep.SetLineInterpolator(self.Interpolator)

        # Messages on the screen
        self.vmtkRenderer.AddKeyBinding(
            'i',
            'Start interaction: select closed contour',
            self.InteractCallback
        )

        self.vmtkRenderer.AddKeyBinding(
            'm',
            'Mark region to be fixed',
            self.ScalarsCallback
        )

        self.vmtkRenderer.AddKeyBinding(
            'f',
            'Fix joined regions',
            self.FixJoinedCallback
        )

        self.vmtkRenderer.AddKeyBinding(
            'd',
            'Delete contour \n',
            self.DeleteContourCallback
        )

        self.vmtkRenderer.AddKeyBinding(
            'space',
            'Fix marked region (directly on closed contour)',
            self.FixMarkedRegion
        )

        self.vmtkRenderer.InputInfo(
            'The available modes to fix the surface are:\n'
            '- Select a closed region and press "space" to fix it;\n'
            '- Select a region, first, by interactively drawing closed\n'
            '  constours in the region where vessels are joined and, after it,\n'
            '  press "f" to fix it.\n'
            '\n'
            'All the available commands are shown on the left.\n'
        )

        self.Display()

    def Execute(self):
        if not self.Surface:
            self.PrintError('Error: no Surface.')

        # Initialize renderer
        if not self.vmtkRenderer:
            self.vmtkRenderer = vmtkrenderer.vmtkRenderer()
            self.vmtkRenderer.Initialize()
            self.OwnRenderer = 1

        self.vmtkRenderer.RegisterScript(self)

        # Filter input surface
        triangleFilter = vtk.vtkTriangleFilter()
        triangleFilter.SetInputData(self.Surface)
        triangleFilter.Update()

        self.Surface = triangleFilter.GetOutput()

        # If clip is true, clip surface
        if self.Clip:
            surfaceClipper = vmtkscripts.vmtkSurfaceClipper()
            surfaceClipper.Surface = self.Surface
            surfaceClipper.InsideOut = True
            surfaceClipper.Execute()

            self.Surface = surfaceClipper.Surface

        connectivityFilter = vtk.vtkPolyDataConnectivityFilter()
        connectivityFilter.SetInputData(self.Surface)
        connectivityFilter.ColorRegionsOff()
        connectivityFilter.SetExtractionModeToLargestRegion()
        connectivityFilter.Update()

        self.Surface = connectivityFilter.GetOutput()

        # Smooth and subdivide before fixing
        if self.Smooth:
            smoother = vmtkscripts.vmtkSurfaceSmoothing()
            smoother.Surface  = self.Surface
            smoother.Method   = 'taubin'
            smoother.PassBand = 0.1
            smoother.NumberOfIterations = 30
            smoother.Execute()

            # subdivider = vmtkscripts.vmtkSurfaceSubdivision()
            # subdivider.Surface = smoother.Surface
            # subdivider.Method  = 'butterfly'
            # # subdivider.NumberOfSubdivisions = 2
            # subdivider.Execute()

            self.Surface = smoother.Surface

        # Start representation and access to all operations
        self.Representation()

        # Clean up surface arrays
        self.Surface.GetPointData().RemoveArray(self.ContourScalarsArrayName)

        # Remesh procedure to increase surface quality
        if self.Remesh:
            remesher = vmtkscripts.vmtkSurfaceRemeshing()
            remesher.Surface = self.Surface
            remesher.ElementSizeMode = "edgelength"
            remesher.TargetEdgeLength = 0.20
            remesher.OutputText("Remeshing procedure ...")
            remesher.Execute()

            self.Surface = remesher.Surface

        if self.OwnRenderer:
            self.vmtkRenderer.Deallocate()


if __name__=='__main__':
    main = pypes.pypeMain()
    main.Arguments = sys.argv
    main.Execute()
