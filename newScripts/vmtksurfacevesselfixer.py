#!/usr/bin/env python

from __future__ import absolute_import #NEEDS TO STAY AS TOP LEVEL MODULE FOR Py2-3 COMPATIBILITY
import vtk
import sys

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

        self.Actor = None
        self.ContourWidget = None
        self.Interpolator  = None
        self.ContourScalarsArrayName = 'ContourScalars'
        self.InsideValue = 0.0
        self.FillValue   = 1.0

        self.SetScriptName('vmtksurfacevesselfixer')
        self.SetScriptDoc('correct a vessel netwrok surface using clipping and capping functions.')

        self.SetInputMembers([
            ['Surface',	'i', 'vtkPolyData', 1, '', 'the input surface', 'vmtksurfacereader'],
            ['vmtkRenderer', 'renderer', 'vmtkRenderer', 1, '', 'external renderer'],
            ['Remesh' , 'remesh', 'bool', 1, '', 'to apply remeshing procedure after fixing it'],
            ['Smooth' , 'smooth','bool',1,'','if surface must be smoothed before fixing'],
        ])

        self.SetOutputMembers([
            ['Surface', 'o', 'vtkPolyData', 1, '', 'the output surface', 'vmtksurfacewriter']
        ])


    def DeleteContourCallback(self, obj):
        self.ContourWidget.Initialize()


    def InteractCallback(self, obj):
        if self.ContourWidget.GetEnabled() == 1:
            self.ContourWidget.SetEnabled(0)
        else:
            self.ContourWidget.SetEnabled(1)


    def Display(self):
        self.vmtkRenderer.Render()


    def Representation(self):
        # Define contour field on surface
        contourScalars = vtk.vtkDoubleArray()
        contourScalars.SetNumberOfComponents(1)
        contourScalars.SetNumberOfTuples(self.Surface.GetNumberOfPoints())
        contourScalars.SetName(self.ContourScalarsArrayName)
        contourScalars.FillComponent(0,self.FillValue)

		# Add array to surface
        self.Surface.GetPointData().AddArray(contourScalars)
        self.Surface.GetPointData().SetActiveScalars(self.ContourScalarsArrayName)

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
        self.ContourWidget.SetInteractor(self.vmtkRenderer.RenderWindowInteractor)

        rep = vtk.vtkOrientedGlyphContourRepresentation.SafeDownCast(self.ContourWidget.GetRepresentation())
        rep.GetLinesProperty().SetColor(1, 0.2, 0)
        rep.GetLinesProperty().SetLineWidth(3.0)

        pointPlacer = vtk.vtkPolygonalSurfacePointPlacer()
        pointPlacer.AddProp(self.Actor)
        pointPlacer.GetPolys().AddItem(self.Surface)
        rep.SetPointPlacer(pointPlacer)

        self.Interpolator = vtk.vtkPolygonalSurfaceContourLineInterpolator()
        self.Interpolator.GetPolys().AddItem(self.Surface)
        rep.SetLineInterpolator(self.Interpolator)

        self.vmtkRenderer.AddKeyBinding('i', 'Start interaction: select region', self.InteractCallback)
        self.vmtkRenderer.AddKeyBinding('space', 'Fix marked region', self.FixMarkedRegion)
        self.vmtkRenderer.AddKeyBinding('d', 'Delete contour', self.DeleteContourCallback)
        self.Display()
       

    def FixMarkedRegion(self, obj):
        rep = vtk.vtkOrientedGlyphContourRepresentation.SafeDownCast(self.ContourWidget.GetRepresentation())
		
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
        surfaceFixer = vmtkscripts.vmtkSurfaceCapper()
        surfaceFixer.Surface = self.clipper.GetOutput()
        surfaceFixer.Method = 'smooth'
        surfaceFixer.ConstraintFactor = 0.5
        surfaceFixer.NumberOfRings = 6
        surfaceFixer.Interactive = 0
        surfaceFixer.Execute()
        
        
        # Update mapper
        self.vmtkRenderer.Renderer.RemoveActor(self.Actor)
        self.mapper.SetInputData(surfaceFixer.Surface)
        self.mapper.ScalarVisibilityOn()
        self.mapper.Update()
        
        # Update scene
        self.Actor = vtk.vtkActor()
        self.Actor.SetMapper(self.mapper)
        self.Actor.GetMapper().SetScalarRange(-1.0, 0.0)
        self.Actor.Modified()

        # Get output
        self.Surface = surfaceFixer.Surface
        self.ContourWidget.Initialize()

		# Call Representation to initialize contour widget 
		# on new clipped surface
        self.Representation()


    def Execute(self):
        if self.Surface == None:
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
        
        connectivityFilter = vtk.vtkPolyDataConnectivityFilter()
        connectivityFilter.SetInputData(triangleFilter.GetOutput())
        connectivityFilter.ColorRegionsOff()
        connectivityFilter.SetExtractionModeToLargestRegion()
        connectivityFilter.Update()
        
        self.Surface = connectivityFilter.GetOutput()
        
        # Start representation
        self.Representation()        
	
        # Remesh procedure to increase surface quality
        if self.Remesh:
            remesher = vmtkscripts.vmtkSurfaceRemeshing()
            remesher.Surface = self.Surface 
            remesher.ElementSizeMode = "edgelength"
            remesher.TargetEdgeLength = 0.15
            remesher.OutputText("Remeshing procedure ...")
            remesher.Execute()

            self.Surface = remesher.Surface
        
        if self.OwnRenderer:
            self.vmtkRenderer.Deallocate()


if __name__=='__main__':
    main = pypes.pypeMain()
    main.Arguments = sys.argv
    main.Execute()
