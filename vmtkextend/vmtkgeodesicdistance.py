#! /usr/bin/env python

# Creating Python Class to handle aneurysms operations
# My idea is to expand this class in the future:
# extract the aneyurysm and calculate other geometric parameters
import sys
import vtk

from vmtk import vtkvmtk
from vmtk import vmtkrenderer
from vmtk import pypes

vmtkgeodesicdistance = 'vmtkGeodesicDistance'

class vmtkGeodesicDistance(pypes.pypeScript):

    # Constructor
    def __init__(self):
        pypes.pypeScript.__init__(self)

        self.Surface      = None
        self.vmtkRenderer = None
        self.OwnRenderer  = 0

        self.Actor = None
        self.ContourWidget = None
        self.Interpolator  = None
        self.GeodesicDistanceArrayName = "GeodesicDistance"

        self.SetScriptName('vmtkgeodesicdistancefromneckline')
        self.SetScriptDoc(
            """compute the geodesic distance from an aneurysm neck line draw
            interactively by the user"""
        )

        self.SetInputMembers([
            ['Surface','i', 'vtkPolyData', 1, '',
                'the input surface', 'vmtksurfacereader'],
        ])

        self.SetOutputMembers([
            ['Surface','o','vtkPolyData',1,'',
             'the input surface with geodesic distances', 'vmtksurfacewriter'],
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

        # Initialize renderer
        if not self.vmtkRenderer:
            self.vmtkRenderer = vmtkrenderer.vmtkRenderer()
            self.vmtkRenderer.Initialize()
            self.OwnRenderer = 1

        self.vmtkRenderer.RegisterScript(self)

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

        self.vmtkRenderer.AddKeyBinding(
            'i',
            'Start interaction: select aneurysm neck',
            self.InteractCallback
        )

        self.vmtkRenderer.AddKeyBinding(
            'd',
            'Delete contour',
            self.DeleteContourCallback
        )

        self.Display()

        # Get contour point of closed path
        pointIds = vtk.vtkIdList()
        self.Interpolator.GetContourPointIds(rep,pointIds)

        # Get points set on the surface
        seedPoints = vtk.vtkPoints()
        seedPoints.SetNumberOfPoints(pointIds.GetNumberOfIds())

        for i in range(pointIds.GetNumberOfIds()):
            pointId = pointIds.GetId(i)
            point   = self.Surface.GetPoint(pointId)
            seedPoints.SetPoint(i,point)

        # Build poly data with seedPoints and Radius array (all zeros by now)
        newPolyData    = vtk.vtkPolyData()

        radiusArray = vtk.vtkDoubleArray()
        radiusArray.SetNumberOfComponents(1)
        radiusArray.SetNumberOfTuples(pointIds.GetNumberOfIds())
        radiusArray.SetName("RadiusArray")
        radiusArray.FillComponent(0, 0.0)

        # If needs to change the array values, set here
        # for i in range(pointIds.GetNumberOfIds()):
        #     radiusArray.SetTuple1(i, 0.0)

        newPolyData.SetPoints(seedPoints)
        newPolyData.GetPointData().AddArray(radiusArray)

        # The vtkvmtkNonManifoldFastMarching filter also computes a positive
        # distance from the neck towards the aneurysm. Hence, we also use
        # the points to invert the sign of the scalars on the aneurysm, to
        # conform with the procedure executed when using the Euclidean distance
        # (where the distance from the aneurysm is > 0 only on the branches)

        # Get array of surface selection based on loop points
        selectionScalarsName = "SelectionScalars"

        selectionFilter = vtk.vtkSelectPolyData()
        selectionFilter.SetInputData(self.Surface)
        selectionFilter.SetLoop(seedPoints)
        selectionFilter.GenerateSelectionScalarsOn()
        selectionFilter.SetSelectionModeToSmallestRegion()
        selectionFilter.Update()

        selectionFilter.GetOutput().GetPointData().GetScalars().SetName(
            selectionScalarsName
        )

        self.Surface = selectionFilter.GetOutput()

        # Apply the fast marching algorithm now in the surface
        geodesicFastMarching = vtkvmtk.vtkvmtkNonManifoldFastMarching()
        geodesicFastMarching.SetInputData(self.Surface)
        geodesicFastMarching.UnitSpeedOn()
        geodesicFastMarching.SetSolutionArrayName(
            self.GeodesicDistanceArrayName
        )
        geodesicFastMarching.SetInitializeFromScalars(0)

        # Choose how to input the seeds or polydata
        geodesicFastMarching.SeedsBoundaryConditionsOn()
        geodesicFastMarching.SetSeeds(pointIds)

        geodesicFastMarching.PolyDataBoundaryConditionsOff()
        # geodesicFastMarching.SetBoundaryPolyData(newPolyData)
        # geodesicFastMarching.SetIntersectedEdgesArrayName("EdgeArrayName")

        geodesicFastMarching.Update()

        self.Surface = geodesicFastMarching.GetOutput()

        # Get selection scalars
        selectionScalars = self.Surface.GetPointData().GetArray(
                               selectionScalarsName
                           )

        gdistanceArray = self.Surface.GetPointData().GetArray(
                             self.GeodesicDistanceArrayName
                         )

        # Where selection value is < 0.0, for this case, invert sign of
        # geodesic distance (inside the aneurysm, in this case)
        for i in range(gdistanceArray.GetNumberOfValues()):
            selectionValue = selectionScalars.GetTuple1(i)
            gdistanceValue = gdistanceArray.GetValue(i)

            if selectionValue < 0.0:
                gdistanceArray.SetTuple1(i, -1.0*gdistanceValue)

        self.Surface.GetPointData().RemoveArray(selectionScalarsName)

        # Map final geodesic distance field to original surface
        surfaceProjection = vtkvmtk.vtkvmtkSurfaceProjection()
        surfaceProjection.SetInputData(polygonalSurface)
        surfaceProjection.SetReferenceSurface(self.Surface)
        surfaceProjection.Update()

        self.Surface = surfaceProjection.GetOutput()

        if self.OwnRenderer:
            self.vmtkRenderer.Deallocate()


if __name__ == '__main__':
    main = pypes.pypeMain()
    main.Arguments = sys.argv
    main.Execute()
