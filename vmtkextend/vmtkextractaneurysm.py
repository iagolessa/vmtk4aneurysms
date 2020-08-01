#! /usr/bin/env python

# Creating Python Class to handle aneurysms operations
# My idea is to expand this class in the future: 
# extract the aneyurysm and calculate other geometric parameters
import sys
import vtk

from vmtk import vtkvmtk
from vmtk import vmtkscripts
from vmtk import vmtkrenderer
from vmtk import pypes

vmtkextractaneurysm = 'vmtkExtractAneurysm'

class vmtkExtractAneurysm(pypes.pypeScript):
    
    # Constructor
    def __init__(self):
        pypes.pypeScript.__init__(self)
        
        self.Surface = None

        self.AneurysmSurface = None
        self.VesselSurface   = None
        self.OstiumSurface   = None
        self.AneurysmType    = None
        self.ManualMode      = True
        self.ComputeOstium   = False
        self.vmtkRenderer    = None
        self.OwnRenderer     = 0
        
        self.Actor = None
        self.InsideValue   = 0.0
        self.FillValue     = 1.0
        self.ContourWidget = None
        self.Interpolator  = None
        self.AneurysmNeckArrayName = 'AneurysmNeckContourArray'
        
        self.SetScriptName('vmtkextractaneurysm')
        self.SetScriptDoc('extract aneurysm from surface and compute geometric'
                          'data.')
        
        self.SetInputMembers([
            ['Surface','i', 'vtkPolyData', 1, '', 
                'the input surface', 'vmtksurfacereader'],

            ['AneurysmType','type', 'str', 1,'["lateral", "terminal"]', 
                'aneurysm type'],

            ['ManualMode','manual', 'bool', 1,'', 
                'enable manual mode (works for both types, however is mandatory for terminal case)'],
            
            ['ComputeOstium','computeostium', 'bool', 1,'', 
                'do not generate ostium surface']
        ])
        
        self.SetOutputMembers([
            ['Surface','o','vtkPolyData',1,'', 
             'the input surface with neck array contour', 
             'vmtksurfacewriter'],
            ['AneurysmSurface','oaneurysm','vtkPolyData',1,'', 
             'the aneurysm sac surface', 
             'vmtksurfacewriter'],
            ['VesselSurface','ovessel','vtkPolyData',1,'', 
             'the clipped surface, i.e., excluding the aneuirysm sac', 
             'vmtksurfacewriter'],
            ['OstiumSurface','oostium','vtkPolyData',1,'', 
             'the ostium surface generated from the contour scalar neck', 
             'vmtksurfacewriter'],
        ])
    
    def SmoothArray(self):
        # Clean poly data, as suggested by Kurt Sansom
        cleaner = vtk.vtkCleanPolyData()
        cleaner.SetInputData(self.Surface)
        cleaner.Update()

        # Change it here to get it local
        arraySmoother = vmtkscripts.vmtkSurfaceArraySmoothing()
        arraySmoother.Surface = cleaner.GetOutput()
        arraySmoother.SurfaceArrayName = self.AneurysmNeckArrayName
        
        # General options
        arraySmoother.Connexity = 1
        arraySmoother.Relaxation = 1.0
        arraySmoother.Iterations = 10
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


    def StartRepresentation(self):
        # Define contour field on surface
        contourScalars = vtk.vtkDoubleArray()
        contourScalars.SetNumberOfComponents(1)
        contourScalars.SetNumberOfTuples(self.Surface.GetNumberOfPoints())
        contourScalars.SetName(self.AneurysmNeckArrayName)
        contourScalars.FillComponent(0,self.FillValue)

    	# Add array to surface
        self.Surface.GetPointData().AddArray(contourScalars)
        self.Surface.GetPointData().SetActiveScalars(self.AneurysmNeckArrayName)

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

        self.vmtkRenderer.AddKeyBinding('i', 
                                        'Start interaction: select aneurysm neck', 
                                        self.InteractCallback)

        self.vmtkRenderer.AddKeyBinding('space', 
                                        'Clip aneurysm surface', 
                                        self.ClipAneurysmManual)
        
        self.vmtkRenderer.AddKeyBinding('d', 
                                        'Delete contour', 
                                        self.DeleteContourCallback)
        
        self.Display()


    def GenerateOstium(self):
        """ Generate an ostium surface based on the aneurysm neck array."""

        cellEntityIdsArrayName = "CellEntityIds"
        method = 'centerpoint' # or simple

        if method == 'simple':
            capper = vtkvmtk.vtkvmtkSimpleCapPolyData()
            capper.SetInputData(self.AneurysmSurface)
        else:
            capper = vtkvmtk.vtkvmtkCapPolyData()
            capper.SetInputData(self.AneurysmSurface)
            capper.SetDisplacement(0.0)
            capper.SetInPlaneDisplacement(0.0)

        capper.SetCellEntityIdsArrayName(cellEntityIdsArrayName)
        capper.SetCellEntityIdOffset(-1) # The neck surface will be 0
        capper.Update()

        # Get maximum id of the surfaces
        ids = capper.GetOutput().GetCellData().GetArray(cellEntityIdsArrayName).GetRange()
        ostiumId = max(ids)

        ostiumExtractor = vtk.vtkThreshold()
        ostiumExtractor.SetInputData(capper.GetOutput())
        ostiumExtractor.SetInputArrayToProcess(0, 0, 0, 1, cellEntityIdsArrayName)
        ostiumExtractor.ThresholdBetween(ostiumId, ostiumId)
        ostiumExtractor.Update()

        # Converts vtkUnstructuredGrid -> vtkPolyData
        gridToSurfaceFilter = vtk.vtkGeometryFilter()
        gridToSurfaceFilter.SetInputData(ostiumExtractor.GetOutput())
        gridToSurfaceFilter.Update()

        ostiumRemesher = vmtkscripts.vmtkSurfaceRemeshing()
        ostiumRemesher.Surface = gridToSurfaceFilter.GetOutput()
        ostiumRemesher.ElementSizeMode = 'edgelength'
        ostiumRemesher.TargetEdgeLength = 0.1
        ostiumRemesher.TargetEdgeLengthFactor = 1.0
        ostiumRemesher.PreserveBoundaryEdges = 1
        ostiumRemesher.Execute()

        ostiumSmoother = vmtkscripts.vmtkSurfaceSmoothing()
        ostiumSmoother.Surface = ostiumRemesher.Surface
        ostiumSmoother.Method = 'taubin'
        ostiumSmoother.NumberOfIterations = 30
        ostiumSmoother.PassBand = 0.1
        ostiumSmoother.BoundarySmoothing = 0
        ostiumSmoother.Execute()

        self.OstiumSurface = ostiumSmoother.Surface



    def ClipAneurysmManual(self, obj):
        """ 
            Interactively select the aneurysm neck to be clipped.
            Obviously, works with both aneurysms types.
        """
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
        contourScalars = self.Surface.GetPointData().GetArray(
                            self.AneurysmNeckArrayName
                        )

	# Update field on surface to include closed region with InsideValue
        for i in range(contourScalars.GetNumberOfTuples()):
            selectionValue = selectionScalars.GetTuple1(i)

            if selectionValue < 0.0:
                contourScalars.SetTuple1(i,self.InsideValue)

        self.Actor.GetMapper().SetScalarRange(contourScalars.GetRange(0))
        self.Surface.Modified()
        
        self.vmtkRenderer.InputInfo('Clipping aneurysm...')
        self.SmoothArray() # surface will have the contour neck array


	# Clip surface on ContourScalars field
        self.clipper = vtk.vtkClipPolyData()
        self.clipper.SetInputData(self.Surface)
        self.clipper.GenerateClippedOutputOn()	
        self.Surface.GetPointData().SetActiveScalars(self.AneurysmNeckArrayName)
        self.clipper.GenerateClipScalarsOff()
        
	# Clip value for generated field (mid value)
        clipValue = 0.5*(self.FillValue + self.InsideValue)
        self.clipper.SetValue(clipValue)
        self.clipper.Update()
        
        # Update mapper
        self.mapper.SetInputData(self.clipper.GetClippedOutput())
        self.mapper.ScalarVisibilityOn()
        self.mapper.Update()
        
        # Update scene
        self.Actor = vtk.vtkActor()
        self.Actor.SetMapper(self.mapper)
        self.Actor.GetMapper().SetScalarRange(-1.0, 0.0)
        self.Actor.Modified()
        
        # Get output and clipper output
        self.VesselSurface   = self.clipper.GetOutput()
        self.AneurysmSurface = self.clipper.GetClippedOutput()

        self.vmtkRenderer.RemoveKeyBinding('i')
        self.vmtkRenderer.RemoveKeyBinding('space')
        self.vmtkRenderer.RemoveKeyBinding('d')
        
        self.ContourWidget.Initialize()
        
        volume = self.Volume()
        surfaceArea = self.SurfaceArea()
        
        self.vmtkRenderer.InputInfo(
            'Done.\n'
            'Aneurysm volume = '+str(round(volume,2))+' mm3\n'
            'Surface area = '+str(round(surfaceArea,2))+' mm2\n'
            'Press q to exit'
        )
        
        
    def ClipLateralAneurysm(self):
        """
        Function to automatically clip lateral aneurysm.
        Works internally by using the DistanceToCenterlines array
        and the clip by array function; the inlet used to calculate
        the model centerline is defined as the patch with 
        largest radius.
        """
        pass        
        # # Get surface inlet and outlet patches' reference systems
        # surfaceRefSystem = vmtkscripts.vmtkBoundaryReferenceSystems()
        # surfaceRefSystem.Surface = self.Surface
        # surfaceRefSystem.Execute()

        # # Store patch info in python dictionary using vmtksurfacetonumpy
        # vmtkToNumpy = vmtkscripts.vmtkSurfaceToNumpy()
        # vmtkToNumpy.Surface = surfaceRefSystem.ReferenceSystems
        # vmtkToNumpy.Execute()
        # dictPatchData = vmtkToNumpy.ArrayDict

        # # Get inlet by maximum radius condition
        # # ~~~~
        # # Get max radius and its index
        # maxRadius = max(dictPatchData['PointData']['BoundaryRadius'])
        # index,    = np.where( dictPatchData['PointData']['BoundaryRadius'] == maxRadius )
        # inletBarycenterArray = dictPatchData['Points'][int(index)]

        # # Build condition array where centers are not equal to inlet center
        # # therefore, outlet centers
        # notInlet = (dictPatchData['Points'] != inletBarycenterArray)

        # # Inlet and outlet centers
        # inletBarycenters  = dictPatchData['Points'][int(index)].tolist()
        # outletBarycenters = np.extract(notInlet, dictPatchData['Points']).tolist()

        # # Computing centerlines
        # centerlines = vmtkscripts.vmtkCenterlines()
        # centerlines.Surface = self.Surface
        # centerlines.SeedSelectorName = 'pointlist'
        # centerlines.SourcePoints     = inletBarycenters
        # centerlines.TargetPoints     = outletBarycenters
        # centerlines.Execute()

        # # Check if centerline is ok
        # vmtk_functions.viewCenterline(centerlines.Centerlines,None)
        # surfaceViewer = vmtkscripts.vmtkSurfaceViewer()
        # surfaceViewer.Surface = self.Surface
        # surfaceViewer.Opacity = 0.3
        # surfaceViewer.Execute()

        # # Multiply radius by a constant
        # arrayOperation = vmtkscripts.vmtkSurfaceArrayOperation()
        # arrayOperation.Surface   = centerlines.Centerlines
        # arrayOperation.Operation = 'multiplybyc'
        # arrayOperation.Constant  = 1.25
        # arrayOperation.InputArrayName  = centerlines.RadiusArrayName
        # arrayOperation.ResultArrayName = 'ModifiedRadius' #centerlines.RadiusArrayName
        # arrayOperation.Execute()

        # # Calculate distance to centerlines array
        # distanceToCenterlines = vmtkscripts.vmtkDistanceToCenterlines()
        # distanceToCenterlines.Surface     = centerlines.Surface
        # distanceToCenterlines.Centerlines = arrayOperation.Surface
        # distanceToCenterlines.UseRadiusInformation = 1
        # distanceToCenterlines.EvaluateTubeFunction = 1
        # distanceToCenterlines.ProjectPointArrays   = 1
        # distanceToCenterlines.EvaluateCenterlineRadius = 1
        # # Important to remember of the MaximumInscribedSphereRadiusArray !
        # distanceToCenterlines.RadiusArrayName = arrayOperation.ResultArrayName
        # # Execute distance to centerlines
        # distanceToCenterlines.Execute()

        # # Aneurysm clipper
        # aneurysmClipper = vmtkscripts.vmtkSurfaceClipper()
        # aneurysmClipper.Surface = distanceToCenterlines.Surface
        # aneurysmClipper.Interactive = False
        # aneurysmClipper.CleanOutput = True
        # aneurysmClipper.ClipValue   = 0.0
        # aneurysmClipper.ClipArrayName = distanceToCenterlines.DistanceToCenterlinesArrayName
        # aneurysmClipper.Execute()

        # surfaceConnect = vmtkscripts.vmtkSurfaceConnectivity()
        # surfaceConnect.Surface = aneurysmClipper.Surface
        # surfaceConnect.Execute()

        # self.Surface = surfaceConnect.Surface

        
    def Volume(self):
        """Calculate volume and area of surface"""
        getProperties = vmtkscripts.vmtkSurfaceMassProperties()
        getProperties.Surface = self.AneurysmSurface
        getProperties.Execute()
        
        #aneurysmMassProperties.SurfaceArea)+' mm2')
        return getProperties.Volume
    
    
    def SurfaceArea(self):
        """Calculate volume and area of surface"""
        getProperties = vmtkscripts.vmtkSurfaceMassProperties()
        getProperties.Surface = self.AneurysmSurface
        getProperties.Execute()
        
        return getProperties.SurfaceArea

    
    def ShowAneurysm(self):
        aneurysmViewer = vmtkscripts.vmtkSurfaceViewer()
        aneurysmViewer.Surface = self.AneurysmSurface
        aneurysmViewer.Execute()
    
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
        
        # Removed connectivty filter because it conflicted with 
        # array smoothing gprocedure
        # connectivityFilter = vtk.vtkPolyDataConnectivityFilter()
        # connectivityFilter.SetInputData(triangleFilter.GetOutput())
        # connectivityFilter.ColorRegionsOff()
        # connectivityFilter.SetExtractionModeToLargestRegion()
        # connectivityFilter.Update()
        
        self.Surface = triangleFilter.GetOutput()
        
        
        if self.AneurysmType == 'terminal' or self.ManualMode:
            # Start representation and manually clip aneurysm
            self.StartRepresentation()
            
        elif self.AneurysmType == 'lateral' and not self.ManualMode:
            self.ClipLateralAneurysm()
        
        else:
            self.PrintError('Aneurysm type not recognized.')
           
        # Generate ostium surface
        if self.ComputeOstium:
            self.GenerateOstium()
                        
        if self.OwnRenderer:
            self.vmtkRenderer.Deallocate() 


if __name__ == '__main__':
    main = pypes.pypeMain()
    main.Arguments = sys.argv
    main.Execute()
