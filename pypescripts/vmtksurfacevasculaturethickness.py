#! /usr/bin/env python3

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


import re
import sys
import vtk
import math
import numpy as np
import vtk.numpy_interface.dataset_adapter as dsa

from vmtk import vtkvmtk
from vmtk import vmtkscripts
from vmtk import vmtkrenderer
from vmtk import pypes

from vmtk4aneurysms import vascular_operations as vscop
from vmtk4aneurysms.lib import polydatatools as tools
from vmtk4aneurysms.lib import centerlines as cl

vmtksurfacevasculaturethickness = 'vmtkSurfaceVasculatureThickness'

class vmtkSurfaceVasculatureThickness(pypes.pypeScript):

    _SMALL = 1e-12

    # Constructor
    def __init__(self):
        pypes.pypeScript.__init__(self)

        # "Private" members
        self._wlrMedium = 0.07
        self._wlrLarge = 0.088
        self._diameterMedium = 2.0
        self._diameterLarge = 3.0

        # Public member
        self.Surface = None
        self.Centerlines = None
        self.RadiusArrayName = "MaximumInscribedSphereRadius"
        self.Aneurysm = True
        self.NumberOfAneurysms = 1

        self.NeckComputationMode = "manual"
        self.DistanceToNeckArrayName = 'DistanceToNeck'
        self.AneurysmInfluencedRegionDistance = 0.5
        self.GlobalScaleFactor = 0.75
        self.UniformWallToLumenRatio = False
        self.WallLumenRatio = self._wlrMedium
        self.ThicknessArrayName = 'Thickness'
        self.SmoothingIterations = 10
        self.UseGeodesicDistance = True

        self.SelectAneurysmRegions = False
        self.LocalScaleFactor = 0.75
        self.OnlyUpdateThickness = False

        self.AbnormalHemodynamicsRegions = False
        self.WallTypeArrayName = "WallType"
        self.AtheroscleroticFactor = 1.20
        self.RedRegionsFactor = 0.95

        self.GenerateWallMesh = False
        self.WallMesh = None
        self.WallMeshLayers = 3

        self.vmtkRenderer = None
        self.OwnRenderer = 0
        self.ContourWidget = None
        self.Actor = None
        self.Interpolator = None

        self.SetScriptName('vmtksurfacevasculaturethickness')
        self.SetScriptDoc('')

        self.SetInputMembers([
            ['Surface', 'i', 'vtkPolyData', 1, '',
                'the input surface', 'vmtksurfacereader'],

            ['Centerlines', 'icenterline', 'vtkPolyData', 1, '',
                'the centerlines of the input surface (optional; if not '\
                'passed, it is calculated automatically', 'vmtksurfacereader'],

            ['RadiusArrayName', 'radiusarray', 'str', 1, '',
                'centerline radius array name, if loaded externally'],

            ['Aneurysm', 'aneurysm', 'bool', 1, '',
                'to indicate presence of an aneurysm'],

            ['NumberOfAneurysms', 'naneurysms', 'int', 1, '',
                'integer with number of aneurysms on vasculature'],

            ['NeckComputationMode','neckcomputationmode', 'str' , 1,
                '["interactive","automatic"]',
                'if the neck array is not in the surface, compute it using '\
                'one of these methods'],

            ['AneurysmInfluencedRegionDistance', 'influencedistance',
                'float', 1, '',
                'distance (in mm) that controls how far the '\
                'aneurysm-influenced region goes from the aneurysm neck line'],

            ['GlobalScaleFactor', 'globalfactor', 'float', 1, '',
                'scale fator to control global aneurysm thickness'],

            ['UniformWallToLumenRatio', 'uniformwlr', 'bool', 1, '',
                'forces a uniform WLR informed by the user'],

            ['WallLumenRatio', 'wlr', 'float', 1, '',
                'wall to lumen ratio if uniformwlr is active'],

            ['ThicknessArrayName', 'thicknessarray', 'str', 1, '',
                'name of the resulting thickness array'],

            ['SmoothingIterations', 'iterations', 'int', 1, '',
                'number of iterations for array smoothing'],

            ['UseGeodesicDistance', 'geodesicdistance', 'bool', 1, '',
                'to use the geodesic distance to the neck in the aneurysm'\
                'thickness computation. If false, use Euclidean distance'],

            ['GenerateWallMesh', 'wallmesh', 'bool', 1, '',
                'automatically extrude wall mesh with thickness array'],

            ['WallMeshLayers', 'layers', 'int', 1, '',
                'the number of layers to extrude the wall mesh'],

            ['SelectAneurysmRegions', 'aneurysmregions', 'bool', 1, '',
                'enable selection of aneurysm thinner or thicker regions'],

            ['LocalScaleFactor', 'localfactor', 'float', 1, '',
                'scale fator to control local aneurysm thickness'],

            ['OnlyUpdateThickness', 'updatethickness', 'bool', 1, '',
                'if the thickness array already exists, this options enables '\
                'only to update it through manual selection or automatically'],

            ['AbnormalHemodynamicsRegions', 'abnormalregions', 'bool', 1, '',
                'enable update on thickness based on WallType array created '\
                'based on hemodynamics variables (must be used with '\
                'OnlyUpdateThickness on)'],

            ['AtheroscleroticFactor', 'atheroscleroticfactor', 'float', 1, '',
                'scale fator to update thickness of atherosclerotic regions '\
                'if AbnormalHemodynamicsRegions is true'],

            ['RedRegionsFactor', 'redregionsfactor', 'float', 1, '',
                'scale fator to update thickness of red regions '\
                'if AbnormalHemodynamicsRegions is true'],

            ['WallTypeArrayName', 'walltypearray', 'str', 1, '',
                'name of wall type characterization array']
        ])

        self.SetOutputMembers([
            ['Surface', 'o', 'vtkPolyData', 1, '',
                'the input surface with thickness array', 'vmtksurfacewriter'],

            ['WallMesh', 'owallmesh', 'vtkUnstructuredGrid', 1, '',
                'the output wall mesh', 'vmtkmeshwriter'],
        ])

    def _delete_contour(self, obj):
        self.ContourWidget.Initialize()

    def _interact(self, obj):
        if self.ContourWidget.GetEnabled() == 1:
            self.ContourWidget.SetEnabled(0)
        else:
            self.ContourWidget.SetEnabled(1)

    def _display(self):
        self.vmtkRenderer.Render()

    # Code based on the vmtksurfacearraysmoothing.py script of the VMTK library
    def _smooth_array(self, surface, array, niterations=5, relax_factor=1.0):
        """Surface array smoother."""

        # Clean up surface prior to procedure
        # (this avoid oscillations in the smoothing procedure)
        cleaner = vtk.vtkCleanPolyData()
        cleaner.SetInputData(surface)
        cleaner.Update()

        surface = cleaner.GetOutput()

        # arraySmoother = vmtkscripts.vmtkSurfaceArraySmoothing()
        # arraySmoother.Surface = surface
        # arraySmoother.SurfaceArrayName = array
        # arraySmoother.Connexity = 1
        # arraySmoother.Relaxation = 1.0
        # arraySmoother.Iterations = niterations
        # arraySmoother.Execute()

        _SMALL = 1e-12
        array = surface.GetPointData().GetArray(array)

        extractEdges = vtk.vtkExtractEdges()
        extractEdges.SetInputData(surface)
        extractEdges.Update()

        # Get surface edges
        surfEdges = extractEdges.GetOutput()

        for n in range(niterations):

            # Iterate over all edges cells
            for i in range(surfEdges.GetNumberOfPoints()):
                # Get edge cells
                cells = vtk.vtkIdList()
                surfEdges.GetPointCells(i, cells)

                sum_ = 0.0
                normFactor = 0.0

                # For each edge cells
                for j in range(cells.GetNumberOfIds()):

                    # Get points
                    points = vtk.vtkIdList()
                    surfEdges.GetCellPoints(cells.GetId(j), points)

                    # Over points in edge cells
                    for k in range(points.GetNumberOfIds()):

                        # Compute distance of the current point
                        # to all surface points
                        if points.GetId(k) != i:

                            # Compute distance between a point and surrounding
                            distance = math.sqrt(
                                vtk.vtkMath.Distance2BetweenPoints(
                                    surface.GetPoint(i),
                                    surface.GetPoint(points.GetId(k))
                                )
                            )

                            # Get inverse to act as weight?
                            weight = 1.0/(distance + _SMALL)

                            # Get value
                            value = array.GetTuple1(points.GetId(k))

                            normFactor += weight
                            sum_ += value*weight

                currVal = array.GetTuple1(i)

                # Average value weighted by the surrounding values
                weightedValue = sum_/normFactor

                newValue = relax_factor*weightedValue + \
                           (1.0 - relax_factor)*currVal

                array.SetTuple1(i, newValue)

        return surface

    def _set_thinner_thickness(self, obj):
        """Set thinner thickness on selected region."""

        rep = vtk.vtkOrientedGlyphContourRepresentation.SafeDownCast(
            self.ContourWidget.GetRepresentation()
        )

        # Get loop points from representation to vtkPoints
        pointIds = vtk.vtkIdList()
        self.Interpolator.GetContourPointIds(rep, pointIds)

        points = vtk.vtkPoints()
        points.SetNumberOfPoints(pointIds.GetNumberOfIds())

        # Get points in surface
        for i in range(pointIds.GetNumberOfIds()):
            pointId = pointIds.GetId(i)
            point = self.Surface.GetPoint(pointId)
            points.SetPoint(i, point)

        # Get array of surface selection based on loop points
        selectionFilter = vtk.vtkSelectPolyData()
        selectionFilter.SetInputData(self.Surface)
        selectionFilter.SetLoop(points)
        selectionFilter.GenerateSelectionScalarsOn()
        selectionFilter.SetSelectionModeToSmallestRegion()
        selectionFilter.Update()

        # Get selection scalars
        selectionScalars = selectionFilter.GetOutput().GetPointData().GetScalars()

        # Update both fields with selection
        thicknessArray = self.Surface.GetPointData().GetArray(
            self.ThicknessArrayName
        )

        # TODO: how to select local scale factor on the fly?
        # queryString = 'Enter scale factor: '
        # self.LocalScaleFactor = int(self.InputText(queryString))
        # print(self.LocalScaleFactor)

        # multiply thickness by scale factor in inside regions
        # where selection value is < 0.0, for this case
        for i in range(thicknessArray.GetNumberOfTuples()):
            selectionValue = selectionScalars.GetTuple1(i)
            thicknessValue = thicknessArray.GetTuple1(i)

            if selectionValue < 0.0:
                thinnerThickness = self.LocalScaleFactor*thicknessValue
                thicknessArray.SetTuple1(i, thinnerThickness)

        self.Actor.GetMapper().SetScalarRange(thicknessArray.GetRange(0))
        self.Surface.Modified()
        self.ContourWidget.Initialize()

    def _compute_local_wlr(self, diameter):
        if diameter > self._diameterLarge:
            return self._wlrLarge

        elif diameter < self._diameterMedium:
            return self._wlrMedium

        else:
            # Linear threshold
            deltaWlr = self._wlrLarge - self._wlrMedium
            deltaDiameter = self._diameterLarge - self._diameterMedium
            angCoeff = deltaWlr/deltaDiameter

            return self._wlrMedium + angCoeff*(diameter - self._diameterMedium)

    def ComputeVasculatureThickness(self):
        """Compute thickness array based on diameter and WLR.

        Given input surface with the radius array, computes the thickness by
        multiplying by the wall-to-lumen ration. The aneurysm portion is also
        multiplyed.
        """

        # Compute centerlines
        if not self.Centerlines:
            self.Centerlines = cl.GenerateCenterlines(self.Surface)

        # Compute distance to centerlines
        # It will hold the thickness field at the end
        distanceToCenterlines = vtkvmtk.vtkvmtkPolyDataDistanceToCenterlines()
        distanceToCenterlines.SetInputData(self.Surface)
        distanceToCenterlines.SetCenterlines(self.Centerlines)

        distanceToCenterlines.SetUseRadiusInformation(True)
        distanceToCenterlines.SetEvaluateCenterlineRadius(True)
        distanceToCenterlines.SetEvaluateTubeFunction(False)
        distanceToCenterlines.SetProjectPointArrays(False)

        distanceToCenterlines.SetDistanceToCenterlinesArrayName(
            self.ThicknessArrayName
        )

        distanceToCenterlines.SetCenterlineRadiusArrayName(self.RadiusArrayName)
        distanceToCenterlines.Update()

        # use numpy interface with VTK
        npSurface = dsa.WrapDataObject(distanceToCenterlines.GetOutput())

        distanceArray = npSurface.GetPointData().GetArray(self.ThicknessArrayName)
        radiusArray   = npSurface.GetPointData().GetArray(self.RadiusArrayName)

        # This portion evaluates if distance is much higher
        # than the actual radius array
        # This necessarily will need some smoothing

        # Set high and low threshold factors
        # Are they arbitrary?
        highRadiusThresholdFactor = 1.4
        lowRadiusThresholdFactor  = 0.9

        npMaxRadiusLim = highRadiusThresholdFactor*radiusArray
        npMinRadiusLim = lowRadiusThresholdFactor*radiusArray

        distanceArray = np.where(
                            distanceArray > npMaxRadiusLim,
                            npMaxRadiusLim,
                            distanceArray
                        )

        distanceArray = np.where(
                            distanceArray < npMinRadiusLim,
                            radiusArray,
                            distanceArray
                        )

        # Smooth the distance to centerline array to avoid sudden changes of
        # thickness in certain regions
        nIterations = 5     # add a little bit of smoothing now
        surface = self._smooth_array(
                      npSurface.VTKObject,
                      self.ThicknessArrayName,
                      niterations=nIterations
                  )

        npSurface = dsa.WrapDataObject(surface)

        # Multiply by WLR to have a prelimimar thickness array
        # I assume that the WLR is the same for medium sized arteries
        # but I can change this in a point-wise manner based on
        # the local radius array by using the algorithm contained
        # in the vmtksurfacearrayoperation script
        distanceArray = npSurface.GetPointData().GetArray(self.ThicknessArrayName)
        radiusArray   = npSurface.GetPointData().GetArray(self.RadiusArrayName)

        if self.UniformWallToLumenRatio:

            npSurface.PointData.append(
                dsa.VTKArray([
                    self.WallLumenRatio*(2.0*r)
                    for r in distanceArray
                ]),
                self.ThicknessArrayName
            )

        else:
            print("Using non uniform WLR", end="\n")

            # Compute are store local WLR for debug
            localWLRArray = dsa.VTKArray([
                                self._compute_local_wlr(2.0*r)
                                for r in distanceArray
                            ])

            npSurface.PointData.append(
                localWLRArray,
                "LocalWLR"
            )

            # Compute thickness array and replace the thickness array
            # (originally stored as the distance to neck array with a new one)
            npSurface.PointData.append(
                localWLRArray*(2.0*distanceArray),
                self.ThicknessArrayName
            )

        self.Surface = npSurface.VTKObject
        self.Surface.GetPointData().RemoveArray(self.RadiusArrayName)

    def AneurysmTypeValidator(self, iaType):
        if iaType == 'lateral' or iaType == 'bifurcation':
            return 1

        else:
            return 0

        # if text == 'i':
        #     self.vmtkRenderer.Render()
        #     return 0

        # try:
        #     float(text)
        # except ValueError:
        #     return 0
        # return 1

    def SetAneurysmThickness(self):
        """Calculate and set aneurysm thickness.

        Based on the vasculature thickness distribution, defined as the outside
        portion of the complete geometry from the neck selected by the user,
        estimates an aneurysm thickness by averaging the vasculature thickness
        using as weight function the distance to the "aneurysm-influenced"
        region line. The estimated aneurysm thickness is, then, set on the
        aneurysm surface in the thickness array.

        The aneurysm-influenced neck line is defined as the region between the
        neck line (provided by the user) and the path that is at a distance of
        'AneurysmInfluencedRegionDistance' value (in mm; default 0.5 mm) from
        the neck line.  This strip around the aneurysm is imagined as a region
        of the original vasculature that had its thickness changed by the
        aneurysm growth.

        If the surface does not already have the 'DistanceToNeckArray' scalar,
        then it will prompt the user to select the neck line, which will be
        stored on the surface.
        """
        # Get point array names
        nPointArrays = self.Surface.GetPointData().GetNumberOfArrays()
        pointArrays  = [self.Surface.GetPointData().GetArray(id_).GetName()
                        for id_ in range(nPointArrays)]

        # New procedure: instead of selecting the 'aneurysm-influenced region'
        # by hand (which is subjective), we define this region based on the
        # DistanceToNeckArrayName created by the operator or input with the
        # surface alredy (although this is still subjective, it is possible to
        # automatize it too. In any case, it is 'less subjective' than the
        # other procedure, because the aneurysm neck line is somewhat possible
        # to define precisely).
        distanceToNeckArrays = {}

        # To also account for the possibility of multiple aneurysms,
        # neck distance array of each aneurysm will be stored in a list

        # Check if there is any 'DistanceToNeck<i>' array in points arrays
        # where 'i' indicates that more than one aneurysm are present on
        # the surface.
        r = re.compile(self.DistanceToNeckArrayName + ".*")

        distanceToNeckArrayNames = list(filter(r.match, pointArrays))

        if not distanceToNeckArrayNames:
            for id_ in range(self.NumberOfAneurysms):

                # Update neck array name if more than one aneurysm
                arrayName = self.DistanceToNeckArrayName + str(id_ + 1) \
                            if self.NumberOfAneurysms > 1 \
                            else self.DistanceToNeckArrayName


                if self.NeckComputationMode == "interactive":

                    self.Surface = vscop.MarkAneurysmSacManually(
                                       self.Surface,
                                       aneurysm_neck_array_name=arrayName
                                   )


                elif self.NeckComputationMode == "automatic":

                    # Get dome point by the user
                    domePoint = tools.SelectSurfacePoint(self.Surface)

                    aneurysmType = self.InputText(
                                       "Type aneurysm type ['lateral','bifurcation']:",
                                       self.AneurysmTypeValidator
                                   )

                    # Add input of aneurysm type by the user together with dome
                    # point
                    parentSurface = vscop.HealthyVesselReconstruction(
                                        self.Surface,
                                        aneurysmType,
                                        domePoint
                                    )

                    # Clip the parent vascular surface
                    clipper = vmtkscripts.vmtkSurfaceClipper()
                    clipper.Surface = parentSurface
                    clipper.InsideOut = False
                    clipper.Execute()

                    parentSurface = clipper.Surface

                    # Does not work well with the vascular cases
                    # parentSurface = vscop.ClipVasculature(parentSurface)

                    # Procedure by Piccinelli's work: defined by the aneurysmal
                    # region
                    # This function destroys the arrays in the surface, so
                    # let's compute it in a copy and interpolate back, because
                    # the Thickness array is already here in this procedure
                    cleanSurface = tools.CopyVtkObject(self.Surface)

                    cleanSurface = vscop.MarkAneurysmalRegion(
                                       cleanSurface,
                                       parent_vascular_surface=parentSurface,
                                       gdistance_to_neck_array_name=arrayName,
                                       aneurysm_point=domePoint
                                   )

                    self.Surface = tools.ProjectPointArray(
                                       self.Surface,
                                       cleanSurface,
                                       arrayName
                                   )

                else:
                    raise ValueError(
                              """Neck computation mode either 'interactive'
                              or 'automatic'. {} passed.""".format(
                                  self.NeckComputationMode
                              )
                          )

                # Append only arrays
                distanceToNeckArrays[arrayName] = \
                    dsa.VTKArray(
                        self.Surface.GetPointData().GetArray(arrayName)
                    )


        else:
            distanceToNeckArrays = {arrayName:
                                    dsa.VTKArray(
                                        self.Surface.GetPointData().GetArray(
                                            arrayName
                                        )
                                   ) for arrayName in distanceToNeckArrayNames}

        npDistanceSurface = dsa.WrapDataObject(self.Surface)

        # Update both fields with selection
        thicknessArray = npDistanceSurface.GetPointData().GetArray(
                             self.ThicknessArrayName
                         )

        _SMALL = 1e-12
        for id_, (name, neckScalars) in enumerate(distanceToNeckArrays.items()):

            # Add array to surface before we change it
            npDistanceSurface.PointData.append(
                neckScalars,
                name
            )

            # First compute aneurysm thickness based on vasculature thickness
            # the vasculature is selection value > 0
            onVasculature = neckScalars > self.AneurysmInfluencedRegionDistance

            # Filter thickness and neckScalars
            thicknesses = onVasculature*thicknessArray
            vasculatureDistances = onVasculature*neckScalars

            # Aneurysm thickness as weighted average
            aneurysmThickness = self.GlobalScaleFactor*np.average(
                                    thicknesses,
                                    weights=np.array([
                                        1.0/x if x != 0.0 else 0.0
                                        for x in vasculatureDistances
                                    ])
                                )

            print(
                "Aneurysm "+str(id_ + 1)+" thickness computed: {}".format(
                    aneurysmThickness
                ),
                end="\n"
            )

            # Then, substitute thickness array by aneurysmThickness
            thicknessArray[thicknesses == 0.0] = aneurysmThickness

        self.Surface = npDistanceSurface.VTKObject


    def SelectThinnerRegions(self):
        """Interactvely select thinner regions of the aneurysm."""

        # Initialize renderer
        if not self.OwnRenderer:
            self.vmtkRenderer = vmtkrenderer.vmtkRenderer()
            self.vmtkRenderer.Initialize()
            self.OwnRenderer = 1

        self.Surface.GetPointData().SetActiveScalars(self.ThicknessArrayName)

        # Create mapper and actor to scene
        mapper = vtk.vtkPolyDataMapper()
        mapper.SetInputData(self.Surface)
        mapper.ScalarVisibilityOn()

        self.Actor = vtk.vtkActor()
        self.Actor.SetMapper(mapper)
        self.Actor.GetMapper().SetScalarRange(-1.0, 0.0)
        self.vmtkRenderer.Renderer.AddActor(self.Actor)

        # Create representation to draw contour
        self.ContourWidget = vtk.vtkContourWidget()
        self.ContourWidget.SetInteractor(
            self.vmtkRenderer.RenderWindowInteractor)

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
            'Start interaction: select region',
            self._interact
        )

        self.vmtkRenderer.AddKeyBinding(
            'space',
           'Update thickness',
           self._set_thinner_thickness
        )

        self.vmtkRenderer.AddKeyBinding(
            'd',
            'Delete contour',
            self._delete_contour
        )

        self.vmtkRenderer.InputInfo(
            'Select regions to update thickness\n'  \
            'Current local scale factor: '+         \
            str(self.LocalScaleFactor)+'\n'
        )

        # Update range for lengend
        thicknessArray = self.Surface.GetPointData().GetArray(
                            self.ThicknessArrayName
                        )

        self.Actor.GetMapper().SetScalarRange(thicknessArray.GetRange(0))
        self.Surface.Modified()

        self.Legend = 1
        if self.Legend and self.Actor:
            self.ScalarBarActor = vtk.vtkScalarBarActor()
            self.ScalarBarActor.SetLookupTable(
                self.Actor.GetMapper().GetLookupTable()
            )
            self.ScalarBarActor.GetLabelTextProperty().ItalicOff()
            self.ScalarBarActor.GetLabelTextProperty().BoldOff()
            self.ScalarBarActor.GetLabelTextProperty().ShadowOff()
            self.ScalarBarActor.SetLabelFormat('%.2f')
            self.ScalarBarActor.SetTitle(self.ThicknessArrayName)
            self.vmtkRenderer.Renderer.AddActor(self.ScalarBarActor)

        self._display()

        if self.OwnRenderer:
            self.vmtkRenderer.Deallocate()
            self.OwnRenderer = 0

    def UpdateAbnormalHemodynamicsRegions(self):
        """Based on wall type array, increase or deacrease thickness.

        With a global thickness array already defined on the surface, update
        the thickness based on the wall type array created based on the
        hemodynamics variables, by multiplying it by a factor defined below. As
        explained in the function WallTypeCharacterization of wallmotion.py,
        the three types of wall and the operation performed here for each are:

        .. table:: Local wall type characterization
            :widths: auto

            =====   =============== =========
            Label   Wall Type       Operation
            =====   =============== =========
                0   Normal wall     Nothing (default = 1)
                1   Atherosclerotic Increase elasticity (default factor = 1.20)
                2   "Red" wall      Decrease elasticity (default factor = 0.95)
            =====   =============== =========

        The multiplying factors for the atherosclerotic and red wall must be
        provided at object instantiation, with default values given above.
        The function will look for the array named "WallType" for defining
        its operation.
        """
        # Labels for wall classification
        normalWall  = 0
        thickerWall = 1
        thinnerWall = 2

        # Update both fields with selection
        thicknessArray = self.Surface.GetPointData().GetArray(
                             self.ThicknessArrayName
                         )

        # factor array: name WallType
        wallTypeArray = self.Surface.GetCellData().GetArray(
                            self.WallTypeArrayName
                        )

        # Update WallType array with scale factor
        # this is important to have a smooth field to multiply with the
        # thickness array (scale factor can be viewed as a continous
        # distribution in contrast to the WallType array that is discrete)
        for i in range(wallTypeArray.GetNumberOfTuples()):
            wallTypeValue  = wallTypeArray.GetTuple1(i)

            if wallTypeValue == thickerWall:
                newValue = self.AtheroscleroticFactor

            elif wallTypeValue == thinnerWall:
                newValue = self.RedRegionsFactor

            else:
                newValue = 1.0

            wallTypeArray.SetTuple1(i, newValue)

        # Interpolate WallType cell data to point data
        cellDataToPointData = vtk.vtkCellDataToPointData()
        cellDataToPointData.SetInputData(self.Surface)
        cellDataToPointData.PassCellDataOff()
        cellDataToPointData.Update()

        self.Surface = cellDataToPointData.GetOutput()

        # self.Surface.GetCellData().RemoveArray(self.WallTypeArrayName)

        wallTypeArray = self.Surface.GetPointData().GetArray(
                            self.WallTypeArrayName
                        )

        # multiply thickness by scale factor
        for i in range(thicknessArray.GetNumberOfTuples()):
            wallTypeValue  = wallTypeArray.GetTuple1(i)
            thicknessValue = thicknessArray.GetTuple1(i)

            thicknessArray.SetTuple1(i, wallTypeValue*thicknessValue)

        # Update name of abnormal regions factor final array
        wallTypeArray.SetName("AbnormalFactorArray")


    def ExtrudeWallMesh(self):
        """Extrude wall along normals with thickness array."""

        normals = vmtkscripts.vmtkSurfaceNormals()
        normals.Surface = self.Surface
        normals.FlipNormals = 0
        normals.Execute()

        surfaceToMesh = vmtkscripts.vmtkSurfaceToMesh()
        surfaceToMesh.Surface = normals.Surface
        surfaceToMesh.Execute()

        wallMesh = vmtkscripts.vmtkBoundaryLayer()
        wallMesh.Mesh = surfaceToMesh.Mesh
        wallMesh.WarpVectorsArrayName = normals.NormalsArrayName
        wallMesh.ThicknessArrayName = self.ThicknessArrayName
        wallMesh.ThicknessRatio = 1
        wallMesh.NumberOfSubLayers = self.WallMeshLayers

        # Setup
        wallMesh.NumberOfSubsteps = 7000
        wallMesh.Relaxation = 0.01
        wallMesh.LocalCorrectionFactor = 0.5

        # Entity ids for new mesh
        wallMesh.VolumeCellEntityId = 0
        wallMesh.InnerSurfaceCellEntityId = 1
        wallMesh.OuterSurfaceCellEntityId = 2
        wallMesh.SidewallCellEntityId = 3

        wallMesh.IncludeSidewallCells = 0
        wallMesh.IncludeSurfaceCells = 0
        wallMesh.NegateWarpVectors = 0
        wallMesh.Execute()

        # Clean fields and cell arrays
        solidWallMesh = wallMesh.Mesh

        nFields     = solidWallMesh.GetFieldData().GetNumberOfArrays()
        nCellArrays = solidWallMesh.GetCellData().GetNumberOfArrays()

        for field in range(nFields):
            solidWallMesh.GetFieldData().RemoveArray(field)

        for array in range(nCellArrays):
            solidWallMesh.GetCellData().RemoveArray(array)

        self.WallMesh = solidWallMesh

    def Execute(self):

        if self.Surface == None:
            self.PrintError('Error: no Surface.')

        # I had a bug with the 'select thinner regions' with
        # polygonal meshes. So, operate on a triangulated surface
        # and map final result to orignal surface
        cleaner = vtk.vtkCleanPolyData()
        cleaner.SetInputData(self.Surface)
        cleaner.Update()

        # Reference to original surface
        polygonalSurface = cleaner.GetOutput()

        # But will operate on this one
        self.Surface = cleaner.GetOutput()

        # Will operate on the triangulated one
        triangulate = vtk.vtkTriangleFilter()
        triangulate.SetInputData(self.Surface)
        triangulate.Update()

        self.Surface = triangulate.GetOutput()

        # Initialize renderer
        if not self.vmtkRenderer:
            self.vmtkRenderer = vmtkrenderer.vmtkRenderer()
            self.vmtkRenderer.Initialize()
            self.OwnRenderer = 1

        self.vmtkRenderer.RegisterScript(self)

        if self.OnlyUpdateThickness and not self.AbnormalHemodynamicsRegions:
            self.SelectThinnerRegions()

        elif self.OnlyUpdateThickness and self.AbnormalHemodynamicsRegions:
            self.UpdateAbnormalHemodynamicsRegions()

        else:
            self.ComputeVasculatureThickness()

            if self.Aneurysm:
                self.SetAneurysmThickness()

                if self.OwnRenderer:
                    self.vmtkRenderer.Deallocate()
                    self.OwnRenderer = 0

                if self.SelectAneurysmRegions:
                    self.SelectThinnerRegions()


        # After array create, smooth it hard
        self.Surface = self._smooth_array(self.Surface,
                                          self.ThicknessArrayName,
                                          niterations=self.SmoothingIterations)

        # Map final thickness field to original surface
        surfaceProjection = vtkvmtk.vtkvmtkSurfaceProjection()
        surfaceProjection.SetInputData(polygonalSurface)
        surfaceProjection.SetReferenceSurface(self.Surface)
        surfaceProjection.Update()

        self.Surface = surfaceProjection.GetOutput()

        if self.GenerateWallMesh:
            self.ExtrudeWallMesh()

        if self.OwnRenderer:
            self.vmtkRenderer.Deallocate()


if __name__ == '__main__':
    main = pypes.pypeMain()
    main.Arguments = sys.argv
    main.Execute()
