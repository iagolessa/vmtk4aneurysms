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

from __future__ import absolute_import #NEEDS TO STAY AS TOP LEVEL MODULE FOR Py2-3 COMPATIBILITY
import vtk
import sys
import os

from vmtk import pypes
from vmtk import vmtkscripts

from vmtk4aneurysms.lib import polydatatools as tools

vmtkfoamparticletracer = 'vmtkFoamParticleTracer'

class vmtkFoamParticleTracer(pypes.pypeScript):

    def __init__(self):

        pypes.pypeScript.__init__(self)

        self.CfdMesh = None
        self.InputDirectoryName = None
        self.Pattern = None
        self.FirstTimeStep = None
        self.LastTimeStep = None
        self.IntervalTimeStep = 1
        self.Source = None
        self.Traces = None
        self.Animate = 0
        self.MinSpeed = 0.1

        self.SetScriptName('vmtkfoamparticletracer')
        self.SetScriptDoc(
            'generate traces for animating OpenFOAM vascular simulation result'
        )

        self.SetInputMembers([
            ['InputDirectoryName','directory','str',1,''],
            ['Pattern','pattern','str',1,''],
            ['FirstTimeStep','firststep','int',1,'(0,)'],
            ['LastTimeStep','laststep','int',1,'(0,)'],
            ['IntervalTimeStep','intervalstep','int',1,'(0,)'],
            ['CfdMesh','cfdmesh','vtkUnstructuredGrid',1,'','mesh with merged time-steps', 
				'vmtkmeshreader'],
            ['Source','s','vtkPolyData',1,'','source points', 'vmtksurfacereader'],
            ['Animate','animate','bool',1,'','whether to compute animation'],
            ['MinSpeed','minspeed','float',1,'(0.0,)','lower speed threshold']
        ])

        self.SetOutputMembers([
            ['Traces','o','vtkPolyData',1,'','the output traces',
                'vmtksurfacewriter']
        ])

    def _merge_time_steps(self):
        if not self.InputDirectoryName:
            self.PrintError('Error: no directory.')

        if not self.Pattern:
            self.PrintError('Error: no pattern.')

        if not self.FirstTimeStep:
            self.PrintError('Error: no first timestep.')

        if not self.LastTimeStep:
            self.PrintError('Error: no last timestep.')

        timeStepMerger = vmtkscripts.vmtkMeshMergeTimesteps()
        timeStepMerger.InputDirectoryName = self.InputDirectoryName
        timeStepMerger.Pattern = self.Pattern
        timeStepMerger.FirstTimeStep = self.FirstTimeStep
        timeStepMerger.LastTimeStep = self.LastTimeStep
        timeStepMerger.IntervalTimeStep = self.IntervalTimeStep

        # OpenFOAM results especific fields
        timeStepMerger.VelocityVector = 1
        timeStepMerger.VelocityVectorArrayName = "U"

        timeStepMerger.Execute()
        timeStepMerger.IOWrite()

        self.CfdMesh = timeStepMerger.Mesh

    def Execute(self):

        if not self.CfdMesh:
            # Generate merged mesh
            self._merge_time_steps()

        # Generate source
        # Remesh it so it have more uniformly distributed points
        self.Source = tools.RemeshSurface(
                     tools.Cleaner(self.Source)
					 # iterations=5
                     # target_cell_area=0.01
                 )

        # Resample the fields of the mesh to the
        # (avoid error from the script)
        self.Source = tools.ResampleFieldsToSurface(
						 self.CfdMesh,
						 self.Source
					 )

        # Compute traces
        particleTracer = vmtkscripts.vmtkParticleTracer()

        particleTracer.Mesh = self.CfdMesh
        particleTracer.Source = self.Source

        # I remember these two args were important to get a correct trace
        particleTracer.MaximumNumberOfSteps = 10000000000

        # This is the minimum vel to get the traces
        # Important to get traces near the wall
        particleTracer.MinSpeed = self.MinSpeed
        particleTracer.Execute()

        self.Traces = particleTracer.Traces

        if self.Animate:
            pass
            # # Generate the animation
            # particleAnimator = vmtkscripts.vmtkPathLineAnimator()

            # particleAnimator.InputTracesInputFileName =
            # particleAnimator.Traces =
            # particleAnimator.TimeStep = 0.001
            # particleAnimator.Legend = 1
            # particleAnimator.Method = "particles"
            # particleAnimator.ImagesDirectory =
            # particleAnimator.ColorMap = "rainbow"
            # particleAnimator.ArrayName = "m/s"
            # particleAnimator.ArrayMax = 0.5
            # particleAnimator.Execute()


if __name__=='__main__':
    main = pypes.pypeMain()
    main.Arguments = sys.argv
    main.Execute()
