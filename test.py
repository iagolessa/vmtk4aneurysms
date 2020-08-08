# Testing: computes hemodynamcis given an aneurysm OF case
import sys
import polydatatools as tools

from vmtk import vtkvmtk
from vmtk import vmtkscripts
from vmtkextend import customscripts

import hemodynamics as hm

foamCase = sys.argv[1]
neckSurfaceFile = sys.argv[2]
statsFile = sys.argv[3]

peakSystoleTime = 2.09
lowDiastoleTime = 2.96
outFile = foamCase.replace('.foam', '_Hemodynamics.vtp')

density = 1056.0

print("Computing hemodynamics", end='\n')
hemodynamicsSurface = hm.hemodynamics(foamCase,
                                      peakSystoleTime,
                                      lowDiastoleTime,
                                      compute_gon=False,
                                      compute_afi=True)

print("Writing hemodynamics surface", end='\n')
# tools.writeSurface(hemodynamicsSurface, outFile)

print("Computing aneurysm stats", end='\n')

neckSurface = tools.readSurface(neckSurfaceFile)

print("Projecting aneurysm neck array on hemodynamics surface", end='\n')
print("Make sure that their scale are the same!", end='\n')

surfaceProjection = vtkvmtk.vtkvmtkSurfaceProjection()
surfaceProjection.SetInputData(hemodynamicsSurface)
surfaceProjection.SetReferenceSurface(neckSurface)
surfaceProjection.Update()
surface = surfaceProjection.GetOutput()

# Computes WSS and OSI statistics

neckArrayName = 'AneurysmNeckContourArray'
parentArteryArrayName = 'ParentArteryContourArray'

with open(statsFile, 'a') as file_:
    for field in ['TAWSS', 'OSI', 'PSWSS', 'TAWSSG']:
        stats = hm.aneurysm_stats(surface, neckArrayName, field)
        file_.write(field+': '+str(stats)+'\n')


    avgWssOverTime = hm.wss_surf_avg(foamCase, 
                                     neckSurface, 
                                     neckArrayName)


    file_.write("WSS avg. over time \n")
    file_.write("Time, WSS\n")

    for time, wss in sorted(avgWssOverTime.items()):
        file_.write(str(time)+','+str(wss)+"\n")


    # lsaOverTime = hm.lsa_instant(foamCase,
                                 # neckSurface,
                                 # neckArrayName, 1.5)

