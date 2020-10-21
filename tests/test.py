# Testing: computes hemodynamcis given an aneurysm OF case
import sys

from vmtk import vtkvmtk
from vmtk import vmtkscripts

from ..vmtkextend import customscripts
from ..aneurysm_neck import AneurysmNeckPlane
from .. import polydatatools as tools
from .. import polydatageometry as geo
from .. import hemodynamics as hm
from .. import vasculature as vsc

density = 1056.0
foamCase = "/home/iagolessa/foam/iagolessa-4.0/run/meshCoarsest/meshCoarsest.foam"
hemodynamicSurfFile = foamCase.replace('.foam', '_Hemodynamics.vtp')
peakSystoleTime = 2.09 
lowDiastoleTime = 2.82

def test_hemodynamics():
    # print("-------------- Processing file "+foamCase+" -------------", end='\n')
    print("Testing hemodynamics", end='\n')

    # The surface must have the following arrays
    necessaryArrays = ['wallShearComponent', 
                       'PSWSS', 'LDWSS', 
                       'WSS_average', 'TAWSS', 
                       'WSS_magnitude_maximum', 
                       'WSS_magnitude_minimum', 
                       'Normals', 'TAWSS_sgradient', 
                       'GON', 'WSSSG_average', 
                       'WSSSG_magnitude_average', 
                       'WSSPI', 'OSI', 'RRT', 
                       'pHat', 'qHat', 'TAWSSG', 
                       'AFI_peak_systole', 'transWSS', 
                       'WSSTG']

    # If computation is correct, than the surface must have 21 arrays
    hemodynamicsSurface = hm.hemodynamics(foamCase,
                                          peakSystoleTime,
                                          lowDiastoleTime,
                                          compute_gon=True,
                                          compute_afi=True)

    # Get computed arrays
    arrays = tools.GetCellArrays(hemodynamicsSurface)
    
    assert set(arrays) == set(necessaryArrays), "Not all arrays computed."

    # Write surface
    # print("Passed test.", end='\n')
    # print("Writing vascular surface.", end='\n')
    tools.WriteSurface(hemodynamicsSurface, hemodynamicSurfFile)


def test_aneurysm_stats():
    statsFile = foamCase.replace('.foam', '_stats.out')
    surface = tools.ReadSurface(hemodynamicSurfFile)

    selectAneurysm = customscripts.vmtkExtractAneurysm()
    selectAneurysm.Surface = surface
    selectAneurysm.Execute()

    surface = selectAneurysm.Surface

    # Computes WSS and OSI statistics
    print("Computing aneurysm stats", end='\n')

    neckArrayName = 'AneurysmNeckContourArray'
    parentArteryArrayName = 'ParentArteryContourArray'

    # Get all computed arrays on surface
    arrays = tools.GetCellArrays(surface)

    with open(statsFile, 'a') as file_:
        for field in arrays:
            print("Computing stats for " + field, end='\n')
            stats = hm.aneurysm_stats(surface, neckArrayName, field)
            file_.write(field + ': ' + str(stats) + '\n')


        # print("Computing WSS avg. over time", end='\n')
        # avgWssOverTime = hm.wss_surf_avg(foamCase,
        #                                  neckSurface,
        #                                  neckArrayName)


        # file_.write("WSS avg. over time \n")
        # file_.write("Time, WSS\n")

        # for time, wss in sorted(avgWssOverTime.items()):
        #     file_.write(str(time)+','+str(wss)+"\n")


        # print("Computing LSA avg. over time", end='\n')
        # lsaOverTime = hm.lsa_instant(foamCase,
        #                              neckSurface,
        #                              neckArrayName, 1.5)

        # file_.write("LSA avg. over time \n")
        # file_.write("Time, LSA\n")

        # for time, lsa in sorted(lsaOverTime.items()):
        #     file_.write(str(time)+','+str(lsa)+"\n")

def test_vasculature_without_aneurysm():
    print("Testing vasculature without aneurysm", end='\n')
    caseFolder = "/home/iagolessa/documents/unesp/doctorate/data/aneurysms/geometries/aneurisk/C0001/"

    surfaceFile = caseFolder + "surface/model.stl"
    vascularReportFile = caseFolder + "vascular_report_without_aneurysm_C0001.out"

    surface = tools.ReadSurface(surfaceFile)

    # Generate a report on the vasculature being loaded
    case = vsc.Vasculature(
        surface,
        with_aneurysm=False,
        manual_aneurysm=False
    )

    # Inspection
    tools.ViewSurface(case.GetSurface(), array_name="Local_Shape_Type")
    tools.ViewSurface(case.GetCenterlines())

    necessaryArraysInCenterlines = ["MaximumInscribedSphereRadius", 
                                    "Curvature", "Torsion", 
                                    "FrenetTangent", "FrenetNormal",
                                    "FrenetBinormal", "Abscissas",
                                    "ParallelTransportNormals"]

    with open(vascularReportFile, 'a') as file_:
        arraysInCenterline = []

        file_.write("Arrays in centerline:\n")

        for index in range(case.GetCenterlines().GetPointData().GetNumberOfArrays()):
            arrayName = case.GetCenterlines().GetPointData().GetArray(index).GetName()

            arraysInCenterline.append(arrayName)
            file_.write("\t" + arrayName + "\n")

        assert set(arraysInCenterline) == set(necessaryArraysInCenterlines), \
               "Missing arrays in centerlines."

        # Inlet and outlets
        file_.write("Inlet:" + str(case.GetInletCenters()) + "\n")
        file_.write("Outlets:" + str(case.GetOutletCenters()) + "\n")

        # Bifurcations and branches
        nBifurcations = case.GetNumberOfBifurcations()
        nBranches = len(case.GetBranches())

        assert nBifurcations == 6, "There should have been 6 bifs. in C0001."

        # Number of branches == nBifurcations*2 + 1
        assert nBranches == 2*nBifurcations + 1, "There should have been 13 branches in C0001."

        file_.write("Number of bifurcations: "+str(nBifurcations)+"\n")
        file_.write("Number of branches: "+str(nBranches)+"\n")

        file_.write("--> Branch Id \t Length (mm)\n")
        file_.write("    --------- \t -----------\n")
        for branchId, branch in enumerate(case.GetBranches()):
            length = round(branch.GetLength(),2)
            file_.write("    "+str(branchId)+" \t\t "+str(length)+"\n")

        # Compute wall thickness
        # print("Computing wall thickness", end='\n')
        # case.computeWallThicknessArray()

        tools.WriteSurface(case.GetSurface(), caseFolder+'vascular_surface.vtp')

def test_vasculature_with_aneurysm():
    print("Testing vasculature with aneurysm", end='\n')

    vascularReportFile = foamCase.replace('.foam', '_vascular_report.out')
    withAneurysm = True

    # Scaling surface
    # This is important: it seems that all VMTK operations needs to be in
    # millimeters
    scaling = vmtkscripts.vmtkSurfaceScaling()
    scaling.Surface = tools.ReadSurface(hemodynamicSurfFile)
    scaling.ScaleFactor = 1000.0
    scaling.Execute()

    surface = scaling.Surface

    # Generate a report on the vasculature being loaded
    case = vsc.Vasculature(
        surface,
        with_aneurysm=withAneurysm,
        manual_aneurysm=False
    )

    # Inspection
    tools.ViewSurface(case.GetSurface(), array_name="Local_Shape_Type")
    tools.ViewSurface(case.GetCenterlines())

    # if withAneurysm:
    #     tools.ViewSurface(case.GetAneurysm().GetSurface())
    #     tools.ViewSurface(case.GetAneurysm().GetHullSurface())

    necessaryArraysInCenterlines = ["MaximumInscribedSphereRadius",
                                    "Curvature", "Torsion",
                                    "FrenetTangent", "FrenetNormal",
                                    "FrenetBinormal", "Abscissas",
                                    "ParallelTransportNormals"]

    with open(vascularReportFile, 'a') as file_:
        arraysInCenterline = []

        file_.write("Arrays in centerline:\n")

        for index in range(case.GetCenterlines().GetPointData().GetNumberOfArrays()):
            arrayName = case.GetCenterlines().GetPointData().GetArray(index).GetName()

            arraysInCenterline.append(arrayName)
            file_.write("\t" + arrayName + "\n")

        assert set(arraysInCenterline) == set(necessaryArraysInCenterlines), \
               "Missing arrays in centerlines."

        # Inlet and outlets
        file_.write("Inlet:" + str(case.GetInletCenters()) + "\n")
        file_.write("Outlets:" + str(case.GetOutletCenters()) + "\n")

        # Bifurcations and branches
        nBifurcations = case.GetNumberOfBifurcations()
        nBranches = len(case.GetBranches())

        # Case 2
        assert nBifurcations == 2, "There should have been 2 bifs. in Case 2."

        # Number of branches == nBifurcations*2 + 1
        assert nBranches == 2*nBifurcations + 1, \
               "There should have been 5 branches in Case 2."

        file_.write("Number of bifurcations: "+str(nBifurcations)+"\n")
        file_.write("Number of branches: "+str(nBranches)+"\n")

        file_.write("--> Branch Id \t Length (mm)\n")
        file_.write("    --------- \t -----------\n")
        for branchId, branch in enumerate(case.GetBranches()):
            length = round(branch.GetLength(),2)
            file_.write("    "+str(branchId)+" \t\t "+str(length)+"\n")

        # Compute wall thickness
        # print("Computing wall thickness", end='\n')
        # case.computeWallThicknessArray()
        # tools.ViewSurface(case.GetSurface(), array_name="Thickness")

        tools.WriteSurface(case.GetSurface(), hemodynamicSurfFile)

        # If has aneurysm
        if withAneurysm:
            file_.write("Aneurysm properties\n")

            aneurysmFileName = foamCase.replace('.foam', '_aneurysm.vtp')
            tools.WriteSurface(case.GetAneurysm().GetSurface(), aneurysmFileName)

            obj = case.GetAneurysm()

            file_.write("\tAneurysms parameters:\n")

            for parameter in dir(obj):
                if parameter.startswith('Get'):
                    attribute = getattr(obj, parameter)()

                    if type(attribute) == float or type(attribute) == tuple:
                        file_.write('\t' + parameter.strip('Get') +
                                    ' = ' + str(attribute) + "\n")


# Test curvature calcuttion
def test_surface_curvature():
    filename = "/home/iagolessa/documents/unesp/doctorate/data/aneurysms/geometries/aneurisk/C0001/surface/model.stl"
    surface = tools.ReadSurface(filename)
    curvatures = geo.SurfaceCurvature(surface)

    tools.ViewSurface(curvatures, array_name="Local_Shape_Type")
    tools.WriteSurface(curvatures, hemodynamicSurfFile)


# Test aneurysm neck computation
def test_aneurysm_neck_plane():
    surfaceFile = '/home/iagolessa/documents/unesp/doctorate/data/aneurysms/geometries/aneurisk/C0001/surface/model.stl'

    print("Computing aneurysm surface", end='\n')
    aneurysmSurface = AneurysmNeckPlane(
                        tools.ReadSurface(surfaceFile),
                        min_variable='area'
                    )

    tools.ViewSurface(aneurysmSurface)
    tools.WriteSurface(aneurysmSurface, '/home/iagolessa/hemodynamics/tmp_aneurysm_area.vtp')

if __name__=='__main__':
    test_hemodynamics()
    # test_aneurysm_stats()
    test_vasculature_without_aneurysm()
    test_vasculature_with_aneurysm()
    # test_surface_curvature()
    test_aneurysm_neck_plane()
