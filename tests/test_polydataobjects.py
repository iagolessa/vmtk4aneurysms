import sys

from vmtk import vtkvmtk
from vmtk import vmtkscripts

from ..vmtkextend import customscripts
from ..aneurysm_neck import AneurysmNeckPlane
from .. import polydatatools as tools
from .. import polydataobjects as obj

def test_SurfaceClass(file_name, n_cells):
    surfaceModel = obj.Surface.from_file(file_name)

    tools.ViewSurface(surfaceModel.GetSurfaceObject(), 
                      array_name='Local_Shape_Type')

    nCells = surfaceModel.GetSurfaceObject().GetNumberOfCells()
    # surfaceModel.GetSurfaceArea(), surfaceModel.GetVolume()
    # surfaceModel.GetCellArrays(), surfaceModel.GetPointArrays()

    assert nCells == n_cells, "Should be "+str(n_cells)

if __name__=='__main__':
    file_ = sys.argv[1]
    cells = int(sys.argv[2])
    test_SurfaceClass(file_, cells)
