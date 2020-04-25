"""Provide functions to work with VTK poly data."""

import vtk
from vmtk import vmtkscripts

def readSurface(file_name):
    """Read surface file to VTK object.
               
    Arguments: 
    file_name -- complete path with file name
    """
    reader = vmtkscripts.vmtkSurfaceReader()
    reader.InputFileName = file_name
    reader.Execute()
    
    return reader.Surface


def viewSurface(surface, array_name=None):
    """View surface vtkPolyData objects.
    
    Arguments:
    surface -- the surface to be displayed.
    """
    viewer = vmtkscripts.vmtkSurfaceViewer()
    viewer.Surface = surface
    
    if array_name != None:
        viewer.ArrayName = array_name
        viewer.Legend = True
    
    viewer.Execute()
    
def writeSurface(surface, file_name, mode='binary'):
    """Write surface vtkPolyData. 
        
    Arguments:
    surface -- surface vtkPolyData object
    file_name -- output file name with full path
    
    Optional arguments:
    mode -- mode to write file (ASCII or binary, default binary)
    """
    writer = vtk.vtkPolyDataWriter()
    writer.SetFileName(file_name)
    writer.SetInputData(surface)
    
    if mode == 'binary':
        writer.SetDataModeToBinary()
    elif mode == 'ascii':
        writer.SetDataModeToAscii()
        
    writer.Write()


def smoothSurface(surface):
    """Smooth surface based on Taubin's algorithm."""
    smoother = vmtkscripts.vmtkSurfaceSmoothing()
    smoother.Surface = surface
    smoother.Method  = 'taubin'
    smoother.NumberOfIterations = intThree*intTen
    smoother.PassBand = intOne/intTen
    smoother.Execute()
    
    return smoother.Surface

def cleaner(surface):
    """Polydata cleaner."""
    cleaner = vtk.vtkCleanPolyData()
    cleaner.SetInputData(surface)
    cleaner.Update()
    
    return cleaner.GetOutput()
