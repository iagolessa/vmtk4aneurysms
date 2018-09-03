from vmtk import vmtkscripts

# Collection of function for some usual tasks

# Image reader
def readImage(imageFileName):
    """ Function to read image VTK object from disk.
        The function returns an image VTK object. 
        
        Input arguments: 
        - imageFileName (str): string containing the
                         image filename with full path;
        
        Output: vtkImageObject 
    """
    # Instantiation of object image reader
    imageReader = vmtkscripts.vmtkImageReader()
    # Input members
    imageReader.InputFileName = imageFileName
    imageReader.Execute()
    
    return imageReader.Image 


def readDicom(dicomDirPath,dicomFilePrefix):
    """ Funtion to read a DICOM directory and 
        returns a vtkImageData. The file prefix 
        can be added, if not must be specified as
        'None'.
        
        Input arguments:
        - dicomDirPath (str): a string with the path to DICOM image files
        - dicomFilePrefix (str or None): prefix of DICOM first file (without extension).
            if the DICOM dir contains only one type of file prefix, then 
            use 'None'
            
        Output: vtkImageObject
    """
    # Instantiating image reader object
    imageReader = vmtkscripts.vmtkImageReader()

    # Try to read directly from directory
    # If not, read from first file
    if dicomFilePrefix is not None:
        imageReader.InputFileName = dicomDirPath+dicomFilePrefix+'.dcm'
        imageReader.Execute()
    else:
        imageReader.Format = 'dicom'
        imageReader.InputDirectoryName = dicomDirPath
        imageReader.Execute()
    
    return imageReader.Image



# Image viewer function
def viewImage(image):
    """ Function for visualize VTK image objects.
    
        Input arguments:
        - vtkImageObject: the image to be displayed.
    
        Output: renderer displaying vtkImageData.
    """
    imageViewer = vmtkscripts.vmtkImageViewer()
    imageViewer.Image = image
    imageViewer.Margins = 1
    imageViewer.Execute()

    
# Writing a loaded vtkImageData to disk 
def writeImage(image,imageOutputFileName):
    """ Function that writes a vtkImageData to disk,
        given the vtkImageObject and an output file name. 
        
        Input arguments:
        - vtkImageData: image object containing the image
                        to be written.
        
        Output: file stored at imageOutputFileNale
    """
    
    # Writing image to file
    imageWriter = vmtkscripts.vmtkImageWriter()
    imageWriter.Image = image
    imageWriter.OutputFileName = imageOutputFileName
    imageWriter.Execute()

    
    
# Surface reader
def readSurface(surfaceFileName):
    """ Function to read surface VTK object from disk.
        The function returns a surface VTK object. 
        
        Input arguments: 
        - surfaceFileName (str): string containing the
                                 surface filename with full path;
        
        Output: vtkPolyData object 
     """
    surfaceReader = vmtkscripts.vmtkSurfaceReader()
    surfaceReader.InputFileName = surfaceFileName
    surfaceReader.Execute()
    
    return surfaceReader.Surface

# Viewing surface
def viewSurface(surface):
    """ Function for visualize VTK surface objects.
    
        Input arguments:
        - vtkPolyData: the surface to be displayed.
    
        Output: renderer displaying vtkPolyData.
    """
    surfaceViewer = vmtkscripts.vmtkSurfaceViewer()
    surfaceViewer.Surface = surface
    surfaceViewer.Execute()

# Writing a surface
def writeSurface(surface,fileName,mode):
    """ Function that writes a surface to disk,
        given the vtkPolyData and an output file name. 
        
        Input arguments:
        - vtkPolyData: poly data object containing the 
                       surface to be written.
        - fileName (str): a string containing the file name.
        - mode (ascii,binary): mode to be written.
        
        Output: file stored at fileName.
    """
    surfaceWriter = vmtkscripts.vmtkSurfaceWriter()
    surfaceWriter.Surface = surface
    surfaceWriter.OutputFileName = fileName
    
    if mode == 'ascii' or mode == 'binary':
        surfaceWriter.Mode = mode
    
    surfaceWriter.Execute()
    


# Using marching cubes to extract an initial surface
def initialSurface(image,level):
    """ Creates an initial preliminary surface using 
        the Marching Cubes algorithm and extract the 
        largest connected region.
        
        Input arguments: 
        - vtkImageData: a VTK image object;
        - level (float): the level to pass to Marching Cubes 
        
        Output: a vtkPolyData (surface) corresponding to
                that level of the image
    """
    surfaceMarchingCubes = vmtkscripts.vmtkMarchingCubes()

    # The input image to Marching Cubes must be the Level Sets image
    surfaceMarchingCubes.Image = image

    # Parameters
    surfaceMarchingCubes.Level = level
    surfaceMarchingCubes.Execute()
    
    # Run vmtksurfacetriangle to prevent pathed surfaces from Marching Cubes
    # before using the vmtksurfaceconnected
    surfaceTriangle = vmtkscripts.vmtkSurfaceTriangle()
    surfaceTriangle.Surface = surfaceMarchingCubes.Surface
    surfaceTriangle.Execute()

    # Extract largest connected surface
    surfaceConnected = vmtkscripts.vmtkSurfaceConnectivity()
    surfaceConnected.Surface = surfaceTriangle.Surface
    surfaceConnected.Execute()
    
    return surfaceConnected.Surface



def viewCenterline(centerline,array):
    """Function to visualize a centerline with or without ids."""
    centerlinesViewer = vmtkscripts.vmtkCenterlineViewer()
    
    listOfArrays = ["CenterlineIds",
                    "TractIds",
                    "GroupIds",
                    "Blanking",
                    "MaximumInscribedSphereRadius"]
    
    centerlinesViewer.Centerlines = centerline
    
    if array in listOfArrays:
        centerlinesViewer.CellDataArrayName = array
    else:
        print("Array name not known")
    
    centerlinesViewer.Execute()
    


# Functions that encapsulate the smoothing and subdivider operations
# because they may be used on other parts of the procedure

# Smooth surface
def smoothSurface(surface):
    """ Surface smoother based on Taubin or Laplace algorithm.
        
        Input arguments:
        - vtkPolyData: the surface object to be smoothed;
        
        Output: vtkPolyData
    """
    
    # Instantiation of vmtk script object
    surfaceSmoother = vmtkscripts.vmtkSurfaceSmoothing()
    
    # Smoothing parameters
    surfaceSmoother.Surface = surface
    surfaceSmoother.Method = 'taubin'
    surfaceSmoother.NumberOfIterations = 30
    surfaceSmoother.PassBand = 0.1
    
    #surfaceSmoother.PrintInputMembers()
    #surfaceSmoother.PrintOutputMembers()
    
    # Smooth
    surfaceSmoother.Execute()
    return surfaceSmoother.Surface


# Subdivide triangles
def subdivideSurface(surface):
    """ Surface triangle subdivider based 
        on Taubin or Laplace algorithm.
        
        Input arguments:
        - vtkPolyData: the surface object to be subdivided;
        
        Output: vtkPolyData
    """
    surfaceSubdivide = vmtkscripts.vmtkSurfaceSubdivision()

    surfaceSubdivide.Surface = surface
    surfaceSubdivide.Method = 'butterfly' # or linear
    surfaceSubdivide.Execute()
    
    return surfaceSubdivide.Surface