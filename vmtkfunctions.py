from vmtk import vmtkscripts
import numpy as np

# Default File Names
# images
imageDicomFile     = 'imageOriginal.vti'
imageVoiFile       = 'imageVoi.vti'
imageLevelSetsFile = 'imageLevelSets.vti'
imageInitialLevelSetsFile = 'imageInitialLevelSets.vti'

# surfaces
surfaceDicomFile       = 'surfaceDicom.vtp'
surfaceSmoothedFile    = 'surfaceSmoothed.vtp'
surfaceVoiFile         = 'surfaceVoi.vtp'
surfaceVoiSmFile       = 'surfaceVoiSm.stl'
surfaceClippedFile     = 'surfaceClipped.vtp'
surfaceEndClippedFile  = 'surfaceEndClipped.vtp'
surfaceRemeshedFile    = 'surfaceRemeshed.vtp'
surfaceRemeshedSmFile  = 'surfaceRemeshedSm.stl'
surfaceCappedFile      = 'surfaceCapped.vtp'
surfaceOrientedFile    = 'surfaceOriented.vtp'
surfaceWithFlowExtFile = 'surfaceWithFlowExt.vtp'
surfaceFinalFile       = 'surfaceFinal.vtp' 
surfaceDistToCenterlinesFile = 'surfaceDistToCenterlines.vtp'

# centerlines
centerlineWithoutAneurysm = 'centerlineWithoutAneurysm.vtp'
centerlineWithAneurysm    = 'centerlineWithAneurysm.vtp'
centerlineSectionsFile    = 'centerlineSections.vtp'
centerlinesAttributesFile = 'centerlinesAttributes.vtp'
centerlineGeometryFile    = 'centerlineGeometry.vtp'

DelaunayTesselationFile = 'DelaunayTesselation.vtp'
VoronoiDiagramFile = 'VoronoiDiagram.vtp'

# other
referenceSystemsInitialOrientationFile = 'referenceSystemsInitialOrientation.dat'
referenceSystemsOrientedFile = 'referenceSystemsOriented.dat'

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
        imageReader.InputFileName = dicomDirPath+dicomFilePrefix
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

    
# View 3D image with opacity
def viewImageMIP(image):
    """ Function for visualize VTK image objects
        with MIP visualization (internally, uses
        the vmtkimagemipviewer script)
    
        Input arguments:
        - vtkImageData: the image to be displayed.
    
        Output: 3D renderization displaying vtkImageData
        in a MIP projection.
    """
    imageViewer = vmtkscripts.vmtkImageMIPViewer()
    imageViewer.Image = image
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


# NEEDS TO BE TESTED YET!
# Function that receives a surface with open inlets and outlets and returns 
# Numpy array with patch geometrical infos, such as center point, outward normal, patch id and patch type
def getPatchInfo(surface):
    """ Function that generates a Numpy structured ndarray
        with geometrical and type information of open
        inlets and outlets of a vessel network surface.
        The array is structured as follows:
        
        [((xCenter1, yCenter1, zCenter1), (xNormal1, yNormal1, zNormal1), radius1, id1, type),
         ((xCenter2, yCenter2, zCenter2), (xNormal2, yNormal2, zNormal2), radius2, id2, type),
         ...
         ((xCenterN, yCenterN, zCenterN), (xNormalN, yNormalN, zNormalN), radiusN, idN, type)]
         
        The function defines the inlet as the section with largest radius, and accepts 
        vessel geometries with only one flow inlet.
        
        Input arguments:
        - Input surface (vtkPolyData): surface to be analyzed;
        
        Output arguments:
        - Numpy structured ndarray.
    """
    # Type of each columns
    colsType = [('Center', tuple),
                ('Normal', tuple),
                ('Radius', float),
                ('PatchId', int),
                ('PatchType','U10')]
    
    # Empty auxiliar list     
    capsGeometryList = []
    
    # Capping the surface
    surfaceCapper = vmtkscripts.vmtkSurfaceCapper()
    surfaceCapper.Surface = surface
    surfaceCapper.Method = 'centerpoint'
    surfaceCapper.Interactive = 0
    surfaceCapper.Execute()

    # Surface boundary inspector
    # Conveting surface to mesh to get cell entity ids
    surfaceToMesh = vmtkscripts.vmtkSurfaceToMesh()
    surfaceToMesh.Surface = surfaceCapper.Surface
    surfaceToMesh.Execute()

    # Inspecting
    surfaceBoundaryInspector = vmtkscripts.vmtkMeshBoundaryInspector()
    surfaceBoundaryInspector.Mesh = surfaceToMesh.Mesh
    surfaceBoundaryInspector.CellEntityIdsArrayName = surfaceCapper.CellEntityIdsArrayName
    surfaceBoundaryInspector.Execute()

    # Store patch info in python dictionary using vmtksurfacetonumpy
    vmtkToNumpy = vmtkscripts.vmtkSurfaceToNumpy()
    vmtkToNumpy.Surface = surfaceBoundaryInspector.ReferenceSystems
    vmtkToNumpy.Execute()

    dictPatchData = vmtkToNumpy.ArrayDict
    # Creation of a 'capsGeometryArray' ndarray structured object
    # which contains Center, Normals, Radius, and patch type
    # of the surface caps

    # Assigning structuired array with patch info
    intPatchesNum = len( dictPatchData['CellData']['CellPointIds'] )

    for index in range(intPatchesNum):
        # Copy formatted to list
        Center  = tuple(dictPatchData['Points'][index])
        Normal  = tuple(dictPatchData['PointData']['BoundaryNormals'][index])
        PatchId = dictPatchData['PointData']['CellEntityIds'][index]
        Radius  = dictPatchData['PointData']['BoundaryRadius'][index]

        capsGeometryList.append((Center, Normal, Radius, PatchId, 'patch'))

    # Convert to array
    capsGeometryArray = np.array(capsGeometryList, 
                                 dtype=colsType)

    # Change patch type based on maximum radius
    maxRadius = np.max(capsGeometryArray['Radius'])

    for i in range(0, len(capsGeometryArray)):
        if  capsGeometryArray['Radius'][i] == maxRadius:
            capsGeometryArray['PatchType'][i] = 'inlet'
        else:
            capsGeometryArray['PatchType'][i] = 'outlet'

    # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
    # Specify the other inlets here
#     if inletIds is not None:
#         extraInletIds = inletIds

#         # For all the patches ids
#         for patchId in capsGeometryArray['PatchId']:
#             # If the id is an extraInlet id
#             # Check if extraInletIds is not empty
#             if extraInletIds:
#                 # If the patch id is on specified inlet list
#                 if patchId in extraInletIds:
#                     # find its index in capsGeometryArray
#                     index = np.where(capsGeometryArray['PatchId'] == patchId)
#                     capsGeometryArray['PatchType'][index[0]] = 'inlet'
#                 # If not, then name it 'outlet'
#                 else:
#                     index = np.where(capsGeometryArray['PatchId'] == patchId)
#                     capsGeometryArray['PatchType'][index[0]] = 'outlet'
#             else:
#                 continue

    return capsGeometryArray


def patchSurface(surface, outputDir, capped=True):
    """ 
    Function to patch a surface to be used 
    in CFD by creating a mesh in snappyHexMesh.
    The function also orient the surface putting
    the inlet (selected by largest radius) center
    on the origin of the system, with its outward
    normal point to -z direction.
    """

    inletFileName  = 'inlet.stl'
    outletFileName = 'outlet.stl'
    wallFileName   = 'wall.stl'
    CellEntityIdsArray   = 'CellEntityIds'
    BoundaryRadiusArray  = 'BoundaryRadius'
    BoundaryNormalsArray = 'BoundaryNormals'
    
    origin = [0, 0, 0]
    orientation = [0, 0, -1]
    
    
    # Check if patch dir exists
    patchDir = outputDir+'patches/'
    if not os.path.isdir(patchDir):
        os.makedirs(patchDir)
#         print(patchDir+' created.')
    
    
    # Check cap condition
    if capped == False:
        surfaceCapper = vmtkscripts.vmtkSurfaceCapper()
        surfaceCapper.Surface = surface
        surfaceCapper.Method  = 'centerpoint'
        surfaceCapper.Interactive = 0
        surfaceCapper.Execute()
        
        surface = surfaceCapper.Surface

        
    # Surface boundary inspector
    # Conveting surface to mesh to get cell entity ids
    surfaceToMesh = vmtkscripts.vmtkSurfaceToMesh()
    surfaceToMesh.Surface = surface
    surfaceToMesh.Execute()

    # Inspecting
    surfaceBoundaryInspector = vmtkscripts.vmtkMeshBoundaryInspector()
    surfaceBoundaryInspector.Mesh = surfaceToMesh.Mesh
    surfaceBoundaryInspector.CellEntityIdsArrayName = CellEntityIdsArray
    surfaceBoundaryInspector.Execute()

    # Store patch info in python dictionary using vmtksurfacetonumpy
    vmtkToNumpy = vmtkscripts.vmtkSurfaceToNumpy()
    vmtkToNumpy.Surface = surfaceBoundaryInspector.ReferenceSystems
    vmtkToNumpy.Execute()

    dictPatchData = vmtkToNumpy.ArrayDict
    

    # Get inlet by maximum radius condition
    # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
    # Get max radius and its index
    maxRadius = max(dictPatchData['PointData'][BoundaryRadiusArray])
    inletPos, = np.where( dictPatchData['PointData'][BoundaryRadiusArray] == maxRadius )

    # Number of patches
    nPatches  = len(dictPatchData['PointData'][CellEntityIdsArray])

    # Update dictPatchData with patch info
    dictPatchData['PatchType'] = np.array(nPatches*['outlet'])
    dictPatchData['PatchType'][int(inletPos)] = 'inlet'

    wallId  = 1
    inletId = dictPatchData['PointData'][CellEntityIdsArray][int(inletPos)]

    # Counting the number of inlets and outlets
    # using the numpy.unique function: it returns an ndarray with the unique itens of the input array
    # with the return_counts=True also returns the number of appearance of each item
    patchTypes, numberOfPatches = np.unique(dictPatchData['PatchType'], return_counts=True)

    
    # Zip both patchTypes and numberOfPatches and convert to dictionary
    # with number of inlets and outlets
    patchCount = dict(zip(patchTypes, numberOfPatches))

    # Surface reorientation
    # The vmtksurfacereferencesystemtransform script takes a surface (the surfaceCapped above) 
    # and rotates and translates it conforming one of its patches (in our case the inlet patch) 
    # to a target reference system for that, it uses the output of the vtmkboundaryinspector
    surfaceTransform = vmtkscripts.vmtkSurfaceReferenceSystemTransform()
    surfaceTransform.Surface = surfaceToMesh.Surface

    # Target reference system parameters
    surfaceTransform.ReferenceOrigin  = origin  # translate to origin of system
    surfaceTransform.ReferenceNormal1 = orientation # inlet normal will coincide with -z axis orientation
    surfaceTransform.ReferenceNormal2 = orientation

    # Surface reference system
    surfaceTransform.ReferenceSystems = surfaceBoundaryInspector.ReferenceSystems
    # to get the reference systems of inlet patch
    # Note that, if there is more than one inlet, the inlet chose is the one with largest area
    surfaceTransform.ReferenceSystemId = inletId
    surfaceTransform.ReferenceSystemsIdArrayName      = CellEntityIdsArray
    surfaceTransform.ReferenceSystemsNormal1ArrayName = BoundaryNormalsArray
    surfaceTransform.ReferenceSystemsNormal2ArrayName = BoundaryNormalsArray
    surfaceTransform.Execute()

    
    # Using vmtkThreshold script to extract patches for mesh generations in snappy
    # Extracting first the wall
    extractThreshold = vmtkscripts.vmtkThreshold()
    extractThreshold.Surface = surfaceTransform.Surface
    extractThreshold.LowThreshold  = wallId
    extractThreshold.HighThreshold = wallId
    extractThreshold.SurfaceOutputFileName = patchDir+wallFileName
    extractThreshold.OutputText('Extracting surface with id '+str(wallId)+'\n')
    extractThreshold.Execute()
    extractThreshold.IOWrite()
    extractThreshold.OutputText('Patch saved in '+extractThreshold.SurfaceOutputFileName+'\n')

    # Outlet initial index (to increment) and to be used in output filename
    outletIndex = 1
    inletIndex  = 1

    # Loop to extract inlet and outlet patches
    for i in np.arange(nPatches):
        # Instantiate vmtkthreshold
        extractThreshold = vmtkscripts.vmtkThreshold()
        extractThreshold.Surface       = surfaceTransform.Surface
        extractThreshold.LowThreshold  = dictPatchData['PointData'][CellEntityIdsArray][i]
        extractThreshold.HighThreshold = dictPatchData['PointData'][CellEntityIdsArray][i]
        extractThreshold.OutputText('Extracting surface with id '+str(dictPatchData['PointData'][CellEntityIdsArray][i])+'\n')
        
        
        # Defining output file names
        if dictPatchData['PatchType'][i] == 'inlet':

            if patchCount['inlet'] == 1:
                inletFilePath = patchDir+inletFileName
            else:
                prefix = inletFileName[0:5]
                suffix = inletFileName[5:9]
                inletFilePath = patchDir + prefix + str(inletIndex) + suffix
                inletIndex += 1
                
            extractThreshold.SurfaceOutputFileName = inletFilePath

            
        elif dictPatchData['PatchType'][i] == 'outlet':
            if patchCount['outlet'] == 1:
                outletFilePath = patchDir+outletFileName
            else:
                prefix = outletFileName[0:6]
                suffix = outletFileName[6:10]
                outletFilePath = patchDir + prefix + str(outletIndex) + suffix
                outletIndex += 1

            extractThreshold.SurfaceOutputFileName = outletFilePath

        extractThreshold.Execute()
        extractThreshold.OutputText('Patch saved in '+extractThreshold.SurfaceOutputFileName+'\n')
        extractThreshold.IOWrite()