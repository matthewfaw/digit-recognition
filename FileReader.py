import struct
import numpy as np
import math


def getImages(filename):
    with open(filename, 'rb') as f:
        # Read magic number
        byte = f.read(4)
        magic=struct.unpack(">i",byte)[0]
        # print magic
        # Read number of images
        byte = f.read(4)
        numIm=struct.unpack(">i",byte)[0]
        # print numIm
        #Read number of rows
        byte = f.read(4)
        rows=struct.unpack(">i",byte)[0]
        # print rows
        #Read number of cols
        byte = f.read(4)
        cols=struct.unpack(">i",byte)[0]
        # print cols
    
        #Go through rest of image training file, parsing to get 60,000 784 dim vectors
        trainIm = np.fromfile(filename,dtype=np.dtype('>B'),count=-1)[16:]
        trainIm = np.reshape(trainIm,(rows,cols,numIm),order='F')
        trainIm = np.ravel(trainIm)
        trainIm = np.reshape(trainIm,(rows*cols,numIm))

        trainIm = trainIm.astype(np.float32, copy=False)/255
        
        return trainIm


def getLabels(filename):
    with open(filename,'rb') as f:
        #read magic number
        byte = f.read(4)
        magic1=struct.unpack(">i",byte)[0]
        # print magic1
        #read number of items
        byte = f.read(4)
        numIt=struct.unpack(">i",byte)[0]
        # print numIt
    
        #get labels:
        trainLab = np.fromfile(filename,dtype=np.dtype('>B'),count=-1)[8:]
        
        return np.transpose(np.asmatrix(trainLab))
        



def getMNIST():
    # print "Training Set Image File:"
    trainIm = getImages('train-images-idx3-ubyte')
    # print "\nTraining Set Label File:"
    trainLab = getLabels('train-labels-idx1-ubyte')
    # print "\nTest Set Image File:"
    testIm = getImages('t10k-images-idx3-ubyte')
    # print "\nTest Set Label File"
    testLab = getLabels('t10k-labels-idx1-ubyte')
    
    return trainIm, trainLab, testIm, testLab


