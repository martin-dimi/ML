import numpy as np
import matplotlib.pyplot as plt
# from scipy.misc import imread
from matplotlib.pyplot import imread

def add_gaussian_noise(im,prop,varSigma):
    N = int(np.round(np.prod(im.shape)*prop))
    index = np.unravel_index(np.random.permutation(np.prod(im.shape))[1:N],im.shape)
    e = varSigma*np.random.randn(np.prod(im.shape)).reshape(im.shape)
    im2 = np.copy(im).astype('float')
    im2[index] += e[index]
    return im2


def add_saltnpeppar_noise(im,prop):
    N = int(np.round(np.prod(im.shape)*prop))
    index = np.unravel_index(np.random.permutation(np.prod(im.shape))[1:N],im.shape)
    im2 = np.copy(im)
    im2[index] = 1-im2[index]
    return im2

# proportion of pixels to alter
prop = 0.7
varSigma = 0.1

im = imread('pug.jpg')
im = im/255
fig = plt.figure()
ax = fig.add_subplot(131)
ax.imshow(im,cmap='gray')

im2 = add_gaussian_noise(im,prop,varSigma)
ax2 = fig.add_subplot(132)
ax2.imshow(im2,cmap='gray')

im3 = add_saltnpeppar_noise(im,prop)
ax3 = fig.add_subplot(133)
ax3.imshow(im3,cmap='gray')

def neighbours(i,j,M,N,size=4):
    if size==4:
        if (i==0 and j==0):
            n=[(0,1), (1,0)]

        elif i==0 and j==N-1:
            n=[(0,N-2), (1,N-1)]

        elif i==M-1 and j==0:
            n=[(M-1,1), (M-2,0)]
            
        elif i==M-1 and j==N-1:
            n=[(M-1,N-2), (M-2,N-1)]

        elif i==0:
            n=[(0,j-1), (0,j+1), (1,j)]

        elif i==M-1:
            n=[(M-1,j-1), (M-1,j+1), (M-2,j)]

        elif j==0:
            n=[(i-1,0), (i+1,0), (i,1)]

        elif j==N-1:
            n=[(i-1,N-1), (i+1,N-1), (i,N-2)]

        else:
            n=[(i-1,j), (i+1,j), (i,j-1), (i,j+1)]

        return n

    if size==8:
        print('Not yet implemented\n')

    return -1

def icm(img, EPOCHS):
    x = np.zeros_like(im)
    hasChanges = False

    for e in range(EPOCHS):
        for i in range():

