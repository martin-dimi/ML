import numpy as np

x = np.array([1, 2, 3]) # create 3 vector
y = np.array([[2], [3], [4]]) # create a 3x1 vector
A = np.array([[1, 2, 4],[2, 6, 8],[3, 3, 3]]) # create a 3x3 matrix

# print dimensionality
print('x:', x.shape, 'y:', y.shape, 'A:', A.shape)

np.zeros((2,2)) # create a 2x2 zero matrix
np.ones((2,3)) # create a 2x3 matrix of ones
np.eye(5) # create a 5x5 identity matrix
np.empty((3,4)) # create a 3x4 placeholder matrix
np.arange(1,9,2) # create a vector with values from 1 to 9 with increment of 2
np.linspace(0,1,100) # create a vector of 100 linearly spaced values between 0 and 1