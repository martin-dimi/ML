import numpy as np

x = np.arange(3).reshape(3,1) # 3x1 vector
y = np.arange(3).reshape(1,3) # 1x3 vector
A = np.arange(9).reshape(3,3) # 3x3 matrix

print('x:', x.shape, 'y:', y.shape, 'A:', A.shape)

print('x: ', x)
print('y: ', y)

print(x-y)
print(y-x)
print(A-x)