import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import norm
from scipy.spatial.distance import cdist


# Radial Basis Function / Gaussian
def rbf_kernel(x1, x2, varSigma, lengthscale):
    # print(x1.shape)
    if x2 is None:
        d = cdist(x1, x1)
    else:
        d = cdist(x1, x2)
    K = varSigma * np.exp(-np.power(d, 2) / lengthscale)
    return K

def surrogate_belief(x,f,x_star,theta):

    kxx = rbf_kernel(x, None, theta[0], theta[1])
    kxStarx = rbf_kernel(x, x_star, theta[0], theta[1]).T
    kxStarxStar = rbf_kernel(x_star, None, theta[0], theta[1])

    # print(kxStarx.shape)
    # print(kxx.shape)
    # print(f.shape)

    mu_star = kxStarx.dot(np.linalg.inv(kxx)).dot(f)
    varSigma_star = kxStarxStar - kxStarx.dot(np.linalg.inv(kxx)).dot(kxStarx.T)


    return mu_star.reshape(x_star.shape[0]), varSigma_star

X = np.array([0, 1])
Y = np.array([1, 1])

X = np.reshape(X,(-1,1))
Y = np.reshape(Y,(-1,1))



x_star = np.linspace(-6, 6, 500).reshape(-1, 1)
theta = [1, 1]

mu_star, varSigma_star = surrogate_belief(X, Y, x_star, theta)
y_star = np.random.multivariate_normal(mu_star, varSigma_star, 20)

fig = plt.figure()
ax = fig.add_subplot(111)
ax.plot(x_star, y_star.T)
ax.scatter(X,Y,200,'k','*',zorder=5)

fig.show()