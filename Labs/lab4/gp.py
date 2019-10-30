import numpy as np
import matplotlib.pyplot as plt
from scipy.spatial.distance import cdist


# Radial Basis Function / Gaussian
def rbf_kernel(x1, x2, varSigma, lengthscale, noise = 0):
    if x2 is None:
        d = cdist(x1, x1)
    else:
        d = cdist(x1, x2)
    K = varSigma * np.exp(-np.power(d, 2) / lengthscale)
    return K

def lin_kernel(x1, x2, varSigma):
    if x2 is None:
        return varSigma*x1.dot(x1.T)
    else:
        return varSigma*x1.dot(x2.T)

def white_kernel(x1, x2, varSigma):
    if x2 is None:
        return varSigma*np.eye(x1.shape[0])
    else:
        return np.zeros(x1.shape[0], x2.shape[0])

def periodic_kernel(x1, x2, varSigma, period, lenthScale):
    if x2 is None:
        d = cdist(x1, x1)
    else:
        d = cdist(x1, x2)
    return varSigma*np.exp(-(2*np.sin((np.pi/period)*np.sqrt(d))**2)/lenthScale**2)

# Each fi (output of function f(xi)) is a random variable which
# is normally distributed N(0, exponentiated quadratic kernal function defined above)

# X data, also the index set for the marginal of the Gaussian Process
x = np.linspace(-6, 6, 200).reshape(-1, 1)

# Covariance matrix for the indexed subset of the Gaussian Process
rbfCov = rbf_kernel(x, None, 1.0, 2.0)
linCov = lin_kernel(x, None, 1.0)
whCov = white_kernel(x, None, 1.0)
perCov = periodic_kernel(x, None, 1.0, 1, 2.0)

# Mean vector set to 0
mu = np.zeros(x.shape[0])

# get 20 samples from the Gaussian Distribution

# fig = plt.figure()
# ax1 = fig.add_subplot(221)
# ax2 = fig.add_subplot(222)
# ax3 = fig.add_subplot(223)
# ax4 = fig.add_subplot(224)


# PLOTTING OUR PRIOR
# f = np.random.multivariate_normal(mu, rbfCov, 20)
# ax1.plot(x, f.T)
#
# f = np.random.multivariate_normal(mu, linCov, 20)
# ax2.plot(x, f.T)
#
# f = np.random.multivariate_normal(mu, whCov, 20)
# ax3.plot(x, f.T)
#
# f = np.random.multivariate_normal(mu, perCov, 20)
# ax4.plot(x, f.T)
#
# plt.show()


####### Compute the posterior

def compute_gp_posterior(x1, y1, xStar, lengthScale, varSigma, noise):
    # Computing the combined/joined kernal that includes:
    #       1. k(x1,x1) .. k(x1,xn)
    #       2. k(x1,x*1) .. k(x1, x*n)
    #       3. k(x*1, x*1) .. k(x*1, x*n)

    k_xx = rbf_kernel(x1, None, varSigma, lengthScale)              # the subset containing only x's
    k_starX = rbf_kernel(xStar, x1, varSigma, lengthScale)          # the subset containing both x's and x*'s
    k_starstar = rbf_kernel(xStar, xStar, varSigma, lengthScale)    # the subset containing only x*'s

    # Compute the mean and var using the formulas we computed for GP
    mu = k_starX.dot(np.linalg.inv(k_xx)).dot(y1)
    var = k_starstar - k_starX.dot(np.linalg.inv(k_xx)).dot(k_starX.T)

    return mu.reshape(500), var


data_samples = 3
function_samples = 300
x = np.linspace(-3.1, 3, data_samples)
y = np.sin(2*np.pi/x) + x*0.1 + 0.3*np.random.randn(x.shape[0])

x = np.reshape(x, (-1, 1))
y = np.reshape(y, (-1, 1))

x_star = np.linspace(-6, 6, 500).reshape(-1, 1)
mu_star, var_star = compute_gp_posterior(x, y, x_star, 3.0, 2.0, noise=5)
fstar = np.random.multivariate_normal(mu_star, var_star, function_samples)

gpFig = plt.figure()
ax = gpFig.add_subplot(111)
ax.plot(x_star, fstar.T)
ax.scatter(x, y, 200, 'k', '*', zorder=5)

# Display covariance
# va = var_star.diagonal()
# ax.fill_between(x_star[:,0], (mu_star - va), (mu_star + va))

plt.show()


























