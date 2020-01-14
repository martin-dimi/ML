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

def f(x, beta=0, alpha1=1.0, alpha2=1.0):
    return np.sin(3.0*x) - alpha1*x + alpha2*x**2 + beta*np.random.rand(x.shape[0])

def expected_improvement(f_star, mu, varSigma, x, theta):
    # norm.cdf(x, loc, scale) evaluates the cdf of the normal distribution

    psi = norm(mu, varSigma).cdf(x)
    exploitation = (f_star - mu).dot(psi)

    exploration = varSigma.dot(np.random.multivariate_normal(mu, varSigma))

    alpha = exploitation + exploration

    return alpha

theta = [1,1]
X = np.linspace(-6, 6, 500).reshape(-1,1)
Y = f(X)

# x = np.reshape(np.array([]), (-1, 1))
# y = np.reshape(np.array([]), (-1, 1))

# mu_star, varSigma_star = surrogate_belief(x, y, X, theta)
# y_star = np.random.multivariate_normal(mu_star, varSigma_star, 20)

fig = plt.figure()
ax = fig.add_subplot(131)
ax.plot(X, Y)

# ax2 = fig.add_subplot(122)
# ax2.plot(x_star, alpha)

# fig.show()


def BO(X, theta, fig, size=3, iter=5):
    # Points
    X_points = np.reshape(np.array([]), (-1, 1))
    Y_points = np.reshape(np.array([]), (-1, 1))
    current_best_y = 10

    # Pick random set of starting points
    index = np.random.permutation(X.shape[0])
    xs = X[index[0:size]]
    # X = np.delete(X, index[0:size])

    for i in range(iter):
        fig = plt.figure()

        X_points = np.reshape(X_points, (-1, 1))
        Y_points = np.reshape(Y_points, (-1, 1))

        # compute the predictive posterior over the points
        mu, varSigma = surrogate_belief(X_points.reshape(-1,1), Y_points.reshape(-1,1), X.reshape(-1, 1), theta)
        y_star = np.random.multivariate_normal(mu, varSigma, 20)

        # compute the acquisition for all the points not included in the model
        alpha = expected_improvement(current_best_y, mu, varSigma, X, theta)

        ax = fig.add_subplot(132)
        ax.plot(X, y_star.T)
        ax.scatter(X_points, Y_points, 200, 'k', '*', zorder=5)

        ax = fig.add_subplot(133)
        ax.plot(X, alpha.T)
        # fig.show()

        # pick the highest (best candidate) alpha
        best_candidate = np.argmax(alpha)

        # Evaluate and save the best candidate
        y_point = f(np.array([X[best_candidate]]))
        X_points = np.append(X_points, X[best_candidate])
        Y_points = np.append(Y_points, y_point)
        X = np.delete(X, best_candidate)

        # Change the current best estimate for min if necessary
        if current_best_y > y_point:
            current_best_y = y_point

BO(X, theta, fig)
fig.show()