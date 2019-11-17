import numpy as np
import matplotlib.pyplot as plt
from scipy.spatial.distance import cdist
from scipy.stats import multivariate_normal

# In supervised learning the believe takes the form of the preference over different hypothesis: we are trying to solve what the f(.) and x's are in 

#                       yi = f(x)

# One of the solutions is just the identity function, hence, x = y. however,
# our preference will tell us that this is not a good hypothesis...


# Gaussian likelihood: p(Y|F) = N(F, β−1I), where β−1 is the precision

fig = plt.figure(figsize=(15, 5))

# GENERATING SPIRAL DATA (X = {x, y})
t = np.linspace(0, 3 * np.pi, 100)
x = np.zeros((t.shape[0], 2))
x[:, 0] = t * np.sin(t)
x[:, 1] = t * np.cos(t)

# Displaying Original Data
pltOriginalX = fig.add_subplot(131)
pltOriginalX.plot(x[:, 0], x[:, 1])
pltOriginalX.set_title("Original X")


# pick a random matrix/weights that maps to Y
# Note that we can't display this data as it lives in a 10D world
w = np.random.randn(10, 2)
y = x.dot(w.T) # Y.shape = (100, 10) => 10 dimentions
y += np.random.randn(y.shape[0], y.shape[1]) # Adding noise
mu_y = np.mean(y, axis=0)


################# Generating back the data ##########################

# maximum likelihood solution to W
def MLW(x,q):
    v,w = np.linalg.eig(np.cov(x.T))
    idx = np.argsort(np.real(v))[::-1][:q]
    return np.real(w[:,idx])

# posterior distribution of latent variable
def posterior(w, x, mu_x, beta):
    A = np.linalg.inv(w.dot(w.T)+1/beta*np.eye(w.shape[0]))
    mu = w.T.dot(A.dot(x-mu_x))
    varSigma = np.eye(w.shape[1]) - w.T.dot(A.dot(w))
    return mu, varSigma

# Get the maximum likelihood solution of W
w = MLW(y, 2)

# Compute predictions for latent space
xPred = np.zeros(x.shape) # shape = (100, 2)
varSigma = [] # shape = (2,2)

for i in range(0, y.shape[0]):
    xPred[i, :], varSigma = posterior(w, y[i, :], mu_y, 1/2)

# Generating density
N = 300
x1 = np.linspace(np.min(xPred[:,0]), np.max(xPred[:, 0]), N)
x2 = np.linspace(np.min(xPred[:,1]), np.max(xPred[:, 1]), N)
x1p, x2p = np.meshgrid(x1, x2)
pos = np.vstack((x1p.flatten(), x2p.flatten())).T

# compute posterior
Z = np.zeros((N,N))
for i in range(0, xPred.shape[0]):
    pdf = multivariate_normal(xPred[i,:].flatten(), varSigma)
    Z += pdf.pdf(pos).reshape(N,N)

ax = fig.add_subplot(132)
ax.set_title("Restored X")
ax.scatter(xPred[:, 0], xPred[:, 1])
ax.set_xticks([])
ax.set_yticks([])

ax = fig.add_subplot(133)
ax.set_title("HeatMap X")
ax.imshow(Z, cmap='hot')
ax.set_ylim(ax.get_ylim()[::-1])
ax.set_xticks([])
ax.set_yticks([])

fig.show()