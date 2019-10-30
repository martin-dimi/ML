import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import multivariate_normal

def plot_line(ax, w):
    # input data
    X = np.zeros((2,2))
    X[0,0] = -5.0
    X[1,0] = 5.0
    X[:,1] = 1.0
    # because of the concatenation we have to flip the transpose
    y = w.dot(X.T)
    ax.plot(X[:,0], y)

# create prior distribution
tau = 1.0*np.eye(2)     # cov
tau_inv = np.linalg.inv(tau) # cov inverse
w_0 = np.zeros((2,1))   # mean

print(tau)

# sample from prior
n_samples = 100
w_samp = np.random.multivariate_normal(w_0.flatten(), tau, size=n_samples)


# Question 1: horizontal line
# Question 2: vertical line

# Generating data, which later we need to recreate the seed for

# X
X = np.ones((200, 2))
X[:, 0] = np.around(np.arange(-1., 1., 0.01), decimals=2)

# w
w = np.array([-1.3, 0.5]).reshape(2,1)
print

# Error
e = np.random.normal([0], [0.3], 200)

# Y
Y = (np.dot(w.T, X.T) + e).flatten()


def plotdistribution(ax, mu, Sigma):
    x = np.linspace(-1.5, 1.5, 100)
    x1p, x2p = np.meshgrid(x, x)
    pos = np.vstack((x1p.flatten(), x2p.flatten())).T

    pdf = multivariate_normal(mu.flatten(), Sigma)
    Z = pdf.pdf(pos)
    Z = Z.reshape(100, 100)

    ax.contour(x1p, x2p, Z, colors='r', lw=5, alpha=0.7)

    return

fig = plt.figure(figsize=(10,5))

lineAx = fig.add_subplot(122)
posAx = fig.add_subplot(121)
posAx.set_xlabel('w_0')
posAx.set_ylabel('w_1')

index = np.random.permutation(X.shape[0])

# for j in range(0, 3):
#     x_i = X[index[:j],:]
#     y_i = Y[index[:j]]

#     # Compute posterior
#     p1 = np.linalg.inv((tau_inv + 0.3*np.dot(x_i.T, x_i)))
#     p2 = (np.dot(tau_inv, w_0) + (0.3*np.dot(x_i.T,y_i)).reshape(2,1))

#     posterior_mean = np.dot(p1, p2)
#     posterior_var = np.linalg.inv(tau_inv + 0.3*np.dot(x_i.T, x_i))

#     plotdistribution(posAx, posterior_mean, posterior_var)

# for i in range(50,):
samples = 50
i = 10

x_i = X[index[:i], :]
y_i = Y[index[:i]]

# Compute posterior
p1 = np.linalg.inv((tau_inv + 0.3*np.dot(x_i.T, x_i)))
p2 = (np.dot(tau_inv, w_0) + (0.3*np.dot(x_i.T,y_i)).reshape(2,1))

posterior_mean = np.dot(p1, p2)
posterior_var = np.linalg.inv(tau_inv + 0.3*np.dot(x_i.T, x_i))

plotdistribution(posAx, posterior_mean, posterior_var)

# Compute lines
postW = np.random.multivariate_normal(posterior_mean.flatten(), posterior_var, size=samples)

for j in range(0, samples):
    plot_line(lineAx, postW[j,:])

# Print actual line
testX = np.array([[-5, 1], [5,1]])

testY = testX.dot(w)
lineAx.plot(testX[:,0], testY, color='r', linewidth=4)

# Question 1: if it's spherical we have no correlation between the weights, which gives us random lines

# Question 2: it gives us some correlation to the weights

# Question 3: 


def predictiveposterior(m0, S0, beta, x_star, X, y):
    mN, SN = posterior(m0, S0, beta, X, y)

    m_star = mN.T.dot(x_star)
    S_star = 1.0/beta + x_star.T.dot(SN).dot(x_star)

    return m_star, S_star

plt.show()


