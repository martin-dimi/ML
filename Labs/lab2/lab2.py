import numpy as np
from scipy.stats import beta
import matplotlib.pyplot as plt

# parameters to generate data
mu = 0.2
N = 100

# generate some data
X = np.random.binomial(1,mu,N)
mu_test = np.linspace(0,1,100)

# now lets define our prior
a = 10
b = 10
prior_mean = a / a + b
delta = []

# p(mu|X) = p(X|mu) * p(mu)
# p(mu) = Beta(alpha,beta)
prior_mu = beta.pdf(mu_test,a,b)
print(prior_mu)

# create figure
fig = plt.figure(figsize=(10,5))
ax = fig.add_subplot(111)

# # Question 3
# ax = fig.add_subplot(111)
# ax.plot(range(0, 100), )

# Qustion 2
# # plot prior
# ax.plot(mu_test,prior_mu,'g')
# ax.fill_between(mu_test,prior_mu,color='green',alpha=0.3)
# ax.set_xlabel('$\mu$')
# ax.set_ylabel('$p(\mu|\mathbf{x})$')

def posterior(a,b,X):
    a_n = a + X.sum()
    b_n = b + (X.shape[0]-X.sum())
    posterior_mean = a_n / a_n + b_n
    delta.append(prior_mean - posterior_mean)
    return beta.pdf(mu_test,a_n,b_n)

# lets pick a random (uniform) point from the data
# and update our assumption with this
index = np.random.permutation(X.shape[0])

for i in range(0,X.shape[0]):
    y = posterior(a,b,X[index[:i]])
    plt.plot(mu_test,y,'r',alpha=0.3)

y = posterior(a,b,X)
ax.plot(range(0, 101), delta, 'r')
# plt.plot(mu_test,y,'b',linewidth=4.0)

plt.show()