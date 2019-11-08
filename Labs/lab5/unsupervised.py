import numpy as np
import matplotlib.pyplot as plt
from scipy.spatial.distance import cdist

# In supervised learning the believe takes the form of the preference over different hypothesis: we are trying to solve what the f(.) and x's are in 

#                       yi = f(x)

# One of the solutions is just the identity function, hence, x = y. however,
# our preference will tell us that this is not a good hypothesis...


# Gaussian likelihood: p(Y|F) = N(F, β−1I), where β−1 is the precision



def MLW(Y,q):
    v, W = np.linalg.eig(np.cov(Y.T))
    idx = np.argsort(np.real(v))[::-1][:q]
    return np.real(W[:,idx])