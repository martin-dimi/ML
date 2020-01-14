import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import norm

# the evidence is the distribution over the data space that is created if we average *all" of the possible hypothesis that we have relative to how probable we think that they are. P(D) = Integral(P(D|Thetha) * p(Thetha))

# Lets say that we have a modelling scenario where we have a Gaussian model, and we DO NOT know the mean nor the variance.

# now to make it simple we have an hypothesis space that only includes THREE different possible settings of the parameters. This would mean the evidence is an average over these three Gaussians as

x = np.linspace(-6,6,200)  
pdf1 = norm.pdf(x,0,1)  # Parameters 1
pdf2 = norm.pdf(x,1,3)  # Parameters 2
pdf3 = norm.pdf(x,-2.5,0.5) # Parameters 3

fig = plt.figure(figsize=(10,5))
ax = fig.add_subplot(111)

ax.plot(x,pdf1,color='r',alpha=0.5)
ax.fill_between(x,pdf1,color='r',alpha=0.3)

ax.plot(x,pdf2,color='g',alpha=0.5)
ax.fill_between(x,pdf2,color='g',alpha=0.3)

ax.plot(x,pdf3,color='b',alpha=0.5)
ax.fill_between(x,pdf3,color='b',alpha=0.3)

# p(θ = red) = 0.3, p(θ = green) = 0.2 and p(θ = blue) = 0.5 

pdf4 = 0.3*pdf1 + 0.2*pdf2 + 0.5*pdf3

ax.plot(x, pdf4, color='k', alpha=0.8, linewidth=3.0, linestyle='--')
ax.fill_between(x, pdf4, color='k', alpha=0.5)

fig.show()
