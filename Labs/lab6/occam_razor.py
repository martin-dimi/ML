import itertools as it
import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import norm

def generate_data(N=3):
    D2 = np.array(list(it.product([-1,1],repeat=N*N)))
    D = []
    for i in range(0,len(D2)):
        d = D2[i]
        D.append(d.reshape([N,N]))
        x = []
    for i in range(-(N-2),N-1):
        for j in range(-(N-2),N-1):
            x.append(np.array([float(i),float(j)]))
    return (D,x)


def model0(theta,x,y):
    return 1.0/(pow(2.,len(x)))


def model1(theta,x,y):
    model = 1.0
    for i in range(len(x)):
        model *= 1.0/(1+np.exp(-y[i]*theta[0]*x[i][0]))
    return model


def model2(theta,x,y):
    model = 1.0
    for i in range(len(x)):
        model *= 1.0/(1+np.exp(-y[i]*(theta[0]*x[i][0]+theta[1]*x[i][1])))
    return model


def model3(theta,x,y):
    model = 1.0
    for i in range(len(x)):
        model *= 1.0/(1+np.exp(-y[i]*(theta[0]*x[i][0]+theta[1]*x[i][1]+theta[2])))
    return model

#generate the parameter vector for the samples
def generate_parameters(N,d,mu,sigma):
    return sigma*np.random.randn(N,d)+mu

# generate the evidence
def compute_evidence(y,x,theta,model):
    evidence = 0.0
    for i in range(len(theta)):
        evidence += model(theta[i],x,y)
    return evidence/len(theta)

N = 3
nr_samples = pow(10,2)
sigma = pow(10,1.5)
mu = 0
[D, x] = generate_data(N)
theta = generate_parameters(nr_samples,3,mu,sigma)

evidence = np.zeros([4,len(D)])

for i in range(len(D)):
    evidence[0,i] = compute_evidence(D[i].ravel(),x,theta,model0)
    evidence[1,i] = compute_evidence(D[i].ravel(),x,theta,model1)
    evidence[2,i] = compute_evidence(D[i].ravel(),x,theta,model2)
    evidence[3,i] = compute_evidence(D[i].ravel(),x,theta,model3)

fig = plt.figure(figsize=(10,5))
ax = fig.add_subplot(111)

index1 = np.argsort(evidence[1,:])
index2 = np.argsort(evidence[2,:])
index3 = np.argsort(evidence[3,:])

ax.plot(evidence[1,index1], 'b')
ax.plot(evidence[2,index2], 'g')
ax.plot(evidence[3,index3], 'k')


# index4 = np.argsort(evidence[1,:])
# ax.plot(evidence[0,index], 'r')
# ax.plot(evidence[1,index], 'g')
# ax.plot(evidence[3,index], 'k')

fig.show()