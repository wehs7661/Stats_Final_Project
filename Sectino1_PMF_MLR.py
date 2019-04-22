import numpy as np
import math
import random
import scipy.stats
import matplotlib.pyplot as plt

# Constants.
kB = 1.381e-23 * 6.022e23 / 1000.0 # Boltzmann constant in kJ/mol/K

temperature = 300 
N = 47 # number of umbrellas
M = 500 # number of data points for each umbrella 
T_k = np.ones(N,float)*temperature # inital temperatures are all equal 
beta = 1.0 / (kB * temperature) # inverse temperature of simulations (in 1/(kJ/mol)) 
d_min, d_max = 0.25, 1.00 # min/max for PMF (nm)

# Allocate storage for simulation data
#N_k = np.zeros([N], dtype = int) # N_k[k] is the number of snapshots from umbrella simulation k
k = np.zeros([N]) # k[n] is the spring constant (in kJ/mol/nm**2) for umbrella simulation n
mu= np.zeros([N]) # mu_kn[k,n] is the center location (in nm) for the n-th umbrella window (all k points have same mu)
mu_kn = np.zeros([N,M]) # mu_kn[k,n] is the center location (in nm) for the n-th umbrella window (all k points have same mu)
x_kn = np.zeros([N,M]) # d_kn[k,n] is the m-th data point of ion-pair distance (in nm) from the n-th umbrella simulation

infile = open('Section 1/centers.dat', 'r')
lines = infile.readlines()
infile.close()
i=0
for n in range(N):
    # Parse line k.
    line = lines[n]
    tokens = line.split()
    mu[n] = float(tokens[0]) # spring center locatiomn (in nm), 1st column in centers.dat
    mu_kn[i,:] = mu[n]*np.ones([1,M])
    i+=1
mu_kn = mu_kn.reshape(1, M*N)

for n in range(N):
    # Read ion-pair distance data.
    filename = 'Section 1/pullx-umbrella%d.xvg' % n
    infile = open(filename, 'r')
    lines = infile.readlines()
    infile.close()
    # Parse data.
    i = 0
    j = 0
    for line in lines:
        if line[0] == '#' or line[0] == '@':
            j +=1 #number of parameter lines

    for line in lines[j:j+M]: #read in data starting from (j+1)-th line and read in M lines in total
        if line[0] != '#' and line[0] != '@':
            tokens = line.split()
            d = float(tokens[1]) # ion-pair distance
            x_kn[n,i] = d
            
            i += 1  
x_kn = x_kn.reshape(1, M*N)

def g(x, mu, sigma):
    return np.exp(-(x-mu)**2/(2*sigma**2))

M1 = np.zeros([M*N, N])
M2 = np.zeros([M*N, N])
for i in range(M*N):
    for j in range(N):
        M1[i,j] = g(x_kn[0][i], mu[j], 1)
        M2[i,j] = g(mu_kn[0][i], mu[j], 1)
M = M1 - M2