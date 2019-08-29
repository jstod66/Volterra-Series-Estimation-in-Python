import numpy as np
import matplotlib.pyplot as mp
from mpl_toolkits.mplot3d import Axes3D
import math
import scipy.signal as sig
import scipy.optimize as opt
from SecondOrderStuff import *
from mpl_toolkits.mplot3d import Axes3D # This import has side effects required for the kwarg projection='3d' in the call to fig.add_subplot

DataRatio = 3;
TOL = 0.02;
n = 60;  #size of the kernels
n1 = n;
n2 = (n**2+n)/2;

#---Define a linear system---

#Georgios system
b1 = np.array([0,0.501]);
a1 = np.array([1,-1.8036,0.8338]);
 
b2 = np.array([0,0.7515]);
a2 = a1;

#noise_std = 35.45
noise_std = 30

#---Generate test input and noisy output --------
print('Generating Data')

N = int(math.floor(3*(n+n2)));   #amount of data compared to parameters

u_in = np.random.normal(0,1,N);

##mp.figure()
##mp.plot(range(len(u_in)),u_in)
##mp.show()

y0 = sig.lfilter(b1,a1,u_in) + np.square(sig.lfilter(b2,a2,u_in))

e = np.random.normal(0,noise_std,N);
y = y0+e

##mp.figure()
##mp.plot(range(len(y)),y)
##mp.show()

#---Formulate LS problem--------
print('Formulating LS Matrices')

Y = y[n-1:]
PHI_1N = np.zeros((n1,(N-n+1)));

for k in np.arange(n,N+1):
    inputpiece = u_in[k-n:k]
    PHI_1N[:,k-n] = inputpiece[::-1]     #the :-1 reverses vector order

PHI_12N = np.zeros((n2,N-n+1));
row_count = 0;

for count in np.arange(0,n):
    j = n-count;
    p12 = np.zeros((j,N-n+1))
    for k in np.arange(n,N+1):
        inputpiece = u_in[k-j:k]
        p12[:,k-n] = inputpiece[::-1]
    PHI_12N[row_count:row_count+j,:] = p12;
    row_count = row_count + j;

PHI_22N = np.zeros((n2,N-n+1));
row_count = 0;

for j in np.arange(0,n):
    p22 = np.zeros((n-j,N-n+1));
    for k in np.arange(n,N+1):
        inputpiece = u_in[k-n:k-j]
        p22[:,k-n] = inputpiece[::-1]
    if j==0:
        PHI_22N[row_count:row_count+n-j,:] = p22;
    else:
        PHI_22N[row_count:row_count+n-j,:] = 2*p22;
    row_count = row_count + n-j;

PHI_2N = np.multiply(PHI_12N,PHI_22N)

PHI = np.concatenate((PHI_1N,PHI_2N))

THETA_LS = np.linalg.lstsq(np.matrix.transpose(PHI),Y,rcond=-1)[0]

h1_LS = THETA_LS[0:n]
h2_LS = THETA_LS[n:]
h2_LS = arrangeKernel2D(h2_LS,n)

mp.figure()
mp.plot(np.arange(0,len(h1_LS)),h1_LS)
##mp.show()

fig = mp.figure()
ax = fig.add_subplot(111, projection='3d')
x = y = np.arange(0,len(h2_LS))
X, Y = np.meshgrid(x, y)
ax.plot_surface(X, Y, h2_LS)
ax.set_xlabel(r'$\tau_1$')
ax.set_ylabel(r'$\tau_2$')
ax.set_zlabel(r'$h_2(\tau_1,\tau_2)$')

mp.show()

#------------REGULARIZATION-------------------------
##print('Optimizing regularization hyperparameters')
##v_coords, u_coords = generate_coords(n,n2)
##
##c10 = 1;
##lambda10 = 0.8;
##c20 = 1;
##lambda20 = 0.8;
##lambda30 = 0.8;
##std0 = noise_std*1.5;
##
##bounds = opt.Bounds([0, 0.3, 0, 0.3, 0.3, 0],[np.inf, 1, np.inf, 1, 1, np.inf])
##x0 = np.array([c10, lambda10, c20, lambda20, lambda30, std0])
##
##res = opt.minimize(TCfunc, x0, args = (PHI,Y,n,v_coords,u_coords), method = 'trust-constr', bounds=bounds)