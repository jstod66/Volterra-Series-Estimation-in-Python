def arrangeKernel2D(x, n):

    import numpy as np
    
    h2 = np.zeros((n,n))

    for diagonal in np.arange(0,n):
        offset = diagonal
        entries = n-diagonal
        for i in np.arange(0,entries):
            h2[i,i+offset] = x[i]
            if diagonal != 0:
                h2[i+offset,i] = x[i]
        x = x[entries:]

    return h2

def generate_coords(n, n2):

    import numpy as np
    
    counter = 0
    v_coords = np.zeros(n2)

    for start_point in np.arange(0,n):
        end_point = 2*n - start_point - 1
        for i in range(start_point,end_point+1,2):
            v_coords[counter] = i
            counter = counter+1

    val = 0
    counter = 0
    u_coords = np.zeros(n2)

    for start_point in  range(1,n+1):
        for i in range(1,n-start_point+2):
            u_coords[counter] = val
            counter = counter+1
        val = val+1
    

    return v_coords, u_coords

def TCfunc(x, PHI, Y, n, v_coords, u_coords):

    import numpy as np
    import scipy as sc
    import time

    c1 = x[0]
    lambda1 = x[1]
    c2 = x[2]
    lambda2 = x[3]
    lambda3 = x[4]
    noise_var = x[5]**2

    
    #---------Form P (prior covariance)--------------
    P1 = np.zeros((n,n));
    for j in range(0,n):
        for k in range(0,n):
            P1[j,k] = c1*(lambda1**(max(j,k)))
            
    n2 = (n**2+n)/2;
    P2v = np.zeros((n2,n2));
    P2u = np.zeros((n2,n2));

    for j in range(0,n2):
        for k in range(j,n2):
            P2v[j,k] = lambda2**(max(v_coords[j],v_coords[k]))
            P2v[k,j] = P2v[j,k]
            P2u[j,k] = lambda3**(max(u_coords[j],u_coords[k]))
            P2u[k,j] = P2u[j,k]
    
    P2 = c2*np.multiply(P2v,P2u)
    
    P = sc.linalg.block_diag(P1,P2)
    
    #-----------------------------------------------(1 sec vast majority in nested for)

##    Rd = np.linalg.qr(np.c_[np.matrix.transpose(PHI),Y])
    SIGY = np.matmul(np.matrix.transpose(PHI),P) #0.2 secs @n=40

    SIGY = np.matmul(SIGY,PHI)#0.7 secs
    for i in range(0,len(SIGY)):
        SIGY[i,i] = SIGY[i,i] + noise_var
    
    (sign,logdet) = np.linalg.slogdet(SIGY) #0.75 secs
    
    temp = np.linalg.solve(SIGY,Y) 
    J = np.matmul(np.matrix.transpose(Y),temp) + logdet #0.8 secs
    
    
    print(J)
    
    
    
    return J 

    

