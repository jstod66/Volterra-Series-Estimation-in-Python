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
