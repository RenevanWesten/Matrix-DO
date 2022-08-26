#Script to normalise and generate orthogonal base
#This function is part of the DO method

import numpy as np

def nrM(v, M):
    """Determine the norm"""
    norm_v	= np.sqrt(np.matmul(v, np.matmul(M, v)))
    return norm_v

def qrM(V, M):
    """Genenerate orthogonal base"""
    
    nrm	= nrM(V[:, 0], M)
    R	= np.zeros((len(V[0]), len(V[0])))
    R[0, 0]	= nrm
    V[:, 0]	= V[:, 0] / nrm
      
    for i in range(1, len(V[0])):
        h	= np.zeros(i)
        for k in range(i):
            #Modified Gramm-Schmid
            h[k]	= np.matmul(V[:, k], np.matmul(M, V[:, i]))
            V[:, i]	= V[:, i] - V[:, k] * h[k]

        nrm		        = nrM(V[:, i], M)
        V[:, i]		    = V[:, i] / nrm
        R[:len(h), i]	= h
        R[len(h), i]	= nrm
      
    return V, R

