#This function solves the ODEs and updates the coefficients
#This function is part of the DO method

import numpy as np
import scipy.linalg
from scipy.sparse import csc_matrix, linalg as sla
import time
import numpy.linalg as lin

#Custom made modules
from Norm_basis import *

def SODEsolve(Y, YY, Exps, jac, bilin, c, dt, nsteps):
    """Solves dY = (f+jac Y) dt + bilin * kron(Y, Y) + c * dW
    Under the assumption that YY = kron(Y, Y) of the actual Y"""
	
    #Get the dimensions
    mY, stoch_size	= len(Y), len(Y[0])
    
    if len(shape(c)) == 1: mc	= 1
    else: mc	= len(c[0])
    
    theta   = 1
    A	    = np.eye(mY) - theta * dt * jac
    tol	    = np.sqrt(np.max(Exps['YY'])) * 10**(-3.0)

    for i in range(int(nsteps)):
        #Get the stochastic part and make symmetric around 0
        dW		= np.sqrt(dt) * np.random.randn(mc, stoch_size)
        dW		= (dW - np.fliplr(dW)) / np.sqrt(2)

        #ODEFunction -> dY
        function_g	= lambda Yh: Yh - Y - theta * dt * ODEFunction(Yh, jac, bilin, np.matmul(c, dW) / dt)
        Yh		= Picard(Y, function_g, A, tol)
        Y		= (Yh - (1 - theta) * Y) / theta

    #Compute the expectations
    YY		= np.zeros((mY * mY, stoch_size))

    #Determine the covariance matrix and bilinearform
    for i in range(mY):
        for j in range(mY):
            YY[i * mY + j]	= Y[i] * Y[j]

    Exps['YY']	= np.sum(YY, axis = 1) / stoch_size
    Exps['dWY']	= nsteps * np.matmul(dW, Y.transpose()) / stoch_size
    if mc == 1: Exps['dWY']	= Exps['dWY'][0]
    Exps['YYY']	= np.matmul(YY, Y.transpose()) / stoch_size
 
    #Division below can be made more stable by make a QR
    invEYY		= np.linalg.pinv(np.array(Exps['YY'].reshape(mY, mY)))
    Exps['YYYDYY']	= np.matmul(Exps['YYY'], invEYY)
    Exps['dWYDYY']	= np.matmul(Exps['dWY'], invEYY)

    return Y, YY, Exps

def ODEFunction(y, jac, bilin, b):
    """Solving rhs of the following relation dY=V'*Jac(x)V*Y + V'*<V,V>(kron(Y,Y)+V' + c*dW"""
    mY, stoch_size	= len(y), len(y[0])
    yy		        = np.zeros((mY * mY, stoch_size))
    yy_ExpYY	    = np.zeros((mY * mY, stoch_size))
   
    #Determine the covariance matrix and bilinearform
    for i in range(mY):
        for j in range(mY):
            yy[i * mY + j]	= y[i] * y[j]
    
    ExpYY	= np.sum(yy, axis = 1) / stoch_size

    for iter_i in range(stoch_size):
        #Subtract mean
        yy_ExpYY[:, iter_i]	= yy[:, iter_i] - ExpYY

    return np.matmul(jac, y) + np.matmul(bilin, yy_ExpYY) + b

def SolveStochBasis(Mass, Jac, fun_BilinUser, V, VV, dt, Exps, c):
    """Solves [ Mass Mass*V ] [dV/dt]=[Jac(x)*V+{F(x,0)*EY'+<V,V>(Ekron(Y,Y)Y')+c*EdWY'/dt}/EYY']
    Update of the basis and the expectations"""
    
    if len(np.shape(c)) == 1: mc	= 1
    else: mc	= len(c[0])
    
    mV, nV	= len(V), len(V[0])
    A	= np.zeros((mV + nV, mV + nV))
    A[:mV,:mV]= Mass - dt * Jac
    A[mV:,:mV]= np.matmul(Mass, V).transpose()
    A[:mV,mV:]= np.matmul(Mass, V)
        
    #Tolerance
    tol	= 10**(-5.0)

    #Determine Jacobican
    JacV	= np.matmul(Jac, V)

    if mc > 1:
        cdWYDYY = np.matmul(c, Exps['dWYDYY'])

    else:
        cdWYDYY= np.zeros((len(c), len(Exps['dWYDYY'])))
       
        for i in range(len(cdWYDYY)):
            for j in range(len(cdWYDYY[0])):
                cdWYDYY[i, j]	= c[i] * Exps['dWYDYY'][j]  
    

    #Jac(x)*V+{F(x,0)*EY'+<V,V>(Ekron(Y,Y)Y')+c*EdWY'/dt}/EYY'
    function_VODE	= lambda Vh: np.concatenate(((JacV + np.matmul(fun_BilinUser(Vh,Vh), Exps['YYYDYY'])) * dt + cdWYDYY, np.zeros((nV, nV))), axis = 0)
    function_g		= lambda dV: np.matmul(A, dV) - function_VODE(V + dV[:mV])
    
    dV	= np.zeros((mV+nV, nV))
    dV	= Picard(dV, function_g, A, tol)

    #Update V, normalise and create orthogonal basis
    V	    = V + dV[:len(V)]
    V, R    = qrM(V, Mass)

    #Update Bilinear form
    VV	= fun_BilinUser(V,V)
    
    return V, VV, R

def Picard(y, function, A, tol):
    """Picard iteration to find y
    Assumption: A is currently a fixed matrix"""     

    #Propagate function
    f	       = function(y)
    norm_f	 = np.max([1, 2*tol])
    counter	 = 0
    max_counter	= 10

    while norm_f > tol and counter < max_counter:
        #Solve for y, dy = -U \ (L \ (P*f)) = -A^-1 * f
        #Determine P * f
        dy	     = np.linalg.solve(-A, f)
        y	     = y + dy
        f	     = function(y)
        norm_f 	= np.linalg.norm(f, 2)
        counter += 1

    if counter == max_counter:
        print('Warning: Picard stopped on max iterations, norm of f:', norm_f)

    return y
