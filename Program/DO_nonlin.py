#This function evolves the DO solution over time
#This function is part of the DO method

from pylab import *
import numpy as np

#Custom made modules
from Solve_ODE_basis import *
from Norm_basis import *
from NewtonRaphson import *

def DONonLin(x, V, Y, Mass, c, fun_rhsUser, fun_jacUser, fun_BilinUser, integr_param, mask_index_x):
    """Solves Mass*dX = F(X,dW) = (L*X+<X,X>)*dt+c*dW 
    By replacing X = x + V*Y where x is constant of the current time step
    At the begining, EY = 0 and is updated to x = x + V*EY and Y = Y - EY
    Finally this leads to the following three equations:
    Mass dx / dt = F(x,0) + <V,V>Ekron(Y,Y)
    dY=  V'*Jac(x)V*Y + V'*<V,V>(kron(Y,Y)+V'*c*dW
    [ Mass Mass*V ] [dV/dt]=[Jac(x)*V+{F(x,0)*EY'+<V,V>(Ekron(Y,Y)Y')+c*EdWY'/dt}/EYY']
    [ V'*Mass   O ] [ blub] [             O                               ]"""

    #Orhogonalise the stochastic basis
    V, R	= qrM(V, Mass)

    #Integration paramaters
    T       = integr_param['T']
    dt      = integr_param['dt']
    ndtsub  = integr_param['ndtsub']
    
    #Constant
    dtsub	= dt / ndtsub

    #Generate covariance matrix
    mV, nV		= len(V), len(V[0])
    mY, stoch_size	= len(Y), len(Y[0])
    YY		        = np.zeros((mY * mY, stoch_size))

    #Determine the covariance matrix and bilinearform
    for i in range(mY):
        for j in range(mY):
            YY[i * mY + j]	= Y[i] * Y[j]

    #Generate dictionary for the expectance for Y and <Y, Y> = YY
    Exps		= {}
    Exps['YY']	= np.sum(YY, axis = 1) / stoch_size	
    VV		    = fun_BilinUser(V, V)
    Jac		    = fun_jacUser(x)
    
    time_all    = np.zeros(int(T/dt)+1)
    x_all       = np.zeros((len(time_all), len(x)))
    V_all       = np.zeros((len(time_all), mV, nV))
    Y_all       = np.zeros((len(time_all), mY, stoch_size))
    norm_x      = np.zeros(len(time_all))
    norm_V      = np.zeros((len(time_all), mY))

    for time_i in range(len(time_all)):
        #Time loop, propagate mean, V and Y
        print('-----------------------------------------------')
        print('Time step '+str(time_i+1)+' of '+str(len(time_all)))
        time_all[time_i] = time_i * dt
 	
        #Determine the reduced stochastic ODE elements
        VJacV	= np.matmul(V.transpose(), np.matmul(Jac, V))
        VVV	    = np.matmul(V.transpose(), VV)
        Vc	    = np.matmul(V.transpose(), c)

        #Reshape dimensions for Vc (if 1 dimensional)
        if len(np.shape(Vc)) == 1: 
            Vc = Vc.reshape((len(Vc), 1))
        
        #Determine and update Y, YY
        print('\nDetermine stochastic coefficients...')
        Y, YY, Exps	= SODEsolve(Y, YY, Exps, VJacV, VVV, Vc, dtsub, ndtsub)
        print('Stochastic coefficients finished \n')

        #Determine and update V, VV
        print('Determine stochastic basis...')
        V, VV, R	= SolveStochBasis(Mass, Jac, fun_BilinUser, V, VV, dt, Exps, c)
        print('Stochastic basis finished \n')

        #Adapt Y and the covariance matrix based on the change in V
        Y		= np.matmul(R, Y)
        Exps['YY']	= np.reshape(np.matmul(np.matmul(R, np.reshape(Exps['YY'], (nV, nV))), R.transpose()), nV * nV)
       
        #Determine the eigenvalues
        w, q            = np.linalg.eig(np.array(np.reshape(Exps['YY'], (nV, nV)), dtype = np.float))
        w               = np.sort(w)[::-1]
        norm_V[time_i]  = w
        
        #Propogate the mean (x): Mass dx/dt = F(x,0) + <V,V>Ekron(Y,Y)
        function_1	= lambda xh: fun_rhsUser(xh, np.matmul(VV, Exps['YY']))
        function_2	= lambda xh: fun_jacUser(xh)
        function_3	= lambda xh: rhs_JacBE(function_1, function_2, Mass, xh, x, dt)

        #Determine the new mean and update the Jacobian	 of mean state
        print('Newton iteration started')
        x, res, test1, flg, test2, Jac	= NewtonRaphson(function_3, x, mask_index_x)
        norm_x[time_i]			        = np.linalg.norm(x, 2)
        print('Newton iteration finished \n')

        #Save all the time series (mean, V and Y)
        x_all[time_i]	= x
        V_all[time_i]	= V
        Y_all[time_i]	= Y
        

    return time_all, x_all, V_all, Y_all, norm_x, norm_V

