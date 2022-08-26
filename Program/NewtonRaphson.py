#Script to solve set of non-linear equations using Newton-Raphson method
#This function is part of the DO method

from pylab import *
import numpy as np
from scipy.sparse import csc_matrix
import scipy.linalg

def NewtonRaphson(function, x_0, mask_index_x, options = False):
    """function is a function handle that returns a vector of residuals equations, F,
    and takes a vector (x) as its only argument.
    
    Optionally, fnction may return the Jacobian, J_{ij} = dF_i / dx_j, as an additional output.
    
    if function only returns one output, then J is estimated using a center difference approximation:
    J_{ij} = dF_i / dx_j = (F_i(x_j + dx) - F_i(x_j - dx)) / 2 / dx
    Options is a structure of solver options, e.g. options = {'Tol_x = ..., Tol_Fun = ..., Max_iter = ...}"""

    if options == False:
        #Get the default parameters
        options	= {}
        options['Tol_x']	= 10**(-12.0)
        options['Tol_Fun']	= 10**(-8.0)
        options['Max_iter']	= 100

    #Wrap the function so it always returns J
    FUN	= lambda x: FunctionWrapper(function, x)

    #Get the (default) parameters	
    tol_x	    = options['Tol_x']	#Relative max step tolerance
    tol_fun	    = options['Tol_Fun']	#Funciton tolerance
    max_iter	= options['Max_iter']	#Max number of iterations
    typ_x	    = np.maximum(abs(x_0), 1)#x scaling values, removes zeros
    alpha	    = 10**(-4.0)		#Criteria for decrease
    min_lambda	= 0.1			#Min lambda
    max_lambda	= 0.5			#Max lambda

    #Set scaling values
    weight	= np.ones(len(x_0))
    J0		= np.zeros((len(typ_x), len(weight)))
    for i in range(len(J0)):
        J0[i]	= weight[i] * (1.0 / typ_x)
            
    #Check initial guess and scale jacobian
    x_0         = np.array(x_0)
    x	 	    = np.copy(x_0)
    F, J, jac	= FUN(x)
    J_star 	    = J / J0

    if np.any(np.isnan(J_star) == True) or np.any(np.isinf(J_star) == True):
        #Matrix may be singular
        exit_flag	= -1
    else:
        #Normal exist
        exit_flag	= 1
	
    #Calculate the norm of the residuals
    res_norm	= np.linalg.norm(F, 2)
    dx		= np.zeros(np.shape(x_0))
    
    #Solve for x
    counter_iter	= 0
    lambda_value	= 1	#Backtracking
	
    while (res_norm > tol_fun or lambda_value < 1) and exit_flag >= 0 and counter_iter <= max_iter:
        if lambda_value == 1:
            #Newton-Raphson solver
            counter_iter += 1
            #dx_star	= np.linalg.solve(-J_star, F)		#Calculate Newton step
            dx_star	= scipy.sparse.linalg.spsolve(csc_matrix(-J_star), F)		#Calculate Newton step
            dx		= np.array(dx_star * typ_x)		#Rescale x
            g	 	= np.matmul(F, J_star)			#Gradient of resnorm
            slope	= np.sum(g * dx_star)			#Slope of gradient
            f_old	= np.sum(F * F)				#Objective function
            x_old	= np.copy(x)				#Initial value		
            lambda_min	= tol_x / np.max(abs(dx) / np.maximum(abs(x_old), 1))
  	
        print('Iteration step '+str(counter_iter)+' of '+str(max_iter))

        if lambda_value < lambda_min:
            #x is too close to x_old
            exit_flag = 2
            break

        elif np.any(np.isnan(dx) == True) or np.any(np.isinf(dx) == True):
            #Martix may be singular
            exit_flag = -1
            break

        #Next guess and evaluate next residuals
        x		= x_old + dx * lambda_value
        x[mask_index_x] = 0.0
        F, J, jac	= FUN(x)

        #Scale next Jacobian and detetermine new objective function
        J_star      = J / J0
        f           = np.sum(F * F)	

        #Check for convergences
        lambda_1	= np.copy(lambda_value)

        if f > f_old + alpha * lambda_value * slope:
            if lambda_value == 1:
                #Calculate lambda
                lambda_value = -slope / 2 / (f - f_old - slope)

            else:
                A       = 1.0 / (lambda_1 - lambda_2)
                B       = np.asarray([[1.0 / lambda_1**2.0, -1.0 / lambda_2**2.0], [-lambda_2 / lambda_1**2.0, lambda_1 / lambda_2**2.0]])	
                C       = np.asarray([f - f_old - lambda_1 * slope, f_2 - f_old - lambda_2 * slope])
                coeff   =  np.matmul(A * B, C)
                a, b    = coeff[0], coeff[1]

                if a == 0:
                    lambda_value = -slope / 2 / b
                else:
                    discriminant = b**2.0 - 3 * a * slope	

                    if discriminant < 0:
                        lambda_value = max_lambda * lambda_1
                    elif b <= 0:
                        lambda_value = (-b + np.sqrt(discriminant)) / 3 / a
                    else:
                        lambda_value = -slope / (b + np.sqrt(discriminant))

                #Minimum step length
                lambda_value = min(lambda_value, max_lambda * lambda_1)

        elif np.isnan(f) == True or np.isinf(f) == True:
            #Limit undefined evaluation or overflow
            lambda_value = max_lambda * lambda_1
        else:
            #Fraction of  newton step
            lambda_value = 1

        if lambda_value < 1:
            #Save second most from previous
            lambda_2	    = np.copy(lambda_1)
            f_2		        = np.copy(f)
            lambda_value	= np.max([lambda_value, min_lambda * lambda_1])
	
        #Calculate residual norm
        res_norm	= np.linalg.norm(F, 2)

        if np.any(np.isnan(J_star) == True) or np.any(np.isinf(J_star) == True):
            #Matrix may be singular
            exit_flag	= -1
            break
            
    #Save the final output
    output		= {}
    output['iterations']= counter_iter
    output['stepsize']	= dx
    output['lambda']	= lambda_value

    if counter_iter >= max_iter:
        exit_flag = 0
        output['message'] = 'Number of iterations exceeded'
    elif exit_flag == 2:
        output['message'] = 'May have converged, but x is too close to x_old'
    elif exit_flag == -1:
        output['message'] = 'Matrix may be singular, step was Nan or Inf'
    else:
        output['message'] = 'Normal exit'

    return x, res_norm, F, exit_flag, output, jac

def FunctionWrapper(function, x):
    #Get the Jacobians
    F, J, jac	= function(x)
    
    #F needs to be a column vector
    return F, J, jac

def rhs_JacBE(Ffun, Jfun, M, x, x_0, dt):
    """Computes the function value of Backward Euler method and its Jacobian
    Jfun(X) gives the Jacobian matrix of Ffun(x)"""
    jac	= Jfun(x)
    JBE	= M - (dt * jac)
    FBE	= np.matmul(M, x - x_0) - dt * Ffun(x)
    
    return FBE, JBE, jac

