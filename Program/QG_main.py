#Program conducts Dynamically Orthogonal (DO) field equations for stochastic dynamical systems
#Original (Matlab) script received from Fred Wubs
#This script contains the main DO frame work

from pylab import *
import numpy as np
import os

#Custom made modules
from QG_model import *
from DO_nonlin import *

#Directory for the output
directory = '../Data/QG/'

#-----------------------------------------------------------------------------
#---------------------------MAIN SCRIPT STARTS HERE---------------------------
#-----------------------------------------------------------------------------	

nx          = 64        #Number of zonal dimensions
ny          = 64        #Number of meridional dimensions
nV          = 4         #Number of DO modes
stoch_iter  = 1000      #Number of realisations    
time_end    = 1.0       #End time (T)
delta_t     = 0.001     #Time step
stoch_amp   = 10**(0)   #Stochastic amplitude

#-----------------------------------------------------------------------------

#Initiate the QG model
QG = QG_model(nx, ny)

#Set new Reynolds number in QG model
QG.Set_parameters(4, 40)

#Set new wind stress field in QG model
QG.Set_parameters(10, 1)

#Generate positive mass matrix
Mass = QG.Mass()

#Get the stochastic forcing field
stoch_for = QG.Stochastic_forcing(stoch_amp)

#Now obtain the functions
fun_rhs	= lambda X, b: b - QG.RHS(X)
fun_jac	= lambda X: QG.Jacobian(X, 0.0)
fun_bil	= lambda X, Y: QG.Bilin(X, Y)

#-----------------------------------------------------------------------------
#Filename of output 
filename = directory+'QG_model_nx_'+str(nx)+'_ny_'+str(ny)+'_dt_'+str(int(delta_t))+'_'+str(delta_t)[2:]+'_nV_'+str(nV)+'_Re_'+str(int(QG.parameters[4]))+'_Stoch_amp_'+str("{:.0e}".format(stoch_amp))+'.nc'

if os.path.exists(filename):
    #Check whether output already exists, continue DO with the latest data
    time_prev, x_prev, V_prev, Y_prev, norm_x_prev, norm_V_prev = QG.ReadOutput(filename)

    #Get the latest time step for x, V and Y
    x, V, Y    = x_prev[-1], V_prev[-1], Y_prev[-1]

else:
    #Generate empty arrays for DO
    x          = np.zeros(QG.ndim)
    V          = np.random.randn(QG.ndim, nV)
    V[:, :2]   = stoch_for
    Y          = np.zeros((nV, stoch_iter))

#Generate dictionary for the integration parameters
integr_param		 = {}
integr_param['T']	 = time_end
integr_param['dt']	 = delta_t
integr_param['ndtsub']	 = 1

#Set the boundaries of the DO mean manually to 0.0 after each time step (otherwise converges of solution)
#This gives simply the indices at the boundaries
mask_index_x = np.where(np.sum(np.matmul(fun_jac(x), stoch_for), axis = 1) == 0)[0]

#Run DO
time_all, x_all, V_all, Y_all, norm_x, norm_V = DONonLin(x, V, Y, Mass, stoch_for, fun_rhs, fun_jac, fun_bil, integr_param, mask_index_x)
#-----------------------------------------------------------------------------

if os.path.exists(filename):
    #Check whether output already exists, add previous files
    time_all = np.concatenate((time_prev, time_all + time_prev[-1] + delta_t), axis = 0)
    x_all    = np.concatenate((x_prev, x_all), axis = 0)
    V_all    = np.concatenate((V_prev, V_all), axis = 0)
    Y_all    = np.concatenate((Y_prev, Y_all), axis = 0)
    norm_x   = np.concatenate((norm_x_prev, norm_x), axis = 0)
    norm_V   = np.concatenate((norm_V_prev, norm_V), axis = 0)

#Write data to NETCDF output
QG.WriteOutput(filename, nV, time_all, x_all, V_all, Y_all, norm_x, norm_V)
