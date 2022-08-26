#Program conducts Dynamically Orthogonal (DO) field equations for stochastic dynamical systems
#Original (Matlab) script received from Fred Wubs
#This script contains the main DO frame work

import netCDF4 as netcdf
import numpy as np
import os

#Custom made modules
from BE_model import *
from DO_nonlin import *

#Directory for the output
directory = '../Data/Burgers/'

#-----------------------------------------------------------------------------
#---------------------------MAIN SCRIPT STARTS HERE---------------------------
#-----------------------------------------------------------------------------	

nx          = 128 #Number of grid points
nV          = 32 #Number of DO modes
stoch_iter  = 1000

mu          = 0.005
time_end    = 0.8
delta_t     = 0.001

#-----------------------------------------------------------------------------

#Initiate the Burgers equation
BE = BE_model(nx)

#Generate the operators
BE.Operators(mu)

#Generate positive mass matrix
Mass = BE.Mass()

#Get the stochastic forcing field
stoch_for = BE.Stochastic_forcing()

#Now obtain the functions
fun_rhs	= lambda X, b: BE.RHS(X, b)
fun_jac	= lambda X: BE.Jacobian(X)
fun_bil = lambda X, Y: BE.Bilin(X, Y)

#-----------------------------------------------------------------------------
#Filename of output 
filename = directory+'BE_model_nx_'+str(nx)+'_mu_'+str(int(mu))+'_'+str(mu)[2:]+'_nV_'+str(nV)+'.nc'

#Generate empty arrays for DO
x          = BE.x0
V          = np.random.rand(BE.nx, nV)
V[:, 0]    = stoch_for
Y          = np.zeros((nV, stoch_iter))

#Generate dictionary for the integration parameters
integr_param            = {}
integr_param['T']       = time_end
integr_param['dt']      = delta_t
integr_param['ndtsub']  = 1

time_all, x_all, V_all, Y_all, norm_x, norm_V = DONonLin(x, V, Y, Mass, stoch_for, fun_rhs, fun_jac, fun_bil, integr_param, [])

#-----------------------------------------------------------------------------
#-----------------------------------------------------------------------------
#-----------------------------------------------------------------------------

#Now sort the basis, from high variance to low variance

for time_i in range(len(time_all)):
	#Reverse the time step (at the beginning the solution changes a lot for ordering, see below)
	#We need every time step to make sure no flip occurs in signs (but can be removed at the end)
	time_i	= len(time_all) - time_i - 1
	print(time_i)

	#First determine the highest variance for the last time frame
	#Note that you can reverse the basis by a factor -1, therefore we need to use the previous time step as well
	D, V	= np.linalg.eig(np.matmul(Y_all[time_i], Y_all[time_i].transpose()))

	#Sort on eigenvalue, this one is now fixed over time
	sorted_index    = np.argsort(D)[::-1]
	D		= D[sorted_index]
	V		= V[:, sorted_index]

	#Now check whether each column has still the same sign in the sorted covariance matrix
	#We use the maximum value of each column (because the sign can not abruptly change (this is gradual))
	max_index	= np.argmax(abs(V), axis = 0)

	if time_i == len(time_all) - 1:
		#First time, make previous array for sign and index
		sign_prev	= np.zeros(len(max_index))
		max_index_prev	= np.copy(max_index)
		for i in range(len(max_index)):
			sign_prev[i]	= np.sign(V[max_index[i], i])

	#Now check previous sign using the previous locations, these values do not have to be the current maximum
	for i in range(len(max_index)):
		sign_current	= np.sign(V[max_index_prev[i], i])

		if sign_current != sign_prev[i]:
			#Sign was flipped, change sign of covariance array
			V[:, i] = -V[:, i]

		#Now overwrite the previous sign and index
		sign_prev[i]		= np.sign(V[max_index[i], i])
		max_index_prev[i]	= max_index[i]

	#Multiply for the correct basis
	V_sort	= np.matmul(V_all[time_i], V)
	Y_sort	= np.matmul(V.transpose(), Y_all[time_i])
	Y_var	= D / stoch_iter

	Y_all[time_i]	= Y_sort
	V_all[time_i]	= V_sort
	norm_V[time_i]	= Y_var

#Reduce the number of output written to file
time_step	= 25
time_all	= time_all[::time_step]
x_all		= x_all[::time_step]
V_all		= V_all[::time_step]
Y_all		= Y_all[::time_step]
norm_x		= norm_x[::time_step]	
norm_V		= norm_V[::time_step]

#-----------------------------------------------------------------------------
#-----------------------------------------------------------------------------
#-----------------------------------------------------------------------------

fh = netcdf.Dataset(filename, 'w')

fh.createDimension('time', len(time_all))
fh.createDimension('x', len(BE.x_grid))
fh.createDimension('nV', nV)
fh.createDimension('nY', stoch_iter)

fh.createVariable('time', float, ('time'), zlib=True)
fh.createVariable('x', float, ('x'), zlib=True)
fh.createVariable('nV', float, ('nV'), zlib=True)
fh.createVariable('nY', float, ('nY'), zlib=True)
fh.createVariable('Mean', float, ('time', 'x'), zlib=True)
fh.createVariable('V', float, ('time', 'x', 'nV'), zlib=True)
fh.createVariable('Y', float, ('time', 'nV', 'nY'), zlib=True)
fh.createVariable('norm_mean', float, ('time'), zlib=True)
fh.createVariable('norm_V', float, ('time', 'nV'), zlib=True)

fh.variables['time'].longname		= 'Array of dimensionless time'
fh.variables['x'].longname		    = 'Array of dimensionless x'
fh.variables['nV'].longname		    = 'Number of DO components'
fh.variables['nY'].longname		    = 'Stochastic realisations'
fh.variables['Mean'].longname		= 'DO mean'
fh.variables['V'].longname		= 'DO components'
fh.variables['norm_mean'].longname	= 'Variance of mean component'
fh.variables['norm_V'].longname		= 'Variance of DO components'

#Writing data to correct variable	
fh.variables['time'][:]     	= time_all
fh.variables['x'][:] 		= BE.x_grid
fh.variables['nV'][:] 		= np.arange(nV)+1
fh.variables['nY'][:] 		= np.arange(stoch_iter)+1
fh.variables['Mean'][:] 	= x_all
fh.variables['V'][:] 		= V_all
fh.variables['Y'][:] 		= Y_all
fh.variables['norm_mean'][:] 	= norm_x
fh.variables['norm_V'][:] 	= norm_V

fh.close()
