#Program conducts the analytical solution the Burgers Equation

import numpy as np
import netCDF4 as netcdf

#Custom made modules
from BE_model import *

#Directory for the output
directory = '../Data/Burgers/'

#-----------------------------------------------------------------------------
#---------------------------MAIN SCRIPT STARTS HERE---------------------------
#-----------------------------------------------------------------------------	

nx          = 128 	#Number of grid points
stoch_iter  = 10000

mu          = 0.005
time_end    = 0.80
delta_t     = 0.00001

#-----------------------------------------------------------------------------

#Initiate the Burgers equation
BE = BE_model(nx)

#Generate the operators
BE.Operators(mu)

#Generate positive mass matrix
Mass = BE.Mass()

#Get the stochastic forcing field
stoch_for = BE.Stochastic_forcing()

#-----------------------------------------------------------------------------
time_all    = np.arange(0, time_end, delta_t)
theta		= 1	#Implicit = 0, Explicit = 1
jac_keep	= 10

x_all		= np.zeros((nx, stoch_iter))

#Get the initial conditions
for stoch_i in range(stoch_iter):
	x_all[:, stoch_i] = BE.x0

for time_i in range(len(time_all)):
	print(time_i)
	#Set tolarance to 1 for every time step
	tol		= 1.0

	#Get the stochastic forcing
	dW		= np.sqrt(delta_t) * np.random.normal(size = stoch_iter)
	stoch_time	= np.zeros((nx, stoch_iter))
	for stoch_i in range(stoch_iter): 
		stoch_time[:, stoch_i]	= stoch_for * dW[stoch_i]

	fix_part 	= (1.0 - theta) * delta_t * (np.matmul(-BE.Lap_op, x_all) + 0.5 * np.matmul(BE.L2, x_all * x_all)) - x_all - stoch_time

	while tol > 10**(-12.0):
		rhs	= x_all + theta * delta_t * (np.matmul(-BE.Lap_op, x_all) + 0.5 * np.matmul(BE.L2, x_all * x_all)) + fix_part
		#Determine new tolarance
		tol 	= np.linalg.norm(rhs, 'fro')

		#We only compute the Jacobian for the first stochastic solution, assuming that
		#the stochastic part is rather small wrt the average part. 
		#This hampers the Newton convergence a bit. 
		#We also keep the Jacobian and factorization fixed for jackeep time steps
		if time_i % jac_keep == 0:
			Jac	= Mass + theta * delta_t * (-BE.Lap_op + np.matmul(BE.L2, np.diag(x_all[:, 0])))

		#Now solve for dx
		dx	= np.linalg.solve(-Jac, rhs)
	
		#Update the current time step
		x_all	= x_all + dx

#-----------------------------------------------------------------------------

fh = netcdf.Dataset(directory+'BE_exact_end_solution_nx_'+str(nx)+'_mu_'+str(int(mu))+'_'+str(mu)[2:]+'.nc', 'w')

fh.createDimension('x', len(BE.x_grid))
fh.createDimension('stoch', stoch_iter)

fh.createVariable('x', float, ('x'), zlib=True)
fh.createVariable('stoch', float, ('stoch'), zlib=True)
fh.createVariable('BE_end', float, ('x', 'stoch'), zlib=True)

fh.variables['x'].longname		= 'Array of dimensionless longitudes'
fh.variables['stoch'].longname = 'Stochastic realisations'

#Writing data to correct variable	
fh.variables['x'][:] 			= BE.x_grid
fh.variables['stoch'][:] 		= np.arange(stoch_iter)+1
fh.variables['BE_end'][:] 		= x_all

fh.close()


#-----------------------------------------------------------------------------