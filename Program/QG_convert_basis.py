#Program sorts the basis in the correct way (high to low variance)
#One can also decrease the output of the original

from pylab import *
import numpy as np
import os

#Custom made modules
from QG_model import *

#Directory for the output
directory = '../Data/QG/'

#-----------------------------------------------------------------------------
#---------------------------MAIN SCRIPT STARTS HERE---------------------------
#-----------------------------------------------------------------------------	

nx          = 64 #Number of zonal dimensions
ny          = 64 #Number of meridional dimensions
nV          = 4	 #Number of DO modes
Re	        = 40
delta_t     = 0.001
stoch_amp   = 10**(0)

#-----------------------------------------------------------------------------
#Initiate the QG model
QG = QG_model(nx, ny)

filename = directory+'QG_model_nx_'+str(nx)+'_ny_'+str(ny)+'_dt_'+str(int(delta_t))+'_'+str(delta_t)[2:]+'_nV_'+str(nV)+'_Re_'+str(Re)+'_Stoch_amp_'+str("{:.0e}".format(stoch_amp))+'.nc'

time, x_all, V_all, Y_all, norm_x, norm_V = QG.ReadOutput(filename)

#-----------------------------------------------------------------------------

stoch_size	= len(Y_all[0, 0])

for time_i in range(len(time)):
	#Reverse the time step (at the beginning the solution changes a lot for ordering, see below)
	#We need every time step to make sure no flip occurs in signs (but can be removed at the end)
	time_i	= len(time) - time_i - 1
	print(time_i)

	#First determine the highest variance for the last time frame
	#Note that you can reverse the basis by a factor -1, therefore we need to use the previous time step as well
	D, P	= np.linalg.eig(np.matmul(Y_all[time_i], Y_all[time_i].transpose()))

	#Sort on eigenvalue, this one is now fixed over time
	sorted_index    = np.argsort(D)[::-1]
	D		= D[sorted_index]
	P		= P[:, sorted_index]

	#Now check whether each column has still the same sign in the sorted covariance matrix
	#We use the maximum value of each column (because the sign can not abruptly change (this is gradual))
	max_index	= np.argmax(abs(P), axis = 0)

	if time_i == len(time) - 1:
		#First time, make previous array for sign and index
		sign_prev	= np.zeros(len(max_index))
		max_index_prev	= np.copy(max_index)
		for i in range(len(max_index)):
			#The maximum value in a column determines the dominant DO mode
			#So first column and index = 3, the current fourth DO mode -> first DO mode (sorted)
			#The largest value copies most of the pattern, but maximum value needs to be positive
			#Otherwise, the pattern is switched in sign
			if np.sign(P[max_index[i], i]) < 0:
				#Switch sign
				P[:, i] = -P[:, i]

			#Get the sign of the initial frame
			sign_prev[i]	= np.sign(P[max_index[i], i])

	#Now check previous sign using the previous locations, these values do not have to be the current maximum
	for i in range(len(max_index)):
		sign_current	= np.sign(P[max_index_prev[i], i])

		if sign_current != sign_prev[i]:
			#Sign was flipped, change sign of covariance array
			P[:, i] = -P[:, i]

		#Now overwrite the previous sign and index
		sign_prev[i]		= np.sign(P[max_index[i], i])
		max_index_prev[i]	= max_index[i]

	#Multiply for the correct basis
	V_sort	= np.matmul(V_all[time_i], P)
	Y_sort	= np.matmul(P.transpose(), Y_all[time_i])
	Y_var	= D / stoch_size

	#Save the sorted basis
	Y_all[time_i]	= Y_sort
	V_all[time_i]	= V_sort
	norm_V[time_i]	= Y_var

#-----------------------------------------------------------------------------

time_step = 50
time	  = time[::time_step]
x_all	  = x_all[::time_step]
V_all	  = V_all[::time_step]
Y_all	  = Y_all[::time_step]
norm_x	  = norm_x[::time_step]	
norm_V	  = norm_V[::time_step]

#-----------------------------------------------------------------------------

filename_out = directory+'QG_sorted_basis_nx_'+str(nx)+'_ny_'+str(ny)+'_dt_'+str(int(delta_t))+'_'+str(delta_t)[2:]+'_nV_'+str(nV)+'_Re_'+str(Re)+'_Stoch_amp_'+str("{:.0e}".format(stoch_amp))+'.nc'

QG.WriteOutput(filename_out, nV, time, x_all, V_all, Y_all, norm_x, norm_V)

