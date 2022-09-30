#Program determines the energy transfer rates
#See Sapsis and Dijkstra 2013, https://doi.org/10.1175/JPO-D-12-047.1

from pylab import *
import numpy as np
import netCDF4 as netcdf

#Custom made modules
from QG_model import *
from Norm_basis import *

#Directory for the output
directory = '../Data/QG/'

def ComputeUV(stream, nx):
	"""Determines the (staggered) horizontal velocities from the streamfunction"""
	dx	= 1.0 / nx
	dy	= 1.0 / nx

	#Empty arrays, create additional row for each component
	u	= np.zeros((nx+1, nx))
	v	= np.zeros((nx, nx+1))

	for x_i in range(nx - 1):
		#Now determine the velocities, u = -d_y psi and v = d_x psi	
		u[x_i+1] 	= -(stream[x_i+1] - stream[x_i]) / dy
		v[:, x_i+1] 	= (stream[:, x_i+1] - stream[:, x_i]) / dx

	#Extend u in y direction for a free slip boundary condition copy the current border -> u_y=0 at the boundary
	#For a no-slip boundary copy minus the current border -> average u = 0, at north and south boundarys we have slip
	u[0] = np.copy(u[1])
	u[-1]= np.copy(u[-2])
	
	#East and west boundaries are no slip
	v[:, 0]	= -v[:, 1]
	v[:, -1]= -v[:, -2]

	return u, v

def CentralGradients(u, v, nx):
	"""Determines the center gradients of the staggered horizontal velocities"""
	dx	= 1.0 / nx
	dy	= 1.0 / nx

	#Determine the central gradients 
	du_dx	= (u[1:-1, 1:] - u[1:-1, :-1]) / dx
	dv_dy	= (v[1:, 1:-1] - v[:-1, 1:-1]) / dy

	#Determine the gradient in the other directions
	tmp	= (u[1:] - u[:-1]) / dy
	tmp	= (tmp[1:] + tmp[:-1]) / 2.0
	du_dy	= (tmp[:, 1:] + tmp[:, :-1]) / 2.0

	tmp	= (v[:, 1:] - v[:, :-1]) / dx
	tmp	= (tmp[:, 1:] + tmp[:, :-1]) / 2.0
	dv_dx	= (tmp[1:] + tmp[:-1]) / 2.0

	return du_dx, du_dy, dv_dx, dv_dy

def AverageUV(u, v):
	"""Determines the velocities at the center of a grid cell"""
	u	= (u[1:-1, 1:] + u[1:-1, :-1]) / 2.0
	v	= (v[1:, 1:-1] + v[:-1, 1:-1]) / 2.0

	return u, v
#-----------------------------------------------------------------------------
#---------------------------MAIN SCRIPT STARTS HERE---------------------------
#-----------------------------------------------------------------------------	

nx          = 64 #Numbber of zonal dimensions
ny          = 64 #Number of meridional dimensions
nV          = 4  #Number of DO modes
Re	        = 40
delta_t	    = 0.0005
stoch_amp   = 1

#-----------------------------------------------------------------------------

#Initiate the QG model
QG = QG_model(nx, ny)

#Using the original or sorted DO basis doesn't matter
filename = directory+'QG_sorted_basis_nx_'+str(nx)+'_ny_'+str(ny)+'_dt_'+str(int(delta_t))+'_'+str(delta_t)[2:]+'_nV_'+str(nV)+'_Re_'+str(Re)+'_Stoch_amp_'+str(stoch_amp)+'_transition.nc'

time, x, V_all, Y_all, norm_x, norm_V = QG.ReadOutput(filename)

#-----------------------------------------------------------------------------

E_diss		= np.zeros((len(time), nV))
E_mean_DO	= np.zeros((len(time), nV))
E_modes_DO	= np.zeros((len(time), nV))

for time_i in range(len(time)):
	print(time_i)

	if time[time_i] < 0:
		#Spin-up, only 1,000 of realisations
		Y	= Y_all[time_i, :, :1000]	

	else:
		#Get the full coefficients after the spin-up
		Y	= Y_all[time_i]

	#Current time mean of Y and set to zero
	Y_exp	= np.mean(Y, axis = 1)
	
	#Set the mean of the coefficients to zero
	for nV_i in range(nV):
		Y[:, nV_i]	= Y[:, nV_i] - Y_exp
	
	#Conduct SVD on the Y coefficients
	U, S, Vh = np.linalg.svd(Y)

	#Determine the variance
	Var	= np.diag(S**2.0) / len(Y[0])

	#Now rescale V and Y
	V	= np.matmul(V_all[time_i], U)
	Y	= np.matmul(U.transpose(), Y)

	if time_i == 0:
		#Fill indices (0 = diagonal) and corresponding values
		index	= np.array([-1, 0, 1])
		values	= np.array([-1, 2, -1])
		D       = np.zeros((nx, nx))

		for i in range(nx):
			#Fill each entry
			index_fill		= np.where((index + i >= 0) & (index + i < 64))[0]
			values_fill		= values[index_fill]
			index_fill		= index[index_fill] + i
			D[i, index_fill]	= values_fill

		#Streamfunction should be zero around boundaries
		D[[0, -1], :]	= 0
		D[:, [0, -1]]	= 0
		D		= np.kron(np.eye(nx), D) + np.kron(D, np.eye(nx))
		D		= np.kron(D, np.array([[0, 0], [0, 1]]))

	#Orhogonalise the stochastic basis
	V, R	= qrM(V, D)

	#Now sort the basis and adapt the Y basis based on the change in V
	Var, P	= np.linalg.eig(np.matmul(np.matmul(R, Var), R.transpose()))

	#Sort on eigenvalue, this one is now fixed over time
	sorted_index    = np.argsort(Var)[::-1]
	Var		= Var[sorted_index]
	P		= P[:, sorted_index]

	#Now sort the stochastic basis and coefficients
	V		= np.matmul(V, P)
	Y		= np.matmul(P.transpose(), np.matmul(R, Y))

	for i in range(2):
		#Current time mean of Y and set to zero
		Y_exp	= np.mean(Y, axis = 1)

		for nV_i in range(nV):
			Y[:, nV_i]	= Y[:, nV_i] - Y_exp

		#Conduct twice SVD on the Y coefficients
		U, S, Vh = np.linalg.svd(Y)

		#Determine the variance
		Var	= np.diag(S**2.0) / len(Y[0])

		#Now rescale V and Y
		V	= np.matmul(V, U)
		Y	= np.matmul(U.transpose(), Y)

	#Determine YY (take the sum over the realisations to find E[YY])
	#YY is sorted as, Y11, Y12, Y13, Y14, Y21, Y22, etc.	
	YY	 = np.zeros((nV * nV, len(Y[0])))

	#Determine the covariance matrix
	for i in range(nV):
		for j in range(nV):
			YY[i * nV + j]	= Y[i] * Y[j]

	#Determine the expectance of YYY
	#First column, Y111, Y121, Y131, Y141, Y211, Y221 etc.
	#Second column: Y112, Y122, Y132, Y142
	YYY	    = np.matmul(YY, Y.transpose()) / len(Y[0])

	#Now only get the streamfunction part
	stream_mean 	= x[time_i, 1::2].reshape(nx, ny).transpose()
	stream_V	= np.zeros((nV, ny, nx))

	for V_i in range(nV):
		stream_V[V_i] = V[1::2, V_i].reshape(nx, ny).transpose()

	#Get the relevant horizontal velocity values for the DO mean
	u_mean, v_mean					= ComputeUV(stream_mean, nx)
	du_dx_mean, du_dy_mean, dv_dx_mean, dv_dy_mean 	= CentralGradients(u_mean, v_mean, nx)
	u_mean, v_mean					= AverageUV(u_mean, v_mean)

	#Empty arrays for the basis
	u_basis		= np.zeros((nV, nx - 1, nx - 1)) 
	v_basis		= np.zeros(np.shape(u_basis))
	du_dx_basis	= np.zeros(np.shape(u_basis))
	du_dy_basis	= np.zeros(np.shape(u_basis))
	dv_dx_basis	= np.zeros(np.shape(u_basis))
	dv_dy_basis	= np.zeros(np.shape(u_basis))

	for V_i in range(nV):
		#Now get the relevant horizontal velocity values for the DO basis
		u_tmp, v_tmp								= ComputeUV(stream_V[V_i], nx)
		du_dx_basis[V_i], du_dy_basis[V_i], dv_dx_basis[V_i], dv_dy_basis[V_i] 	= CentralGradients(u_tmp, v_tmp, nx)
		u_basis[V_i], v_basis[V_i]						= AverageUV(u_tmp, v_tmp)
	
	#Finally compute the energy transfer
	dx	= 1.0 / nx
	dy	= 1.0 / nx

	for V_i in range(nV):
		E_diss[time_i, V_i] = -(Var[V_i, V_i]/Re)*dx*dy* np.sum(du_dx_basis[V_i]**2.0+du_dy_basis[V_i]**2.0+dv_dx_basis[V_i]**2.0+dv_dy_basis[V_i]**2.0)

		E_mean_DO[time_i, V_i] = -Var[V_i, V_i]*dx*dy*np.sum(du_dx_mean * u_basis[V_i]**2.0+dv_dy_mean * v_basis[V_i]**2.0+(du_dy_mean+dv_dx_mean)*u_basis[V_i]*v_basis[V_i])

		#Start with empty array for Einsten notation
		E_modes	= 0.0

		for V_p in range(nV):
			for V_q in range(nV):
				#Loop over each component (Einstein notation)
				E = np.sum(du_dx_basis[V_i]*u_basis[V_p]*u_basis[V_q] +dv_dy_basis[V_i]*v_basis[V_p]*v_basis[V_q]+du_dy_basis[V_i]*u_basis[V_p]*v_basis[V_q] +dv_dx_basis[V_i]*v_basis[V_p]*u_basis[V_q]) * dx*dy

				#E[Y_i*Y_p*Y_q] = YY[V_i * nV + V_p]Y[V_q]
				E_modes += YYY[V_i * nV + V_p, V_q] * E

		#Save the energy transfer from the DO modes to 1 DO mode
		E_modes_DO[time_i, V_i] = -E_modes


#-----------------------------------------------------------------------------

filename = directory+'QG_energy_nx_'+str(nx)+'_ny_'+str(ny)+'_dt_'+str(int(delta_t))+'_'+str(delta_t)[2:]+'_nV_'+str(nV)+'_Re_'+str(Re)+'_Stoch_amp_'+str(stoch_amp)+'_transition.nc'

fh = netcdf.Dataset(filename, 'w')

fh.createDimension('time', len(time))
fh.createDimension('nV', nV)

fh.createVariable('time', float, ('time'), zlib=True)
fh.createVariable('nV', float, ('nV'), zlib=True)
fh.createVariable('E_diss', float, ('time', 'nV'), zlib=True)
fh.createVariable('E_mean_DO', float, ('time', 'nV'), zlib=True)
fh.createVariable('E_modes_DO', float, ('time', 'nV'), zlib=True)

fh.variables['time'].longname		= 'Array of dimensionless time'
fh.variables['nV'].longname		    = 'Number of DO components'
fh.variables['E_diss'].longname		= 'Energy dissipation per mode'
fh.variables['E_mean_DO'].longname	= 'Energy transfer from DO mean to a mode'
fh.variables['E_modes_DO'].longname	= 'Energy transfer from DO modes to a mode'

#Writing data to correct variable	
fh.variables['time'][:]     	= time
fh.variables['nV'][:] 		    = np.arange(nV)+1
fh.variables['E_diss'][:] 	    = E_diss
fh.variables['E_mean_DO'][:] 	= E_mean_DO
fh.variables['E_modes_DO'][:] 	= E_modes_DO

fh.close()