#Program plots the energy transfer rates

from pylab import *
import numpy as np
import netCDF4 as netcdf

#Directory for the output
directory = '../Data/QG/'

#-----------------------------------------------------------------------------
#---------------------------MAIN SCRIPT STARTS HERE---------------------------
#-----------------------------------------------------------------------------	

nx          = 64 #Numbber of zonal dimensions
ny          = 64 #Number of meridional dimensions
nV          = 4 #Number of DO modes
Re	        = 40
delta_t	    = 0.0005
stoch_amp   = 1


#-----------------------------------------------------------------------------

filename = directory+'QG_energy_nx_'+str(nx)+'_ny_'+str(ny)+'_dt_'+str(int(delta_t))+'_'+str(delta_t)[2:]+'_nV_'+str(nV)+'_Re_'+str(Re)+'_Stoch_amp_'+str(stoch_amp)+'_transition.nc'

fh = netcdf.Dataset(filename, 'r')

time		= fh.variables['time'][:]     
E_diss		= fh.variables['E_diss'][:] 	
E_mean_DO	= fh.variables['E_mean_DO'][:] 
E_modes_DO	= fh.variables['E_modes_DO'][:]

fh.close()

filename = directory+'QG_sorted_basis_nx_'+str(nx)+'_ny_'+str(ny)+'_dt_'+str(int(delta_t))+'_'+str(delta_t)[2:]+'_nV_'+str(nV)+'_Re_'+str(Re)+'_Stoch_amp_'+str(stoch_amp)+'_transition.nc'

fh = netcdf.Dataset(filename, 'r')

time		= fh.variables['time'][:]     
norm_x		= fh.variables['norm_mean'][:] 
norm_V		= fh.variables['norm_V'][:]

fh.close()

#-----------------------------------------------------------------------------

fig, ax     = subplots()

graph_1	= ax.plot(time, E_mean_DO[:,0], linestyle = '-', color = 'red', linewidth = 1.5, label = '$v_1$')
graph_2	= ax.plot(time, E_mean_DO[:,1], linestyle = '-', color = 'blue', linewidth = 1.5, label = '$v_2$')
graph_3	= ax.plot(time, E_mean_DO[:,2], linestyle = '-', color = 'firebrick',linewidth = 1.5, label = '$v_3$')
graph_4	= ax.plot(time, E_mean_DO[:,3], linestyle = '-', color = 'cyan', linewidth = 1.5, label = '$v_4$')

ax.set_ylim(-10, 90)
ax.set_xlim(-0.5, 5)
ax.set_xlabel('Time')
ax.set_ylabel('Energy transfer rate')
ax.grid()

ax.axvline(x = 0, linewidth = 2.0, linestyle = '--', color = 'k')
ax.text(-0.2, 40, 'Spin-up', verticalalignment='center', horizontalalignment='center', color = 'k', fontsize=14, rotation=90)

graphs		= graph_1 + graph_2 + graph_3 + graph_4
legend_labels 	= [l.get_label() for l in graphs]
legend      	= ax.legend(graphs, legend_labels, loc = 'center right', ncol=1, numpoints = 1, framealpha=1)

ax2 	= fig.add_axes([0.27, 0.30, 0.50, 0.40])

#graph_1	= ax2.plot(time, norm_x, '-k', linewidth = 2.0, label = r'$\overline{v}$')
graph_2	= ax2.plot(time, norm_V[:,0], linestyle = '-', color = 'red', linewidth = 1.5, label = '$v_1$')
graph_3	= ax2.plot(time, norm_V[:,1], linestyle = '-', color = 'blue', linewidth = 1.5, label = '$v_2$')
graph_4	= ax2.plot(time, norm_V[:,2], linestyle = '-', color = 'firebrick',linewidth = 1.5, label = '$v_3$')
graph_5	= ax2.plot(time, norm_V[:,3], linestyle = '-', color = 'cyan', linewidth = 1.5, label = '$v_4$')

ax2.set_yscale('log')
ax2.set_ylim(10**(0), 10**8.0)
ax2.set_xlim(-0.5, 5)
ax2.grid()

ax2.axvline(x = 0, linewidth = 2.0, linestyle = '--', color = 'k')
ax2.text(-0.2, 1.5*10**(1), 'Spin-up', verticalalignment='center', horizontalalignment='center', color = 'k', fontsize=9, rotation=90)
ax2.set_title('Variance')

if Re == 40 and nV == 4 and stoch_amp == 1:
	ax.set_title(r'a) $\overline{v} \rightarrow v_i$, $\sigma = 1$')

if Re == 40 and nV == 4 and stoch_amp == 5:
	ax.set_title(r'd) $\overline{v} \rightarrow v_i$, $\sigma = 5$')

if Re == 40 and nV == 4 and stoch_amp == 10:
	ax.set_title(r'g) $\overline{v} \rightarrow v_i$, $\sigma = 10$')

if Re == 40 and nV == 4 and stoch_amp == 15:
	ax.set_title(r'j) $\overline{v} \rightarrow v_i$, $\sigma = 15$')

fig, ax     = subplots()

graph_1	= ax.plot(time, E_modes_DO[:,0], linestyle = '-', color = 'red', linewidth = 1.5, label = '$v_1$')
graph_2	= ax.plot(time, E_modes_DO[:,1], linestyle = '-', color = 'blue', linewidth = 1.5, label = '$v_2$')
graph_3	= ax.plot(time, E_modes_DO[:,2], linestyle = '-', color = 'firebrick',linewidth = 1.5, label = '$v_3$')
graph_4	= ax.plot(time, E_modes_DO[:,3], linestyle = '-', color = 'cyan', linewidth = 1.5, label = '$v_4$')

if nV == 4:
	ax.set_ylim(-3, 3)
	ax.axvline(x = 0, linewidth = 2.0, linestyle = '--', color = 'k')
	ax.text(-0.2, -1, 'Spin-up', verticalalignment='center', horizontalalignment='center', color = 'k', fontsize=14, rotation=90)
else:
	ax.set_ylim(-6, 6)
	ax.axvline(x = 0, linewidth = 2.0, linestyle = '--', color = 'k')
	ax.text(-0.2, -2, 'Spin-up', verticalalignment='center', horizontalalignment='center', color = 'k', fontsize=14, rotation=90)

ax.set_xlim(-0.5, 5)
ax.set_xlabel('Time')
ax.set_ylabel('Energy transfer rate')
ax.grid()


graphs		= graph_1 + graph_2 + graph_3 + graph_4
legend_labels 	= [l.get_label() for l in graphs]
legend      	= ax.legend(graphs, legend_labels, loc = 'lower right', ncol=1, numpoints = 1, framealpha=1)

if Re == 40 and nV == 4 and stoch_amp == 1:
	ax.set_title(r'b) DO $\rightarrow v_i$, $\sigma = 1$')

if Re == 40 and nV == 4 and stoch_amp == 5:
	ax.set_title(r'e) DO $\rightarrow v_i$, $\sigma = 5$')

if Re == 40 and nV == 4 and stoch_amp == 10:
	ax.set_title(r'h) DO $\rightarrow v_i$, $\sigma = 10$')

if Re == 40 and nV == 4 and stoch_amp == 15:
	ax.set_title(r'k) DO $\rightarrow v_i$, $\sigma = 15$')

fig, ax     = subplots()

graph_1	= ax.plot(time, E_diss[:,0], linestyle = '-', color = 'red', linewidth = 1.5, label = '$v_1$')
graph_2	= ax.plot(time, E_diss[:,1], linestyle = '-', color = 'blue', linewidth = 1.5, label = '$v_2$')
graph_3	= ax.plot(time, E_diss[:,2], linestyle = '-', color = 'firebrick',linewidth = 1.5, label = '$v_3$')
graph_4	= ax.plot(time, E_diss[:,3], linestyle = '-', color = 'cyan', linewidth = 1.5, label = '$v_4$')

ax.set_ylim(-90, 10)
ax.set_xlim(-0.5, 5)
ax.set_xlabel('Time')
ax.set_ylabel('Energy dissipation rate')
ax.grid()

ax.axvline(x = 0, linewidth = 2.0, linestyle = '--', color = 'k')
ax.text(-0.2, -40, 'Spin-up', verticalalignment='center', horizontalalignment='center', color = 'k', fontsize=14, rotation=90)

graphs		= graph_1 + graph_2 + graph_3 + graph_4
legend_labels 	= [l.get_label() for l in graphs]
legend      	= ax.legend(graphs, legend_labels, loc = 'center right', ncol=1, numpoints = 1, framealpha=1)

if Re == 40 and nV == 4 and stoch_amp == 1:
	ax.set_title(r'c) $v_i \rightarrow$ Dissipation, $\sigma = 1$')

if Re == 40 and nV == 4 and stoch_amp == 5:
	ax.set_title(r'f) $v_i \rightarrow$ Dissipation, $\sigma = 5$')

if Re == 40 and nV == 4 and stoch_amp == 10:
	ax.set_title(r'i) $v_i \rightarrow$ Dissipation, $\sigma = 10$')

if Re == 40 and nV == 4 and stoch_amp == 15:
	ax.set_title(r'l) $v_i \rightarrow$ Dissipation, $\sigma = 15$')

show()

