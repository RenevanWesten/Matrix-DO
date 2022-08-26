#Class for generating QG model 

import numpy as np
from scipy.sparse import spdiags
import netCDF4 as netcdf
import os

#Custom made modules
from QG_functions import *

class QG_model:
    
    def __init__(self, nx, ny):
        """Generates the set-up for the QG model"""
        self.nx     = nx
        self.ny     = ny
        self.ndim   = 2 * nx * ny
        self.adim   = 25 * self.ndim
        self.hdim   = 600.              #Depth first layer (m)
        self.f0     = 1.0 * 10**(-4.0)  #Coriolis acceleration (1/s)
        self.beta0  = 1.6 * 10**(-11.0) #Gradient in planitary vorticity (1/ms)
        self.Lx     = 1.0 * 10**(6.0)   #Zonal length basin (m)
        self.Ly     = 1.0 * 10**(6.0)   #Meridional length basin (m)
        self.U      = 1.6 * 10**(-2.0)  #Sverdrup velocity (m/s)
        self.g      = 2.0 * 10**(-2.0)  #Density first layer (kg/m^3)
        self.rho    = 1.0 * 10**(3.0)   #Density (kg/m^3)
        self.tau    = 1.5 * 10**(-1.0)  #Wind stress (N/m^2)
        self.Ah     = 1.0 * 10**(3.0)   #Lateral friction (m^2/s)
        self.bf     = 0.0               #Bottom friction (1/s)
        self.xmin   = 0.0               #Scaled zonal minimum
        self.xmax   = 1.0               #Scaled zonal maximum
        self.ymin   = 0.0               #Scaled meridional minimum
        self.ymax   = self.Ly / self.Lx #Scaled meridional maximum
        
        #Parameter list
        parameters      = np.zeros(21)
        parameters[0]   = 1.0 * 10**(3.0)                           #Alpha tau
        parameters[1]   = self.beta0 * self.Lx * self.Ly / self.U   #Beta parameter
        parameters[2]   = 0.0                                       #No bottom friction
        parameters[3]   = self.Ly / self.Lx                         #Aspect ratio (W/L)
        parameters[4]   = self.U * self.Lx / self.Ah                #Reynolds number
        parameters[5]   = 1.0                                       #No (free) slip EW boundary 1.0 (0.0)
        parameters[6]   = 0.0                                       #No (free) slip NS boundary 1.0 (0.0)
        self.parameters = parameters
        
        #Now determine the scaled grid spacing and coordinates
        self.dx     = (self.xmax - self.xmin) / (nx - 1)
        self.dy     = (self.ymax - self.ymin) / (ny - 1)
        self.x      = np.linspace(self.xmin, self.xmax, nx)
        self.y      = np.linspace(self.ymin, self.ymax, ny)
        
        #Get the wind forcing
        self.taux, self.tauy  = WindForcing(self.x, self.y)
        
        #Determine the linear operators which are independent of the state
        self.z, self.dxx, self.dyy, self.cor, self.Llzz, self.Llzp, self.Llpz, self.Llpp = Linear(self.parameters, self.nx, self.ny, self.dx, self.dy)
    
    def Set_parameters(self, index, value):
        """Set/change parameter value"""
        self.parameters[index] = value
        
        #Update operators
        self.taux, self.tauy  = WindForcing(self.x, self.y)
        self.z, self.dxx, self.dyy, self.cor, self.Llzz, self.Llzp, self.Llpz, self.Llpp = Linear(self.parameters, self.nx, self.ny, self.dx, self.dy)

    def Mass(self):
        """Generate mass matrix for QG model"""
        Mass    = np.zeros(self.ndim)
        
        self.Tlzz, self.Tlzp = Time_dependance(self.parameters, self.nx, self.ny)
        B = AssembleB(self.nx, self.ny, self.ndim, self.adim, self.Tlzz, self.Tlzp)

        for i in range(self.ndim):
            Mass[i] = 0
            for j in range(int(B['beg'][i]), int(B['beg'][i+1])):
                if B['jco'][j] != i:
                    if B['co'][j] != 0.0:
                        raise ValueError('Mass matrix has wrong format')
                        
                else:
                    Mass[i] = B['co'][j]
        
        #Diagonalise mass matrix
        Mass        = -spdiags(Mass, 0, self.ndim, self.ndim)
        self.Mass   = np.array(Mass.todense())

        return self.Mass
    
    def Stochastic_forcing(self, stoch_amp):
        """Generate the stochastic forcing for the QG model"""
        
        #Forcing array
        forcing = np.zeros((self.ndim, 2))
        
        #Local variables
        l  	= 0.125
        C  	= 1.0
        nx_for  = np.sqrt(self.ndim / 2.0)

        for col_i in range(2):
            eta_x = C * col_i
            eta_y = C * (1 - col_i)
            
            for i in range(1, self.ndim, 2):
                
                j = i / 2.0
                x = (j % nx_for) / (nx_for - 1)
                y = np.floor(j / nx_for) / (nx_for - 1)
     
                forcing[i-1, col_i]   = np.exp(-2 * ((x-0.5)**2.0 + (y-0.5)**2.0) / (4.0 * l**2.0)) * (eta_y - 2 * eta_y * x + 2 * eta_x * y - eta_x) / (2 * l**2.0)
                forcing[i, col_i] = 0.0
                
                if self.Mass[i-1, i-1] == 0.0:
                    forcing[i-1, col_i] = 0.0

        return forcing * stoch_amp
                
            
    def RHS(self, X):
        """Generates the RHS for the DO method"""
        Frc = np.zeros(self.ndim)
        
        #Wind stress magnitude
        alpha_tau   = self.parameters[0] * self.parameters[10]  
        
        #Convert X array to streamfunction (psi)
        om, ps      = U_to_Psi(self.nx, self.ny, X)
        
        #Determine the local matrices for nonlinear operators
        self.Nlzz, self.Nlzp = Nlin_RHS(self.dx, self.dy, self.nx, self.ny, self.parameters[9], om, ps)
       
        for x_i in range(1, self.nx-1):
            for y_i in range(1, self.ny-1):
                row          = 2 * (self.ny * x_i + y_i)
                Frc[row]     = alpha_tau * self.taux[y_i, x_i]
                Frc[row+1]   = 0.0
                                
        self.Alzz   = self.Llzz + self.Nlzz
        self.Alzp   = self.Llzp + self.Nlzp
        self.Alpz   = np.copy(self.Llpz)
        self.Alpp   = np.copy(self.Llpp)
        
        #Impose the boundaries conditions (slip or no slip)
        self.Alzz, self.Alzp, self.Alpz, self.Alpp = Boundaries(self.dx, self.dy, self.nx, self.ny, self.parameters, self.Alzz, self.Alzp, self.Alpz, self.Alpp)

        #Determine the A matrix
        A = AssembleA(self.nx, self.ny, self.ndim, self.adim, self.Alzz, self.Alzp, self.Alpz, self.Alpp)

        #B = Au - Frc
        b   = np.zeros(self.ndim)
        for n_i in range(self.ndim):
            b[n_i] = -Frc[n_i]
            for v in range(int(A['beg'][n_i]), int(A['beg'][n_i+1])):
                b[n_i] = b[n_i] + (A['co'][v] * X[int(A['jco'][v])])
                
        return b

    def Jacobian(self, X, sig):
        """Generates the Jacobian for the DO method"""
        
        #Convert X array to streamfunction (psi)
        om, ps      = U_to_Psi(self.nx, self.ny, X)  
            
        #Determine the local matrices for nonlinear operators
        self.Nlzz, self.Nlzp = Nlin_Jac(self.dx, self.dy, self.nx, self.ny, self.parameters[9], om, ps)     
        self.Tlzz, self.Tlzp = Time_dependance(self.parameters, self.nx, self.ny)

        self.Alzz   = self.Llzz + self.Nlzz + sig * self.Tlzz
        self.Alzp   = self.Llzp + self.Nlzp + sig * self.Tlzp
        self.Alpz   = np.copy(self.Llpz)
        self.Alpp   = np.copy(self.Llpp)

        #Impose the boundaries conditions (slip or no slip)
        self.Alzz, self.Alzp, self.Alpz, self.Alpp = Boundaries(self.dx, self.dy, self.nx, self.ny, self.parameters, self.Alzz, self.Alzp, self.Alpz, self.Alpp)

        #Determine the A matrix
        A = AssembleA(self.nx, self.ny, self.ndim, self.adim, self.Alzz, self.Alzp, self.Alpz, self.Alpp)

        #
        beg = np.copy(A['beg'])
        co  = np.zeros(self.ndim * 4 * 9) 
        jco = np.zeros(self.ndim * 4 * 9) 
        
        for i in range(int(beg[self.ndim])):
            co[i]  = A['co'][i]
            jco[i] = A['jco'][i]
           
        row = 0
        idx = 0
        values  = np.zeros(int(beg[-1]))
            
        while row < self.ndim:
            for k in range(int(beg[row]), int(beg[row+1])):
                values[idx] = row
                idx         += 1
            row += 1
           
        Jac = np.zeros((self.ndim, self.ndim))
        
        for i in range(len(values)):
            Jac[int(values[i]), int(jco[i])] = -co[i]
        
        return Jac  
    
    def Bilin(self, V, W):
        """Generates the Bilinear form for the DO method"""
        #Get the dimensions for V and W
        if len(np.shape(V)) == 1: 
            nV	= 1
            V   = np.reshape(V, (len(V), 1))
        else: 
            nV	= len(V[0])
          
        if len(np.shape(W)) == 1: 
            nW	= 1
            W   = np.reshape(W, (len(W), 1))
        else: 
            nW	= len(W[0])
        
        Z   = np.zeros((len(V), nV * nW))

        for V_i in range(nV):

            #Get the individual DO components for V
            v_comp = V[:, V_i]

            #Convert X array to streamfunction (psi)
            om, ps = U_to_Psi(self.nx, self.ny, v_comp) 

            #Determine the local matrices for nonlinear operators
            self.Nlzz, self.Nlzp = Nlin_RHS(self.dx, self.dy, self.nx, self.ny, self.parameters[9], om, ps)

            self.Alzz   = np.copy(self.Nlzz)
            self.Alzp   = np.copy(self.Nlzp)
            self.Alpz   = np.zeros((self.ny, self.nx, 10))
            self.Alpp   = np.zeros((self.ny, self.nx, 10))

            #Determine the A matrix
            A = AssembleA(self.nx, self.ny, self.ndim, self.adim, self.Alzz, self.Alzp, self.Alpz, self.Alpp)

            for W_i in range(nW):
                
                #Get the individual DO components for W
                w_comp = W[:, W_i]

                #B = Au
                b   = np.zeros(self.ndim)
                for n_i in range(self.ndim):
                    for v in range(int(A['beg'][n_i]), int(A['beg'][n_i+1])):
                        b[n_i] = b[n_i] - (A['co'][v] * w_comp[int(A['jco'][v])])
        
                Z[:, V_i * nW + W_i] = b
                
        return Z
              
    def WriteOutput(self, filename, nV, time_all, x_all, V_all, Y_all, norm_x, norm_V):
        """Write out the output as NETCDF file"""
        
        stream_mean = np.zeros((len(time_all), len(self.y), len(self.x)))
        stream_all 	= np.zeros((len(time_all), nV, len(self.y), len(self.x)))
        vor_mean	= np.zeros((len(time_all), len(self.y), len(self.x)))
        vor_all		= np.zeros((len(time_all), nV, len(self.y), len(self.x)))
        stoch_iter  = len(Y_all[0,0])

        for time_i in range(len(time_all)):
            #Map to correct grid (2-dof, 0 = vor, 1 = stream)
            vor_mean[time_i] 	= x_all[time_i, 0::2].reshape(self.nx, self.ny)
            stream_mean[time_i] = x_all[time_i, 1::2].reshape(self.nx, self.ny)

            for V_i in range(nV):
                vor_all[time_i, V_i] 	= V_all[time_i, 0::2, V_i].reshape(self.nx, self.ny)
                stream_all[time_i, V_i] = V_all[time_i, 1::2, V_i].reshape(self.nx, self.ny)

        #First remove file and generate new one (otherwise error?)
        if os.path.exists(filename):
            os.remove(filename)

        fh = netcdf.Dataset(filename, 'w')

        fh.createDimension('time', len(time_all))
        fh.createDimension('x', len(self.x))
        fh.createDimension('y', len(self.y))
        fh.createDimension('nV', nV)
        fh.createDimension('nY', stoch_iter)
        
        fh.createVariable('time', float, ('time'), zlib=True)
        fh.createVariable('x', float, ('x'), zlib=True)
        fh.createVariable('y', float, ('y'), zlib=True)
        fh.createVariable('nV', float, ('nV'), zlib=True)
        fh.createVariable('nY', float, ('nY'), zlib=True)
        fh.createVariable('stream_mean', float, ('time', 'y', 'x'), zlib=True)
        fh.createVariable('vor_mean', float, ('time', 'y', 'x'), zlib=True)
        fh.createVariable('stream_V', float, ('time', 'nV', 'y', 'x'), zlib=True)
        fh.createVariable('vor_V', float, ('time', 'nV', 'y', 'x'), zlib=True)
        fh.createVariable('Y', float, ('time', 'nV', 'nY'), zlib=True)
        fh.createVariable('norm_mean', float, ('time'), zlib=True)
        fh.createVariable('norm_V', float, ('time', 'nV'), zlib=True)
        
        fh.variables['time'].longname		= 'Array of dimensionless time'
        fh.variables['x'].longname		    = 'Array of dimensionless longitudes'
        fh.variables['y'].longname		    = 'Array of dimensionless latitudes'
        fh.variables['nV'].longname		    = 'Number of DO components'
        fh.variables['nY'].longname		    = 'Stochastic realisations'
        fh.variables['stream_mean'].longname= 'Streamfunction (DO mean)'
        fh.variables['stream_V'].longname	= 'Streamfunction (DO components)'
        fh.variables['vor_mean'].longname	= 'Vorticity (DO mean)'
        fh.variables['vor_V'].longname		= 'Vorticity (DO components)'
        fh.variables['norm_mean'].longname	= 'Variance of mean component'
        fh.variables['norm_V'].longname		= 'Variance of DO components'
        
        #Writing data to correct variable	
        fh.variables['time'][:]     	= time_all
        fh.variables['x'][:] 			= self.x
        fh.variables['y'][:] 			= self.y
        fh.variables['nV'][:] 			= np.arange(nV)+1
        fh.variables['nY'][:] 			= np.arange(stoch_iter)+1
        fh.variables['stream_mean'][:] 	= stream_mean
        fh.variables['stream_V'][:] 	= stream_all
        fh.variables['vor_mean'][:] 	= vor_mean
        fh.variables['vor_V'][:] 		= vor_all
        fh.variables['Y'][:] 			= Y_all
        fh.variables['norm_mean'][:] 	= norm_x
        fh.variables['norm_V'][:] 		= norm_V
        
        fh.close()

    def ReadOutput(self, filename):
        """Read the output of an already existing NETCDF file"""

        fh = netcdf.Dataset(filename, 'r')
        
        time_all    = fh.variables['time'][:]     		   		   		   		
        stream_mean = fh.variables['stream_mean'][:] 
        stream_all  = fh.variables['stream_V'][:]
        vor_mean    = fh.variables['vor_mean'][:] 
        vor_all     = fh.variables['vor_V'][:]
        Y_all       = fh.variables['Y'][:]
        norm_x      = fh.variables['norm_mean'][:]
        norm_V      = fh.variables['norm_V'][:]

        fh.close
        
        x_all   = np.zeros((len(time_all), self.ndim))
        V_all   = np.zeros((len(time_all), self.ndim, len(vor_all[0])))

        counter = 0
        for y_i in range(self.ny):
            for x_i in range(self.nx):
                x_all[:, counter]   = vor_mean[:, y_i, x_i]
                x_all[:, counter+1] = stream_mean[:, y_i, x_i]
                V_all[:, counter]   = vor_all[:, :, y_i, x_i]
                V_all[:, counter+1] = stream_all[:, :, y_i, x_i]
                counter += 2

        return time_all, x_all, V_all, np.array(Y_all), np.array(norm_x), np.array(norm_V)
