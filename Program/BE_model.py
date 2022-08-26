#Class for generating Burgers Equation

import numpy as np

class BE_model:
    
    def __init__(self, nx):
        """Generates the set-up for the Burgers Equation"""
        self.nx     = nx
        self.x_grid = np.linspace(0, 1.0, nx+1)[:-1]
        self.dx     = self.x_grid[1] - self.x_grid[0]
        
        #Set up the initial condition
        self.x0     = 0.5 * (np.exp(np.cos(2 * np.pi * self.x_grid)) - 1.5) * np.sin(2 * np.pi * (self.x_grid + 0.37))
                
    def Operators(self, mu):
        """Generates the operators depending on mu"""

        self.mu = mu
        Lap_op  = np.zeros((self.nx, self.nx))
        L1      = np.zeros((self.nx, self.nx))
        L2      = np.zeros((self.nx, self.nx))

        #Fill indices (0 = diagonal) and corresponding values
        index	= np.asarray([-self.nx + 1, -1, 0, 1, self.nx - 1])
        values	= np.asarray([1, 1, -2, 1, 1])
        
        for i in range(self.nx):
            #Fill each entry
            index_fill              = np.where((index + i >= 0) & (index + i < self.nx))[0]
            values_fill             = values[index_fill]
            index_fill              = index[index_fill] + i
            Lap_op[i, index_fill]   = values_fill
            
        #Get correct Laplacian operating space
        Lap_op	= Lap_op * self.mu / (self.dx**2.0)

        for i in range(self.nx):
            #Fill the diagonals with ones
            L1[i, i] = 1.0
            
        #Fill indices (0 = diagonal) and corresponding values
        index	= np.asarray([-self.nx + 1, -1, 1, self.nx - 1])
        values	= np.asarray([1, -1, 1, -1])

        for i in range(self.nx):
            #Fill each entry
            index_fill          = np.where((index + i >= 0) & (index + i < self.nx))[0]
            values_fill         = values[index_fill]
            index_fill          = index[index_fill] + i
            L2[i, index_fill]   = values_fill

        #Get correct Laplacian operating space
        L2              = L2 / (2.0 * self.dx)

        self.Lap_op     = Lap_op
        self.L1         = L1
        self.L2         = L2
    
    def Mass(self):
        """Generate mass matrix for Burgers equation"""
        self.Mass    = np.identity(self.nx)

        return self.Mass
    
    def Stochastic_forcing(self):
        """Generate the stochastic forcing for the Burgers equation"""
        return 0.5 * np.cos(4.0 * np.pi * self.x_grid) 
    
    def RHS(self, x, b):
        """Generates the RHS for the Burgers equation"""
        return np.matmul(self.Lap_op, x) - 0.5 * np.matmul(self.L2, x*x) + b

    def Jacobian(self, x):
        """Generates the Jacobian for the Burgers equation"""
        return self.Lap_op - np.matmul(self.L2, np.diag(x))
        
    def Bilin(self, V, W):
        """Generates the Bilinear form for the Burgers equation"""
        #Get the dimensions for V and W
        try: nV, mV 	= np.shape(V)[0], np.shape(V)[1]
        except: nV, mV	= len(V), 1
        try: nW, mW 	= np.shape(W)[0], np.shape(W)[1]
        except: nW, mW	= len(W), 1

        if mV * mW == 1:
            bilin	= V * W

        else:
            bilin	= np.zeros((nV, mV * mW))
            for i in range(mV):
                for j in range(mW):
                    #Determine the bilinair form
                    bilin[:, i * mV + j]	= V[:, i] * W[:, j]
            
        bilin = 0.5 * np.matmul(self.L2, bilin)
        return -bilin
