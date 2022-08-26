#Functions for the QG model 

import numpy as np

def WindForcing(x_grid, y_grid):
    """Determines the wind-stress curl for the field"""
    #Get the wind stress 2D fields
    taux   = np.zeros((len(y_grid), len(x_grid)))
    tauy   = np.zeros((len(y_grid), len(x_grid)))
    
    for y_i in range(1, len(y_grid) - 1):
        for x_i in range(1, len(x_grid) - 1):
            #Get the dimensionless wind stress curl (not on boundaries)
            #tau_x = -1/(2 pi) cos (2 pi y)
            #tau_y = 0
            taux[y_i, x_i] = - np.sin(2 * np.pi * y_grid[y_i] / np.max(y_grid))
            tauy[y_i, x_i] = 0
 
    #return taux, tauy
    return taux.transpose(), tauy.transpose()

def Linear(parameters, nx, ny, dx, dy):
    """Quasi-geostrophic barotropic equations,
    produce local element matrices for linear operators"""
    
    #Cell structure and index
    #2 5 
    #1 4 7
    #  3 6
    
    #Get the standard parameters (Reynolds, Beta, bottom friction)
    Re, Beta, rbf  = parameters[4], parameters[1], parameters[2]

    #Empty arrays for determining the linear operators
    z   = np.zeros((ny, nx, 10))   
    dxx = np.zeros((ny, nx, 10))   
    dyy = np.zeros((ny, nx, 10))                       
    cor = np.zeros((ny, nx, 10)) 
    Llpz= np.zeros((ny, nx, 10))

    #Bottom friction only applies at the center of grid cell
    #And determine for operator for Llpz
    for y_i in range(1, ny - 1):
        for x_i in range(1, nx - 1):  
            z[y_i, x_i, 4]    = 1.0
            Llpz[y_i, x_i, 4] = 1.0
 
    #Determine the second order finite difference in x direction
    #f(x = west) - 2 f(x = center) + f(x = east) / dx^2
    dx_dx = 1.0 / (dx**2.0)
    for y_i in range(1, ny - 1):
        for x_i in range(1, nx - 1):  
            dxx[y_i, x_i, 1] = dx_dx         #West
            dxx[y_i, x_i, 4] = -2.0 * dx_dx  #Center
            dxx[y_i, x_i, 7] = dx_dx         #East
            
    #Determine the second order finite difference in y direction
    #f(y = south) - 2 f(y = center) + f(y = north) / dy^2
    dy_dy = 1.0 / (dy**2.0)
    for y_i in range(1, ny - 1):
        for x_i in range(1, nx - 1):  
            dyy[y_i, x_i, 3] = dy_dy         #South
            dyy[y_i, x_i, 4] = -2.0 * dy_dy  #Center
            dyy[y_i, x_i, 5] = dy_dy         #North  
  
    #Determine the coriolis operator, only in west-east direction
    dx_2 = 1.0 / (2 * dx)
    for y_i in range(1, ny - 1):
        for x_i in range(1, nx - 1):  
            cor[y_i, x_i, 1] = - dx_2       #West
            cor[y_i, x_i, 7] = dx_2         #East
         
    #Operators for Llzz, Llzp and Llpp
    Llzz = -(dxx + dyy) / Re + rbf * z
    Llzp = Beta * cor
    Llpp = -(dxx + dyy) 
        
    return z, dxx, dyy, cor, Llzz, Llzp, Llpz, Llpp

def Time_dependance(parameters, nx, ny):
    """Generate time dependance operators"""
   
    #Cell structure and index
    #2 5 
    #1 4 7
    #  3 6
    
    #Get the standard parameters
    F = parameters[9]

    #Empty arrays for determining the linear operators (time dependant)
    Tlzz = np.zeros((ny, nx, 10))
    Tlzp = np.zeros((ny, nx, 10))
    
    #Determine operator at center of grid cell
    for y_i in range(1, ny - 1):
        for x_i in range(1, nx - 1):  
            Tlzz[y_i, x_i, 4] = 1.0
            Tlzp[y_i, x_i, 4] = -F
            
    return Tlzz, Tlzp

def AssembleA(nx, ny, ndim, adim, Alzz, Alzp, Alpz, Alpp):
    """Generates the A matrix"""
    A           = {}
    A['beg']    = np.zeros(ndim+1)
    A['jco']    = np.zeros(ndim*800)
    A['co']     = np.zeros(ndim*800)
    A['di']     = np.zeros(ndim)
    
    v   = 0
    i_s = 2
    j_s = 2 * ny
    
    for y_i in range(ny):
        #Southern grid cell
        row             = 2 * y_i
        A['beg'][row]   = v
        A['beg'][row+1] = v + 3
        A['jco'][v]     = row
        A['jco'][v+1]   = row + j_s             
        A['jco'][v+2]   = row + 1 + j_s
        A['jco'][v+3]   = row + 1    
        v += 4

    for x_i in range(1, nx - 1):
        #Western grid cell
        row             = 2 * ny * x_i
        A['beg'][row]   = v
        A['beg'][row+1] = v + 3 
        A['jco'][v]     = row
        A['jco'][v+1]   = row + i_s             
        A['jco'][v+2]   = row + 1 + i_s
        A['jco'][v+3]   = row + 1 
        v += 4
        
        for y_i in range(1, ny - 1):
            #Central grid cell
            row             = 2 * (ny * x_i + y_i)
            A['beg'][row]   = v
            A['beg'][row+1] = v + 10

            A['jco'][v]     = row - i_s
            A['jco'][v+1]   = row - i_s + 1             
            A['jco'][v+2]   = row - j_s
            A['jco'][v+3]   = row - j_s + 1
            A['jco'][v+4]   = row
            A['jco'][v+5]   = row + 1
            A['jco'][v+6]   = row + j_s           
            A['jco'][v+7]   = row + j_s + 1
            A['jco'][v+8]   = row + i_s
            A['jco'][v+9]   = row + i_s + 1
                
            A['jco'][v+10]  = row - i_s
            A['jco'][v+11]  = row - i_s + 1             
            A['jco'][v+12]  = row - j_s
            A['jco'][v+13]  = row - j_s + 1
            A['jco'][v+14]  = row
            A['jco'][v+15]  = row + 1
            A['jco'][v+16]  = row + j_s           
            A['jco'][v+17]  = row + j_s + 1
            A['jco'][v+18]  = row + i_s
            A['jco'][v+19]  = row + i_s + 1
            v += 20

        #Eastern grid cell
        row = 2 * (ny * x_i + ny - 1)
        A['beg'][row]   = v
        A['beg'][row+1] = v + 3 
        A['jco'][v]     = row - i_s
        A['jco'][v+1]   = row - i_s + 1            
        A['jco'][v+2]   = row
        A['jco'][v+3]   = row + 1 
        v += 4

    for y_i in range(ny):
        #Northern grid cell
        row             = 2 * (ny * (nx - 1) + y_i)
        A['beg'][row]   = v
        A['beg'][row+1] = v + 3
        A['jco'][v]     = row - j_s
        A['jco'][v+1]   = row - j_s + 1            
        A['jco'][v+2]   = row
        A['jco'][v+3]   = row + 1 
        v += 4
        
    #Last cell
    A['beg'][ndim]   = v
    #-------------------------------------------------------------------------
    v = 0
    for y_i in range(ny):
        #Southern grid cell
        A['co'][v]      = Alzz[y_i, 0, 4]
        A['co'][v+1]    = Alzz[y_i, 0, 5]
        A['co'][v+2]    = Alzp[y_i, 0, 5]
        A['co'][v+3]    = Alpp[y_i, 0, 4]
        v += 4
         
    for x_i in range(1, nx - 1):
        #Western grid cell
        A['co'][v]      = Alzz[0, x_i, 4]
        A['co'][v+1]    = Alzz[0, x_i, 7]
        A['co'][v+2]    = Alzp[0, x_i, 7]
        A['co'][v+3]    = Alpp[0, x_i, 4]
        v += 4
        
        for y_i in range(1, ny - 1):
            #Central grid cell
            #Vorticity equation
            A['co'][v]      = Alzz[y_i, x_i, 1]
            A['co'][v+1]    = Alzp[y_i, x_i, 1]
            A['co'][v+2]    = Alzz[y_i, x_i, 3]
            A['co'][v+3]    = Alzp[y_i, x_i, 3]
            A['co'][v+4]    = Alzz[y_i, x_i, 4]
            A['co'][v+5]    = Alzp[y_i, x_i, 4]
            A['co'][v+6]    = Alzz[y_i, x_i, 5]
            A['co'][v+7]    = Alzp[y_i, x_i, 5]
            A['co'][v+8]    = Alzz[y_i, x_i, 7]
            A['co'][v+9]    = Alzp[y_i, x_i, 7]

            #Psi equation
            A['co'][v+10]   = Alpz[y_i, x_i, 1]
            A['co'][v+11]   = Alpp[y_i, x_i, 1]
            A['co'][v+12]   = Alpz[y_i, x_i, 3]
            A['co'][v+13]   = Alpp[y_i, x_i, 3]
            A['co'][v+14]   = Alpz[y_i, x_i, 4]
            A['co'][v+15]   = Alpp[y_i, x_i, 4]
            A['co'][v+16]   = Alpz[y_i, x_i, 5]
            A['co'][v+17]   = Alpp[y_i, x_i, 5]
            A['co'][v+18]   = Alpz[y_i, x_i, 7]
            A['co'][v+19]   = Alpp[y_i, x_i, 7]
            v += 20

        #Eastern grid cell
        A['co'][v]      = Alzz[ny-1, x_i, 1]
        A['co'][v+1]    = Alzp[ny-1, x_i, 1]
        A['co'][v+2]    = Alzz[ny-1, x_i, 4]
        A['co'][v+3]    = Alpp[ny-1, x_i, 4]
        v += 4

    for y_i in range(ny):
        #Southern grid cell
        A['co'][v]      = Alzz[y_i, nx-1, 3]
        A['co'][v+1]    = Alzp[y_i, nx-1, 3]
        A['co'][v+2]    = Alzz[y_i, nx-1, 4]
        A['co'][v+3]    = Alpp[y_i, nx-1, 4]
        v += 4

    A = MatrixPack(A, ndim)
    A = MatrixSort(A, ndim)
    
    return A

def AssembleB(nx, ny, ndim, adim, Tlzz, Tlzp):
    """Generates the B matrix"""
    B           = {}
    B['beg']    = np.zeros(ndim+1)
    B['jco']    = np.zeros(ndim*800)
    B['co']     = np.zeros(ndim*800)
    B['di']     = np.zeros(ndim)

    v = 0
    
    for y_i in range(ny):
        #Southern grid cell
        row             = 2 * y_i
        B['beg'][row]   = v
        B['beg'][row+1] = v
        
    for x_i in range(1, nx - 1):
        #Western grid cell
        row             = 2 * ny * x_i
        B['beg'][row]   = v
        B['beg'][row+1] = v      
        
        for y_i in range(1, ny - 1):
            #Central grid cell
            row             = 2 * (ny * x_i + y_i)
            B['beg'][row]   = v
            B['beg'][row+1] = v + 2             
            B['jco'][v]     = row
            B['jco'][v+1]   = row + 1
            v += 2
            
        #Eastern grid cell
        row = 2 * (ny * x_i + ny - 1)
        B['beg'][row]   = v
        B['beg'][row+1] = v  
        
    for y_i in range(ny):
        #Northern grid cell
        row             = 2 * (ny * (nx - 1) + y_i)
        B['beg'][row]   = v
        B['beg'][row+1] = v
    
    #Last cell
    B['beg'][ndim]   = v
    
    v = 0
    for y_i in range(1, ny - 1):
        for x_i in range(1, nx - 1):
            B['co'][v]      = -Tlzz[y_i, x_i, 4]
            B['co'][v+1]    = -Tlzp[y_i, x_i, 4]
            v += 2
     
    B = MatrixPack(B, ndim)
    
    return B

def MatrixPack(Matrix, ndim):
    """Remove entries smaller than 10^-12"""
    vv = 0
    
    for i in range(ndim):
        begin = np.copy(vv)
                
        for v in range(int(Matrix['beg'][i]), int(Matrix['beg'][i+1])):
            if np.abs(Matrix['co'][v]) > 10**(-12.0):
                Matrix['co'][vv] = Matrix['co'][v]
                Matrix['jco'][vv] = Matrix['jco'][v]
                vv += 1
                
        Matrix['beg'][i] = begin
        
    #Last cell
    Matrix['beg'][ndim]   = vv

    return Matrix

def MatrixSort(Matrix, ndim):
    """Sort the Matrix"""
    for i in range(ndim):           
        for v in range(int(Matrix['beg'][i]), int(Matrix['beg'][i+1])-1): 
            k = np.copy(v)
            for vv in range(k+1, int(Matrix['beg'][i+1])):
                if Matrix['jco'][vv] < Matrix['jco'][k]:
                    k = np.copy(vv)
            
            dum              = Matrix['co'][k]
            jdum             = Matrix['jco'][k]
            Matrix['co'][k]  = Matrix['co'][v]
            Matrix['jco'][k] = Matrix['jco'][v]
            Matrix['jco'][v] = jdum
            Matrix['co'][v]  = dum

    return Matrix     

    
def U_to_Psi(nx, ny, vel):
    """Converts the ndim velocity field to 2D streamfunction field"""
    om  = np.zeros((ny, nx))
    ps  = np.zeros((ny, nx))
    
    for x_i in range(nx):
        for y_i in range(ny):
            row          = 2 * (ny * x_i + y_i)
            om[y_i, x_i] = vel[row]
            ps[y_i, x_i] = vel[row+1]

    return om, ps

def Nlin_RHS(dx, dy, nx, ny, F, om, ps):
    """Produces local matrices for nonlinear operators for the RHS"""
    
    #Components for the streamfunction
    U_dx    = Nonlin(dx, dy, nx, ny, om, ps, 1)
    V_dy    = Nonlin(dx, dy, nx, ny, om, ps, 2)
    
    #Determine the operators
    Nlzz    = U_dx + V_dy
    Nlzp    = - F * (U_dx + V_dy)

    return Nlzz, Nlzp

def Nlin_Jac(dx, dy, nx, ny, F, om, ps):
    """Produces local matrices for nonlinear operators for the Jacobian"""

    #Components for the streamfunction
    U_dx    = Nonlin(dx, dy, nx, ny, om, ps, 1)
    V_dy    = Nonlin(dx, dy, nx, ny, om, ps, 2)
    u_dZ_dx = Nonlin(dx, dy, nx, ny, om, ps, 3)
    v_dZ_dy = Nonlin(dx, dy, nx, ny, om, ps, 4)
    
    #Determine the operators
    Nlzz    = U_dx + V_dy
    Nlzp    = (u_dZ_dx + v_dZ_dy) - F * (U_dx + V_dy)
   
    return Nlzz, Nlzp
    
def Nonlin(dx, dy, nx, ny, om, ps, case):
    """Nonlinear terms for the zeta (vorticity) and psi (pressure) equation
    See page 118-119, Dynamical Oceanography, Henk Dijkstra"""
    
    #6 different cases for each component
    #1: U / dx
    #2: V / dy
    #3: u dZ / dx (Z = Zeta)
    #4: v dZ / dY (Z = Zeta)
    #5: u dP / dx
    #6: v dP / dy
    
    #Cell structure and index
    #2 5 
    #1 4 7
    #  3 6
    
    #First determine the distances
    dx_dy_4     = 1.0 / (4.0 * dx * dy)
    
    if case == 1:
        #the U / dx component
        U_dx    = np.zeros((ny, nx, 10))
        
        for y_i in range(1, ny - 1):
            for x_i in range(1, nx - 1):
                U_dx[y_i, x_i, 1]   = dx_dy_4 * (ps[y_i, x_i+1] - ps[y_i, x_i-1])
                U_dx[y_i, x_i, 7]   = -dx_dy_4 * (ps[y_i, x_i+1] - ps[y_i, x_i-1])

        return U_dx
    
    if case == 2:
        #the V / dy component
        V_dy    = np.zeros((ny, nx, 10))
        
        for y_i in range(1, ny - 1):
            for x_i in range(1, nx - 1):
                V_dy[y_i, x_i, 3]   = -dx_dy_4 * (ps[y_i+1, x_i] - ps[y_i-1, x_i])
                V_dy[y_i, x_i, 5]   = dx_dy_4 * (ps[y_i+1, x_i] - ps[y_i-1, x_i])

        return V_dy

    if case == 3:
        #the u dZ / dx component
        u_dZ_dx    = np.zeros((ny, nx, 10))
        
        for y_i in range(1, ny - 1):
            for x_i in range(1, nx - 1):
                u_dZ_dx[y_i, x_i, 3]   = dx_dy_4 * (om[y_i+1, x_i] - om[y_i-1, x_i])
                u_dZ_dx[y_i, x_i, 5]   = -dx_dy_4 * (om[y_i+1, x_i] - om[y_i-1, x_i])
                #u_dZ_dx[y_i, x_i, 1]   = dx_dy_4 * (om[y_i, x_i+1] - om[y_i, x_i-1])
                #u_dZ_dx[y_i, x_i, 7]   = -dx_dy_4 * (om[y_i, x_i+1] - om[y_i, x_i-1])
        return u_dZ_dx

    if case == 4:
        #the u dZ / dx component
        v_dZ_dy    = np.zeros((ny, nx, 10))
        
        for y_i in range(1, ny - 1):
            for x_i in range(1, nx - 1):
                v_dZ_dy[y_i, x_i, 1]   = -dx_dy_4 * (om[y_i, x_i+1] - om[y_i, x_i-1])
                v_dZ_dy[y_i, x_i, 7]   = dx_dy_4 * (om[y_i, x_i+1] - om[y_i, x_i-1])
                #v_dZ_dy[y_i, x_i, 3]   = -dx_dy_4 * (om[y_i+1, x_i] - om[y_i-1, x_i])
                #v_dZ_dy[y_i, x_i, 5]   = dx_dy_4 * (om[y_i+1, x_i] - om[y_i-1, x_i])
        return v_dZ_dy
    
def Boundaries(dx, dy, nx, ny, parameters, Alzz, Alzp, Alpz, Alpp):
    """Insert conditions at the real boundaries of the domain"""
    
    #Cell structure and index
    #2 5 
    #1 4 7
    #  3 6
    
    #East-West boundaries
    oml2    = parameters[5] / 2.0
    oml3    = -3. * parameters[5] / (dx * dx)
    omr2    = parameters[5] / 2.0
    omr3    = -3. * parameters[5] / (dx * dx)
    
    #North-South boundaries
    omb2    = parameters[6] / 2.0
    omb3    = -3. * parameters[6] / (dy * dy)
    omt2    = parameters[6] / 2.0
    omt3    = -3. * parameters[6] / (dy * dy)
    
    for x_i in range(1, nx - 1):
        #West (is this correct?)
        Alzz[0, x_i, 4] = 1.0
        Alzz[0, x_i, 7] = oml2
        Alzp[0, x_i, 7] = oml3
        Alpp[0, x_i, 4] = 1.0
        
        #East
        Alzz[ny-1, x_i, 4] = 1.0
        Alzz[ny-1, x_i, 1] = omr2
        Alzp[ny-1, x_i, 1] = omr3
        Alpp[ny-1, x_i, 4] = 1.0      
        
    for y_i in range(ny):
        #South
        Alzz[y_i, 0, 4] = 1.0
        Alzz[y_i, 0, 5] = omb2
        Alzp[y_i, 0, 5] = omb3
        Alpp[y_i, 0, 4] = 1.0
        
        #North
        Alzz[y_i, nx-1, 4] = 1.0
        Alzz[y_i, nx-1, 3] = omt2
        Alzp[y_i, nx-1, 3] = omt3
        Alpp[y_i, nx-1, 4]  = 1.0 

    return Alzz, Alzp, Alpz, Alpp
            
      
