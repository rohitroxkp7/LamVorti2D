import numpy as np
import scipy.sparse as sp
import os
from scipy.sparse import vstack, hstack
from scipy.sparse.linalg import spsolve

from transformation_metrics import grid_metrics_cylinder

def solve(imax, jmax, r1, r2, Re, speed, rot_time, dtau, t_start, t_end, deltaT, restart_flag, plot_flag, conv_flag, simulation_name, steady_flag):
    
    metrics = grid_metrics_cylinder(imax,jmax,r1,r2,plot_flag)
    alfa , beta , gama , P , Q , Jac , kc,kn,ks,ke,kw,knw,kne,ksw,kse,x,y, detadx , detady , alen2= metrics.compute_metrics()

    #t_start = 0
    #t_end = 400
    #deltaT = 0.1
    N = imax * jmax
    
    
    # Grid spacing
    dksi = 1.0 / (imax - 1)
    deta = 1.0 / (jmax - 1)
    
    # Pseudo-time step and convergence criteria
    #dtau = np.inf
    residual = 1.0e5
    epsilon = 1.0e-10
    
    # Computational coordinates
    ksii = np.zeros((imax, jmax))
    etaa = np.zeros((imax, jmax))
    for i in range(imax):
        for j in range(jmax):
            ksii[i, j] = dksi * i
            etaa[i, j] = deta * j
    
    # Sparse coefficient matrices (analogous to spalloc in MATLAB)
    
    APsiPsi = sp.lil_matrix((N, N))
    APsiOme = sp.lil_matrix((N, N))
    AOmePsi = sp.lil_matrix((N, N))
    AOmeOme = sp.lil_matrix((N, N))
    
    # Right-hand side vectors
    bPsi = np.zeros(imax * jmax)
    bOme = np.zeros(imax * jmax)
    
    # Solution variables
    deltaPsi   = np.zeros(imax * jmax)
    deltaOme   = np.zeros(imax * jmax)
    
    if restart_flag==1:
        psi       = np.loadtxt('psi_restart_file.csv')
        omeold    = np.loadtxt('ome_old1_restart_file.csv')
        ome       = np.loadtxt('ome_old1_restart_file.csv')
        omeoldold = np.loadtxt('ome_old2_restart_file.csv')
    else:
        psi        = np.zeros(imax * jmax)
        omeold     = np.zeros(imax * jmax)
        omeoldold  = np.zeros(imax * jmax)
        ome        = np.zeros(imax * jmax)
    
    for i in range(imax):
        for j in range(1, jmax - 1):  # j = 2 to jmax-1 in MATLAB
    
            k = kc[i, j] - 1
            n = kn[i, j] - 1
            s = ks[i, j] - 1
    
            if i == 0:
                e  = ke[i, j] - 1
                ne = kne[i, j] - 1
                se = kse[i, j] - 1
                w  = k + imax - 2
                nw = n + imax - 2
                sw = s + imax - 2
            elif i == imax - 1:
                e  = k - imax + 2
                ne = n - imax + 2
                se = s - imax + 2
                w  = kw[i, j] - 1
                nw = knw[i, j] - 1
                sw = ksw[i, j] - 1
            else:
                e  = ke[i, j] - 1
                ne = kne[i, j] - 1
                se = kse[i, j] - 1
                w  = kw[i, j] - 1
                nw = knw[i, j] - 1
                sw = ksw[i, j] - 1
    
            # At current node k
            APsiPsi[k, k] = 1.0 / dtau + (2.0 * alfa[k]) / (dksi**2 * Re) + (2.0 * beta[k]) / (deta**2 * Re)
    
            # NEWS neighbors
            APsiPsi[k, e] = -(alfa[k] / (Re * dksi**2) + P[k] / (2 * Re * dksi))
            APsiPsi[k, w] = -(alfa[k] / (Re * dksi**2) - P[k] / (2 * Re * dksi))
            APsiPsi[k, n] = -(beta[k] / (Re * deta**2) + Q[k] / (2 * Re * deta))
            APsiPsi[k, s] = -(beta[k] / (Re * deta**2) - Q[k] / (2 * Re * deta))
    
            # Cross neighbors (second mixed derivatives)
            APsiPsi[k, ne] = -(2 * gama[k]) / (4 * Re * dksi * deta)
            APsiPsi[k, nw] =  (2 * gama[k]) / (4 * Re * dksi * deta)
            APsiPsi[k, se] =  (2 * gama[k]) / (4 * Re * dksi * deta)
            APsiPsi[k, sw] = -(2 * gama[k]) / (4 * Re * dksi * deta)
    
            # Coupling with omega
            APsiOme[k, k] = -1.0 / Re
    
    # Far-field boundary condition (j = jmax in MATLAB → j = jmax - 1 in Python)
    j = jmax - 1
    for i in range(imax):
        k = kc[i, j] - 1
        psi[k] = y[k]  # y is assumed to be 1D, same as in your earlier setup
        ome[k] = 0.0   # Not necessary since ome is already initialized to zero
                      # but included for clarity
                      
    # Time step and iteration counter
    iter = 0
    
    # Create output directories if they don't exist
    base_dir = simulation_name
    os.makedirs(os.path.join(base_dir, "psi_data"), exist_ok=True)
    os.makedirs(os.path.join(base_dir, "omega_data"), exist_ok=True)
    os.makedirs(os.path.join(base_dir, "residual_data"), exist_ok=True)
    
    if steady_flag==1:
        t1    = 0
        t2    = 1
        tstep = 1
    else:
        t1    = t_start
        t2    = t_end + deltaT
        tstep = deltaT
    
    for t in np.arange(t1, t2, tstep):
        
        residual = 1.0e5
        residual_vec = []
    
        while residual > epsilon:
            for i in range(imax):
                for j in range(1, jmax - 1):
                    k = kc[i, j] - 1
                    n = kn[i, j] - 1
                    s = ks[i, j] - 1
            
                    if i == 0:
                        e  = ke[i, j] - 1
                        ne = kne[i, j] - 1
                        se = kse[i, j] - 1
                        w  = (k + imax - 2) 
                        nw = (n + imax - 2) 
                        sw = (s + imax - 2) 
                    elif i == imax - 1:
                        e  = (k - imax + 2)
                        ne = (n - imax + 2) 
                        se = (s - imax + 2) 
                        w  = kw[i, j] - 1
                        nw = knw[i, j] - 1
                        sw = ksw[i, j] - 1
                    else:
                        e  = ke[i, j] - 1
                        ne = kne[i, j] - 1
                        se = kse[i, j] - 1
                        w  = kw[i, j] - 1
                        nw = knw[i, j] - 1
                        sw = ksw[i, j] - 1
                    
                    Aa = (psi[n] - psi[s]) / (2 * deta)
                    Bb = (ome[e] - ome[w]) / (2 * dksi)
                    Cc = (psi[e] - psi[w]) / (2 * dksi)
                    Dd = (ome[n] - ome[s]) / (2 * deta)
                    
                    Ee = (alfa[k] / Re) * ((ome[e] - 2 * ome[k] + ome[w]) / (dksi ** 2))
                    Ff = 2 * (gama[k] / Re) * ((ome[ne] - ome[nw] - ome[se] + ome[sw]) / (4 * deta * dksi))
                    Gg = (beta[k] / Re) * ((ome[n] - 2 * ome[k] + ome[s]) / (deta ** 2))
                    Hh = (P[k] / Re) * ((ome[e] - ome[w]) / (2 * dksi))
                    Ii = (Q[k] / Re) * ((ome[n] - ome[s]) / (2 * deta))
                    
                    Pp = (alfa[k] / Re) * ((psi[e] - 2 * psi[k] + psi[w]) / (dksi ** 2))
                    Qq = 2 * (gama[k] / Re) * ((psi[ne] - psi[nw] - psi[se] + psi[sw]) / (4 * deta * dksi))
                    Rr = (beta[k] / Re) * ((psi[n] - 2 * psi[k] + psi[s]) / (deta ** 2))
                    Ss = (P[k] / Re) * ((psi[e] - psi[w]) / (2 * dksi))
                    Tt = (Q[k] / Re) * ((psi[n] - psi[s]) / (2 * deta))
                    
                    AOmeOme[k, k]  = 1 / dtau + (2.0 * alfa[k]) / (Re * dksi ** 2) + (2.0 * beta[k]) / (Re * deta ** 2) + (1.5 / deltaT)
                    AOmeOme[k, e]  =  (Jac[k] * Aa) / (2 * dksi) - alfa[k] / (Re * dksi ** 2) - P[k] / (2 * Re * dksi)
                    AOmeOme[k, w]  = -(Jac[k] * Aa) / (2 * dksi) - alfa[k] / (Re * dksi ** 2) + P[k] / (2 * Re * dksi)
                    AOmeOme[k, n]  = -(Jac[k] * Cc) / (2 * dksi) - beta[k] / (Re * deta ** 2) - Q[k] / (2 * Re * deta)
                    AOmeOme[k, s]  =  (Jac[k] * Cc) / (2 * dksi) - beta[k] / (Re * deta ** 2) + Q[k] / (2 * Re * deta)
                    AOmeOme[k, ne] = -2 * gama[k] / (4 * Re * dksi * deta)
                    AOmeOme[k, nw] =  2 * gama[k] / (4 * Re * dksi * deta)
                    AOmeOme[k, se] =  2 * gama[k] / (4 * Re * dksi * deta)
                    AOmeOme[k, sw] = -2 * gama[k] / (4 * Re * dksi * deta)
                    
                    AOmePsi[k, n] =  (Jac[k] * Bb) / (2 * deta)
                    AOmePsi[k, s] = -(Jac[k] * Bb) / (2 * deta)
                    AOmePsi[k, e] = -(Jac[k] * Dd) / (2 * dksi)
                    AOmePsi[k, w] =  (Jac[k] * Dd) / (2 * dksi)
                    
                    bPsi[k] = Pp + Qq + Rr + Ss + Tt + ome[k] / Re
                    bOme[k] = -(Jac[k] * Aa * Bb) + (Jac[k] * Cc * Dd) + Ee + Ff + Gg + Hh + Ii - (
                        (3 * ome[k] - 4 * omeold[k] + omeoldold[k]) / (2 * deltaT)
                    )
                    
            
            # Far field (Top boundary)
            # Psi and Ome are fixed constant values.
            # psi = y and ome = 0
            j = jmax - 1  # Python 0-based indexing
            
            for i in range(imax):
                k = kc[i, j] - 1
            
                APsiPsi[k, k] = 1.0
                bPsi[k] = 0.0  # or: y[k] - psi[k]
            
                AOmeOme[k, k] = 1.0
                bOme[k] = 0.0  # or: -ome[k]
            
            # Bottom boundary (wall)
            # Applies to i = 0 to imax-1 (inclusive)
            j = 0  # Python indexing
            
            for i in range(imax):
                k = kc[i, j] - 1
                n = kn[i, j] - 1 
                nn = n + imax  # Assuming linear index increment
            
                AOmeOme[k, k]  = -1.0
                AOmePsi[k, k]  = (7.0 / 2.0) * (beta[k] / (deta ** 2))
                AOmePsi[k, n]  = -(8.0 / 2.0) * (beta[k] / (deta ** 2))
                AOmePsi[k, nn] = (1.0 / 2.0) * (beta[k] / (deta ** 2))
            
                APsiPsi[k, k] = 1.0
                bPsi[k] = 0.0  # or -psi[k] if enforcing Dirichlet strongly
            
                theta = 2 * np.pi * (1.0 - alen2[i])
                aaa = speed if t <= rot_time else 0.0
            
                denom = detadx[k] * np.cos(theta) + detady[k] * np.sin(theta)
                term1 = (7.0 / 2.0) * (beta[k] / (deta ** 2)) * psi[k]
                term2 = (8.0 / 2.0) * (beta[k] / (deta ** 2)) * psi[n]
                term3 = (1.0 / 2.0) * (beta[k] / (deta ** 2)) * psi[nn]
                forcing = (Q[k] - 3.0 * (beta[k] / deta)) * (aaa / denom)
            
                bOme[k] = ome[k] - term1 + term2 - term3 - forcing
            
            # Combine sparse blocks into one large system matrix A
            A = vstack([
                hstack([APsiPsi, APsiOme]),
                hstack([AOmePsi, AOmeOme])
                ]).tocsr()
            
            b = np.concatenate([bPsi, bOme])
            
            # Solve the linear system on GPU
            sol = spsolve(A, b)
            
            # Extract solution components
            deltaPsi = sol[:imax * jmax]
            deltaOme = sol[imax * jmax:]
            
            # Update state vectors
            psi += deltaPsi
            ome += deltaOme
            
            iter += 1
            residual = np.linalg.norm(sol) / (2 * imax * jmax)
            residual_vec.append(np.log10(residual))
            iter_vector = np.arange(1, len(residual_vec) + 1).reshape(-1, 1)
            if conv_flag==1:
                print("Inner-iter: " , iter , "\n" , "Residual: " , residual , "\n" , "Physical time: " , t , "\n")
        
        
        psi_column = psi.reshape(-1, 1)
        ome_column = ome.reshape(-1, 1)
        
        time_idx = round(t / deltaT)
        psi_filename      = os.path.join(base_dir, f'psi_data/psi_t{time_idx:04d}.csv')
        omega_filename    = os.path.join(base_dir, f'omega_data/omega_t{time_idx:04d}.csv')
        residual_filename = os.path.join(base_dir, f'residual_data/residual_history_t{time_idx:04d}.csv')
        
        # Save psi and omega
        np.savetxt(psi_filename, psi_column, delimiter=",")
        np.savetxt(omega_filename, ome_column, delimiter=",")
        
        # Save residual history
        iter_vector = np.arange(1, len(residual_vec) + 1).reshape(-1, 1)
        residual_table = np.hstack((iter_vector, np.array(residual_vec).reshape(-1, 1)))
        np.savetxt(residual_filename, residual_table, delimiter=",")
        
        omeoldold = omeold.copy()
        omeold = ome.copy()
    #print("Simulation complete")
