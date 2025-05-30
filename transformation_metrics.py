import numpy as np

class grid_metrics_cylinder:
    def __init__(self, imax, jmax, r1, r2, plot_flag):
        
        self.imax = imax
        self.jmax = jmax
        self.flag = plot_flag
        self.r1   = r1
        self.r2   = r2

    def compute_metrics(self):
        
        imax = self.imax # 181
        jmax = self.jmax
        
        inner_rad =  self.r1 # 0.5
        outer_rad = self.r2  # 50.0
        
        # Set dksi and deta
        dksi = 1.0 / (imax - 1)
        deta = 1.0 / (jmax - 1)
        
        # Arrays for index pointers
        kc  = np.zeros((imax, jmax), dtype=int)
        ke  = np.zeros((imax, jmax), dtype=int)
        kw  = np.zeros((imax, jmax), dtype=int)
        ks  = np.zeros((imax, jmax), dtype=int)
        kn  = np.zeros((imax, jmax), dtype=int)
        kne = np.zeros((imax, jmax), dtype=int)
        knw = np.zeros((imax, jmax), dtype=int)
        kse = np.zeros((imax, jmax), dtype=int)
        ksw = np.zeros((imax, jmax), dtype=int)
        
        # Arrays for metrics
        N = imax * jmax
        eta   = np.zeros(N)
        ksi   = np.zeros(N)
        x     = np.zeros(N)
        y     = np.zeros(N)
        Jac   = np.zeros(N)
        alfa  = np.zeros(N)
        beta  = np.zeros(N)
        gama  = np.zeros(N)
        P     = np.zeros(N)
        Q     = np.zeros(N)
        
        # Form the pointer system
        for i in range(1, imax + 1):
            for j in range(1, jmax + 1):
        
                # Convert to 0-based indices for Python
                ip = i - 1
                jp = j - 1
        
                kc[ip, jp]  = imax * jp + i
                ke[ip, jp]  = kc[ip, jp] + 1
                kw[ip, jp]  = kc[ip, jp] - 1
                kn[ip, jp]  = kc[ip, jp] + imax
                ks[ip, jp]  = kc[ip, jp] - imax
        
                if i == 1:
                    kw[ip, jp] = kc[ip, jp] - 2 + imax
                elif i == imax:
                    ke[ip, jp] = kc[ip, jp] + 2 - imax
        
                kne[ip, jp] = ke[ip, jp] + imax
                knw[ip, jp] = kw[ip, jp] + imax
                kse[ip, jp] = ke[ip, jp] - imax
                ksw[ip, jp] = kw[ip, jp] - imax
        
                # Define mesh in computational domain
                ksi[kc[ip, jp] - 1] = dksi * (i - 1)   # -1 because kc is 1-based in MATLAB
                eta[kc[ip, jp] - 1] = deta * (j - 1)
        
        # Initialize alen and alen2
        alen = np.zeros(jmax)
        alen[1] = 1.0
        alen[2] = 2.0
        alen[3] = 3.0
        
        # Fill alen starting from j = 5 
        for j in range(4, jmax):
            alen[j] = alen[j - 1] + (j - 2) ** 1  # j - 3 + 1
        
        # Normalize
        alen /= alen[jmax - 1]  # jmax index is jmax - 1 in Python
        
        # Initialize alen2
        alen2 = np.zeros(imax)
        for i in range(1, imax):
            alen2[i] = alen2[i - 1] + min(i, imax - i) ** 0.6
        
        # Normalize
        alen2 /= alen2[imax - 1]
        
        # x and y are already defined earlier as 1D arrays of size imax * jmax
        xg = np.zeros((imax, jmax))
        yg = np.zeros((imax, jmax))
        
        # Generate grid in physical domain using clustering
        for i in range(1, imax + 1):
            for j in range(1, jmax + 1):
                ip = i - 1
                jp = j - 1
                k = kc[ip, jp] - 1  
        
                theta = 2 * np.pi * (1 - alen2[ip])
                rad = inner_rad + (outer_rad - inner_rad) * alen[jp]
        
                x[k] = rad * np.cos(theta)
                y[k] = rad * np.sin(theta)
        
        # Map 1D x, y arrays to 2D xg, yg for plotting
        for i in range(1, imax + 1):
            for j in range(1, jmax + 1):
                ip = i - 1
                jp = j - 1
                k = kc[ip, jp] - 1
                xg[ip, jp] = x[k]
                yg[ip, jp] = y[k]
        
        # Plot the O-Grid

        if self.flag==1:
            
            import matplotlib.pyplot as plt
            import niceplots
            
            plt.style.use(niceplots.get_style())
            
            fig = plt.figure(figsize=(8, 4))
            ax = fig.add_subplot(111)
            
            # Plot the computational mesh
            for i in range(imax):
                ax.plot(xg[i, :], yg[i, :], color='k', linewidth=0.5)  # horizontal lines
            for j in range(jmax):
                ax.plot(xg[:, j], yg[:, j], color='k', linewidth=0.5)  # vertical lines
            
            # Set axis properties
            ax.set_aspect('equal')  # like axis equal
            ax.set_xlim([-2, 16])
            ax.set_ylim([-4, 4])
            
            plt.title("Computational Mesh")
            plt.tight_layout()
            plt.show()
        
        
        # First derivatives
        dxdksi  = np.zeros(N)
        dxdeta  = np.zeros(N)
        dydksi  = np.zeros(N)
        dydeta  = np.zeros(N)
        
        # Second derivatives
        ddxdksidksi   = np.zeros(N)
        ddxdetadeta   = np.zeros(N)
        ddxdksideta   = np.zeros(N)
        ddydksidksi   = np.zeros(N)
        ddydetadeta   = np.zeros(N)
        ddydksideta   = np.zeros(N)
        
        # Compute first and second derivatives at interior points
        for i in range(imax):
            for j in range(1, jmax - 1):  # skip j=0 and j=jmax-1 (boundary)
        
                k  = kc[i, j] - 1  # 1D index 
                ke_ = ke[i, j] - 1
                kw_ = kw[i, j] - 1
                kn_ = kn[i, j] - 1
                ks_ = ks[i, j] - 1
        
                # First derivatives
                dxdksi[k]  = (x[ke_] - x[kw_]) / (2 * dksi)
                dydksi[k]  = (y[ke_] - y[kw_]) / (2 * dksi)
                dxdeta[k]  = (x[kn_] - x[ks_]) / (2 * deta)
                dydeta[k]  = (y[kn_] - y[ks_]) / (2 * deta)
        
                # Second derivatives
                ddxdksidksi[k] = (x[ke_] - 2 * x[k] + x[kw_]) / dksi**2
                ddydksidksi[k] = (y[ke_] - 2 * y[k] + y[kw_]) / dksi**2
                ddxdetadeta[k] = (x[kn_] - 2 * x[k] + x[ks_]) / deta**2
                ddydetadeta[k] = (y[kn_] - 2 * y[k] + y[ks_]) / deta**2
        
        # Bottom boundary (j = 0 in Python)
        j = 0
        for i in range(imax):
            k   = kc[i, j] - 1
            ke_ = ke[i, j] - 1
            kw_ = kw[i, j] - 1
        
            # First derivatives
            dxdksi[k] = (x[ke_] - x[kw_]) / (2 * dksi)
            dydksi[k] = (y[ke_] - y[kw_]) / (2 * dksi)
        
            # Second derivatives
            ddxdksidksi[k] = (x[ke_] - 2 * x[k] + x[kw_]) / dksi**2
            ddydksidksi[k] = (y[ke_] - 2 * y[k] + y[kw_]) / dksi**2
        
        # Bottom boundary (eta-derivatives) — j = 0 in Python
        j = 0
        for i in range(imax):
            k     = kc[i, j] - 1
            kn0   = kn[i, j]     - 1  # kn(i,j)
            kn1   = kn[i, j + 1] - 1  # kn(i,j+1)
            kn2   = kn[i, j + 2] - 1  # kn(i,j+2)
        
            # First η-derivatives (forward-biased stencil)
            dxdeta[k] = (-3 * x[k] + 4 * x[kn0] - x[kn1]) / (2 * deta)
            dydeta[k] = (-3 * y[k] + 4 * y[kn0] - y[kn1]) / (2 * deta)
        
            # Second η-derivatives
            ddxdetadeta[k] = (2 * x[k] - 5 * x[kn0] + 4 * x[kn1] - x[kn2]) / (deta**2)
            ddydetadeta[k] = (2 * y[k] - 5 * y[kn0] + 4 * y[kn1] - y[kn2]) / (deta**2)
        
        # Top boundary (xi-derivatives) — j = jmax - 1 in Python
        j = jmax - 1
        for i in range(imax):
            k   = kc[i, j] - 1
            ke_ = ke[i, j] - 1
            kw_ = kw[i, j] - 1
        
            # First derivatives
            dxdksi[k] = (x[ke_] - x[kw_]) / (2 * dksi)
            dydksi[k] = (y[ke_] - y[kw_]) / (2 * dksi)
        
            # Second derivatives
            ddxdksidksi[k] = (x[ke_] - 2 * x[k] + x[kw_]) / dksi**2
            ddydksidksi[k] = (y[ke_] - 2 * y[k] + y[kw_]) / dksi**2
        
        # Top boundary (eta-derivatives) — j = jmax - 1 in Python
        j = jmax - 1
        for i in range(imax):
            k     = kc[i, j] - 1
            ks0   = ks[i, j]     - 1  # ks(i, j)
            ks1   = ks[i, j - 1] - 1  # ks(i, j-1)
            ks2   = ks[i, j - 2] - 1  # ks(i, j-2)
        
            # First η-derivatives (backward-biased stencil)
            dxdeta[k] = ( 3 * x[k] - 4 * x[ks0] + x[ks1]) / (2 * deta)
            dydeta[k] = ( 3 * y[k] - 4 * y[ks0] + y[ks1]) / (2 * deta)
        
            # Second η-derivatives
            ddxdetadeta[k] = (2 * x[k] - 5 * x[ks0] + 4 * x[ks1] - x[ks2]) / (deta**2)
            ddydetadeta[k] = (2 * y[k] - 5 * y[ks0] + 4 * y[ks1] - y[ks2]) / (deta**2)
        
        for i in range(imax):
            for j in range(jmax):
                k   = kc[i, j] - 1
                ke_ = ke[i, j] - 1
                kw_ = kw[i, j] - 1
        
                ddxdksideta[k] = (dxdeta[ke_] - dxdeta[kw_]) / (2 * dksi)
                ddydksideta[k] = (dydeta[ke_] - dydeta[kw_]) / (2 * dksi)
        
        for i in range(imax):
            for j in range(jmax):
                k = kc[i, j] - 1
                Jac[k] = 1.0 / (
                    dxdksi[k] * dydeta[k] - dxdeta[k] * dydksi[k]
                )
                
        dksidx  = np.zeros(N)
        dksidy  = np.zeros(N)
        detadx  = np.zeros(N)
        detady  = np.zeros(N)
        
        alfa    = np.zeros(N)
        beta    = np.zeros(N)
        gama    = np.zeros(N)
        
        P       = np.zeros(N)
        Q       = np.zeros(N)
        
        for i in range(imax):
            for j in range(jmax):
                k = kc[i, j] - 1  
        
                # Derivatives of computational coords w.r.t physical coords
                dksidx[k]  =  Jac[k] * dydeta[k]
                dksidy[k]  = -Jac[k] * dxdeta[k]
                detadx[k]  = -Jac[k] * dydksi[k]
                detady[k]  =  Jac[k] * dxdksi[k]
        
                # Metric coefficients
                alfa[k] = dksidx[k]**2 + dksidy[k]**2
                beta[k] = detadx[k]**2 + detady[k]**2
                gama[k] = dksidx[k] * detadx[k] + dksidy[k] * detady[k]
        
                # Forcing term P
                P[k] = -(
                    alfa[k] * (ddxdksidksi[k] * dksidx[k] + ddydksidksi[k] * dksidy[k]) +
                    2 * gama[k] * (ddxdksideta[k] * dksidx[k] + ddydksideta[k] * dksidy[k]) +
                    beta[k] * (ddxdetadeta[k] * dksidx[k] + ddydetadeta[k] * dksidy[k])
                )
        
                # Forcing term Q
                Q[k] = -(
                    alfa[k] * (ddxdksidksi[k] * detadx[k] + ddydksidksi[k] * detady[k]) +
                    2 * gama[k] * (ddxdksideta[k] * detadx[k] + ddydksideta[k] * detady[k]) +
                    beta[k] * (ddxdetadeta[k] * detadx[k] + ddydetadeta[k] * detady[k])
                )
        
        return alfa , beta , gama , P , Q , Jac , kc,kn,ks,ke,kw,knw,kne,ksw,kse,x,y, detadx , detady , alen2
        


        
        
        
        
        
        
        
        
        
        
        
