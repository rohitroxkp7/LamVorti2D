from solver import solve
import numpy as np

# Grid parameters
imax         = 181
jmax         = 181
r1           = 0.5
r2           = 50
plot_flag    = 0

# Flow parameters
Re           = 100
rot_speed    = 0#0.05
rot_time     = 2

# Time stepping settings
dtau         = np.inf
t_start      = 0
t_end        = 0.5
deltaT       = 0.1#np.inf

# Solution restart (BDF two time-step unsteady restart)
restart_flag = 0

# Monitor convergence
conv_flag    = 1

sim_name = "rohit"

steady_flag = 0

solve(imax, jmax, r1, r2, Re, rot_speed, rot_time, 
      dtau, t_start, t_end, deltaT, restart_flag, plot_flag, conv_flag, sim_name, steady_flag)