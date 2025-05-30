# run_single_simulation.py
from solver import solve
import numpy as np
import sys

# Get Reynolds number from command-line argument
Re = int(sys.argv[1])

# Grid parameters
imax = 181
jmax = 181
r1 = 0.5
r2 = 50
plot_flag = 0

# Flow parameters
rot_speed = 0.05
rot_time = 2

# Time stepping settings
dtau = np.inf
t_start = 0
t_end = 0.2
deltaT = 0.1

# Restart and convergence
restart_flag = 0
conv_flag = 0

# Simulation name
sim_name = f"Re_{Re}_with_rot"

# Call solver
solve(imax, jmax, r1, r2, Re, rot_speed, rot_time, dtau,
      t_start, t_end, deltaT, restart_flag, plot_flag, conv_flag, sim_name)
