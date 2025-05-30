import multiprocessing
import subprocess
import sys
import os
from datetime import datetime

def run_sim(Re):
    try:
        print(f"[{datetime.now():%H:%M:%S}] [PID {os.getpid()}] Starting Re = {Re}", flush=True)

        # Run simulation silently — no log file unless it fails
        subprocess.run(
            [sys.executable, 'run_single_simulation.py', str(Re)],
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL,
            check=True
        )

        print(f"[{datetime.now():%H:%M:%S}] [PID {os.getpid()}] Finished Re = {Re}", flush=True)

    except subprocess.CalledProcessError as e:
        error_msg = (f"[{datetime.now():%H:%M:%S}] [PID {os.getpid()}] "
                     f"FAILED Re = {Re} — subprocess error\n")
        print(error_msg, flush=True)
        with open("error_log.txt", "a") as err_log:
            err_log.write(error_msg)

    except Exception as e:
        error_msg = (f"[{datetime.now():%H:%M:%S}] [PID {os.getpid()}] "
                     f"FAILED Re = {Re} — unexpected error: {e}\n")
        print(error_msg, flush=True)
        with open("error_log.txt", "a") as err_log:
            err_log.write(error_msg)

if __name__ == "__main__":
    reynolds_list = list(range(50, 52))  # Re = 50 to 160 inclusive
    num_cores = 2  # Adjust as needed

    print(f"Launching batch with {num_cores} parallel processes.")
    print(f"Reynolds numbers: {reynolds_list}")

    with multiprocessing.Pool(processes=num_cores) as pool:
        pool.map(run_sim, reynolds_list)
