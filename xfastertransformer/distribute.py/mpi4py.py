from mpi4py import MPI
import os
import subprocess
import signal

def start_server(rank):
    cmd = ['python', 'serve.py']
    process = subprocess.Popen(cmd)
    return process

def stop_server(process):
    process.terminate()

def main():
    comm = MPI.COMM_WORLD
    rank = comm.Get_rank()
    size = comm.Get_size()

    if size != 2:
        print("This script should be run with exactly 2 MPI processes.")
        return

    # Start servers in both MPI processes
    if rank == 0 or rank == 1:
        process = start_server(rank)

    # Wait for servers to start
    comm.Barrier()

    # Stop servers in both MPI processes
    if rank == 0 or rank == 1:
        stop_server(process)

if __name__ == "__main__":
    main()
