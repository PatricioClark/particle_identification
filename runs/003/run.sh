#!/bin/bash
#SBATCH -J pid003 # Job name
#SBATCH -p normal # Slurm partition
#SBATCH -o job.%j.out # Name of stdout error and output file
#SBATCH -N 1 # Total number of nodes requested
#SBATCH -n 18 # Total number of mpi tasks requested
#SBATCH -t 48:00:00 # Run time (hh:mm:ss)
#SBATCH -w g1,g2

# OpenMPI transport layers
export OMPI_MCA_pml="ob1"
export OMPI_MCA_btl="self,vader,tcp"
export OMPI_MCA_btl_tcp_if_include="192.168.1.0/24"
# Executable file
EXEC=./GHOST
# Launch MPI job
mpirun --use-hwthread-cpus --bind-to hwthread ${EXEC}
