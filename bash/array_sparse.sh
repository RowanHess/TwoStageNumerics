#!/bin/bash

#SBATCH -a 1-4
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH --partition=mit_normal
#SBATCH --time=0-01:00
#SBATCH -o run_\%a.out
#SBATCH -e run_\%a.err


source ~/.bashrc

#Load software
module load julia
module load gurobi


julia -t 8 sparse/simulation_sparse.jl $1 $SLURM_ARRAY_TASK_ID