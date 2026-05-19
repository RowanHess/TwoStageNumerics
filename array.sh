#!/bin/bash

#SBATCH -a 1-7
#SBATCH --cpus-per-task=8
#SBATCH --mem=16G
#SBATCH --partition=mit_normal
#SBATCH --time=0-01:00
#SBATCH -o run_\%a.out
#SBATCH -e run_\%a.err


source ~/.bashrc

#Load software
module load julia
module load gurobi

#Run the script as usualf
julia -t 8 simulations.jl $1 $SLURM_ARRAY_TASK_ID
