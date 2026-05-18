#!/bin/bash

#SBATCH -a 1-8
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

#Run the script as usual
julia simulations.jl 10 $SLURM_ARRAY_TASK_ID
