#!/bin/bash

#SBATCH --cpus-per-task=8
#SBATCH --mem=20G
#SBATCH --partition=mit_normal
#SBATCH --time=0-01:00
#SBATCH -o run_initial.out
#SBATCH -e run_initial.err

source ~/.bashrc

#Load software
module load julia
module load gurobi

#Run the script as usual
julia -t 8 simulations.jl $1 0

