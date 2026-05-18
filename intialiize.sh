#!/bin/bash

#SBATCH --cpus-per-task=8
#SBATCH --mem=16G
#SBATCH --partition=mit_normal
#SBATCH --time=0-01:00
#SBATCH -o run_initial.out
#SBATCH -e run_initial.err

source ~/.bashrc

#Load software
module load julia
module load gurobi

#Run the script as usual
julia simulations.jl 10 0

