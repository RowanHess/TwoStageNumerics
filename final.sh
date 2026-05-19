#!/bin/bash

#SBATCH --cpus-per-task=1
#SBATCH --mem=8G
#SBATCH --partition=mit_normal
#SBATCH --time=0-0:10
#SBATCH -o run_final.out
#SBATCH -e run_final.err


source ~/.bashrc

#Load software
module load julia
module load gurobi

#Run the script as usual
julia simulations.jl $1 9