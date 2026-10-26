#!/bin/bash

#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH --partition=mit_normal
#SBATCH --time=0-01:00
#SBATCH -o run_acc_%j.out
#SBATCH -e run_acc_%j.err


source ~/.bashrc

#Load software
module load julia
module load gurobi

#Run the script as usual
julia -t 8 simulations.jl $1 9