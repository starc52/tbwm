#!/bin/bash
#SBATCH -A research
#SBATCH --gres=gpu:2
#SBATCH --mem=40000
#SBATCH --cpus-per-gpu=10
#SBATCH --mail-type=ALL
#SBATCH --mail-user=schakkera@cs.stonybrook.edu
#SBATCH --time=4-00:00:00
#SBATCH --output=op_generate.txt
#SBATCH --error=er_generate.txt



cd /home/schakkera/Robotics/world-models/
conda run -n robo python data/generation_script.py --rollouts 1000 --rootdir datasets/carracing --threads 8
