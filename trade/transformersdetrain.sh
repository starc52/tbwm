#!/bin/bash
#SBATCH -A research
#SBATCH --gres=gpu:2
#SBATCH --mem=40000
#SBATCH --cpus-per-gpu=10
#SBATCH --mail-type=ALL
#SBATCH --mail-user=schakkera@cs.stonybrook.edu
#SBATCH --time=4-00:00:00
#SBATCH --output=op_transformersde_train.txt
#SBATCH --error=er_transformersde_train.txt



cd /home/schakkera/Robotics/trade/
conda run -n robo python traintransformersde.py --logdir exp_dir
