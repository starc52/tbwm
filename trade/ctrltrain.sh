#!/bin/bash
#SBATCH -A research
#SBATCH --gres=gpu:2
#SBATCH --mem=40000
#SBATCH --cpus-per-gpu=10
#SBATCH --mail-type=ALL
#SBATCH --mail-user=schakkera@cs.stonybrook.edu
#SBATCH --time=4-00:00:00
#SBATCH --output=op_mdrnn_train.txt
#SBATCH --error=er_mdrnn_train.txt



cd /home/schakkera/Robotics/world-models/
xvfb-run -s "-screen 0 1400x900x24" python traincontroller.py --logdir exp_dir --n-samples 16 --pop-size 64 --target-return 950 --display 2>&1 | tee -a op_ctrl_train.txt
conda run -n robo python trainmdrnn.py --logdir exp_dir
