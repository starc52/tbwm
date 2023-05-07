""" Test controller """
import argparse
from os.path import join, exists
from os import mkdir
from utils.misc import RolloutGenerator
import torch
import cv2

parser = argparse.ArgumentParser()
parser.add_argument('--logdir', type=str, help='Where models are stored.')
args = parser.parse_args()

ctrl_file = join(args.logdir, 'ctrl', 'best.tar')

assert exists(ctrl_file),\
    "Controller was not trained..."

sample_directory = join(args.logdir, 'test_sample')
if not exists(sample_directory):
    mkdir(sample_directory)

device = torch.device('cpu')

generator = RolloutGenerator(args.logdir, device, 1000, test_save_directory=sample_directory)

with torch.no_grad():
    cum_reward=generator.rollout(None)
    print("Cummulative reward for episode: ", cum_reward)
