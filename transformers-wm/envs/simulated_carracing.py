"""
Simulated carracing environment.
"""
import argparse
from os.path import join, exists
import torch
from torch.distributions.categorical import Categorical
import gym
from gym import spaces
from models.vae import VAE
from models import MDRNNCell, TransformersDE
from utils.misc import LSIZE, RSIZE, RED_SIZE, LBUFFER_SIZE

import numpy as np


class SimulatedCarracing(gym.Env): # pylint: disable=too-many-instance-attributes
    """
    Simulated Car Racing.

    Gym environment using learnt VAE and MDRNN to simulate the
    CarRacing-v0 environment.

    :args directory: directory from which the vae and transformers_de are
    loaded.
    """
    def __init__(self, directory):
        vae_file = join(directory, 'vae', 'best.tar')
        transformers_de_file = join(directory, 'transformers_de', 'best.tar')
        assert exists(vae_file), "No VAE model in the directory..."
        assert exists(transformers_de_file), "No TransformersDE model in the directory..."

        # spaces
        self.action_space = spaces.Box(np.array([-1, 0, 0]), np.array([1, 1, 1]))
        self.observation_space = spaces.Box(low=0, high=255, shape=(RED_SIZE, RED_SIZE, 3),
                                            dtype=np.uint8)

        # load VAE
        vae = VAE(3, LSIZE)
        vae_state = torch.load(vae_file, map_location=lambda storage, location: storage)
        print("Loading VAE at epoch {}, "
              "with test error {}...".format(
                  vae_state['epoch'], vae_state['precision']))
        vae.load_state_dict(vae_state['state_dict'])
        self._decoder = vae.decoder

        # load MDRNN
        self._transformers_de = TransformersDE(32, 3, RSIZE, 5)
        transformers_de_state = torch.load(transformers_de_file, map_location=lambda storage, location: storage)
        print("Loading TransformersDE at epoch {}, "
              "with test error {}...".format(
                  transformers_de_state['epoch'], transformers_de_state['precision']))
        self._transformers_de.load_state_dict(transformers_de_state_dict)

        # init state
        self._lstate = torch.randn(1, LSIZE)
        self._astate = None
        self._hstate = torch.zeros(1, RSIZE)

        self.prev_latents_buffer_size = LBUFFER_SIZE

        # obs
        self._obs = None
        self._visual_obs = None

        # rendering
        self.monitor = None
        self.figure = None

    def reset(self):
        """ Resetting """
        import matplotlib.pyplot as plt
        self._lstate = torch.randn(1, LSIZE)
        self._astate = None
        # also reset monitor
        if not self.monitor:
            self.figure = plt.figure()
            self.monitor = plt.imshow(
                np.zeros((RED_SIZE, RED_SIZE, 3),
                         dtype=np.uint8))

    def step(self, action):
        """ One step forward """
        with torch.no_grad():
            action = torch.Tensor(action).unsqueeze(0)
            if self._astate is None:
                self._astate = action
            else:
                self._astate = torch.cat([self._astate, action], dim=0)
            if self._astate.size(0) > self.prev_latents_buffer_size:
                self._astate = self._astate[-self.prev_latents_buffer_size:]
            mu, sigma, pi, r, d, _ = self._transformers_de(self._astate, self._lstate.unsqueeze(1))
            pi = pi.squeeze()
            mixt = Categorical(torch.exp(pi[-1])).sample().item()

            self._lstate = torch.cat([self._lstate, mu[-1, :, mixt, :]], dim=0) # + sigma[:, mixt, :] * torch.randn_like(mu[:, mixt, :])
            if self._lstate.size(0) > self.prev_latents_buffer_size:
                self._lstate = self._lstate[-self.prev_latents_buffer_size:]
            self._obs = self._decoder(mu[-1, :, mixt, :])
            np_obs = self._obs.numpy()
            np_obs = np.clip(np_obs, 0, 1) * 255
            np_obs = np.transpose(np_obs, (0, 2, 3, 1))
            np_obs = np_obs.squeeze()
            np_obs = np_obs.astype(np.uint8)
            self._visual_obs = np_obs

            return np_obs, r[-1].item(), d[-1].item() > 0

    def render(self): # pylint: disable=arguments-differ
        """ Rendering """
        import matplotlib.pyplot as plt
        if not self.monitor:
            self.figure = plt.figure()
            self.monitor = plt.imshow(
                np.zeros((RED_SIZE, RED_SIZE, 3),
                         dtype=np.uint8))
        self.monitor.set_data(self._visual_obs)
        plt.pause(.01)



if __name__ == '__main__':
    # argument parsing
    parser = argparse.ArgumentParser()
    parser.add_argument('--logdir', type=str, help='Directory from which MDRNN and VAE are '
                        'retrieved.')
    args = parser.parse_args()
    env = SimulatedCarracing(args.logdir)

    env.reset()
    action = np.array([0., 0., 0.])

    def on_key_press(event):
        """ Defines key pressed behavior """
        if event.key == 'up':
            action[1] = 1
        if event.key == 'down':
            action[2] = .8
        if event.key == 'left':
            action[0] = -1
        if event.key == 'right':
            action[0] = 1

    def on_key_release(event):
        """ Defines key pressed behavior """
        if event.key == 'up':
            action[1] = 0
        if event.key == 'down':
            action[2] = 0
        if event.key == 'left' and action[0] == -1:
            action[0] = 0
        if event.key == 'right' and action[0] == 1:
            action[0] = 0

    env.figure.canvas.mpl_connect('key_press_event', on_key_press)
    env.figure.canvas.mpl_connect('key_release_event', on_key_release)
    while True:
        _, _, done = env.step(action)
        env.render()
        if done:
            break
