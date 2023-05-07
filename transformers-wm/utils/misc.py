""" Various auxiliary utilities """
import math
from os.path import join, exists
import torch
from torchvision import transforms
import numpy as np
from models import MDRNNCell, VAE, Controller, TransformersDE
import gym
import gym.envs.box2d
import cv2
# A bit dirty: manually change size of car racing env
gym.envs.box2d.car_racing.STATE_W, gym.envs.box2d.car_racing.STATE_H = 96, 96

# Hardcoded for now
ASIZE, LSIZE, RSIZE, RED_SIZE, SIZE =\
    3, 32, 256, 96, 96
LBUFFER_SIZE=500
# Same
transform = transforms.Compose([
    transforms.ToPILImage(),
    transforms.Resize((RED_SIZE, RED_SIZE)),
    transforms.ToTensor()
])

def sample_continuous_policy(action_space, seq_len, dt):
    """ Sample a continuous policy.

    Atm, action_space is supposed to be a box environment. The policy is
    sampled as a brownian motion a_{t+1} = a_t + sqrt(dt) N(0, 1).

    :args action_space: gym action space
    :args seq_len: number of actions returned
    :args dt: temporal discretization

    :returns: sequence of seq_len actions
    """
    actions = [action_space.sample()]
    for _ in range(seq_len):
        daction_dt = np.random.randn(*actions[-1].shape)
        actions.append(
            np.clip(actions[-1] + math.sqrt(dt) * daction_dt,
                    action_space.low, action_space.high))
    return actions

def save_checkpoint(state, is_best, filename, best_filename):
    """ Save state in filename. Also save in best_filename if is_best. """
    torch.save(state, filename)
    if is_best:
        torch.save(state, best_filename)

def flatten_parameters(params):
    """ Flattening parameters.

    :args params: generator of parameters (as returned by module.parameters())

    :returns: flattened parameters (i.e. one tensor of dimension 1 with all
        parameters concatenated)
    """
    return torch.cat([p.detach().view(-1) for p in params], dim=0).cpu().numpy()

def unflatten_parameters(params, example, device):
    """ Unflatten parameters.

    :args params: parameters as a single 1D np array
    :args example: generator of parameters (as returned by module.parameters()),
        used to reshape params
    :args device: where to store unflattened parameters

    :returns: unflattened parameters
    """
    params = torch.Tensor(params).to(device)
    idx = 0
    unflattened = []
    for e_p in example:
        unflattened += [params[idx:idx + e_p.numel()].view(e_p.size())]
        idx += e_p.numel()
    return unflattened

def load_parameters(params, controller):
    """ Load flattened parameters into controller.

    :args params: parameters as a single 1D np array
    :args controller: module in which params is loaded
    """
    proto = next(controller.parameters())
    params = unflatten_parameters(
        params, controller.parameters(), proto.device)

    for p, p_0 in zip(controller.parameters(), params):
        p.data.copy_(p_0)

class RolloutGenerator(object):
    """ Utility to generate rollouts.

    Encapsulate everything that is needed to generate rollouts in the TRUE ENV
    using a controller with previously trained VAE and TransformersDE.

    :attr vae: VAE model loaded from mdir/vae
    :attr transformers_de: TransformersDE model loaded from mdir/transformers_de
    :attr controller: Controller, either loaded from mdir/ctrl or randomly
        initialized
    :attr env: instance of the CarRacing-v0 gym environment
    :attr device: device used to run VAE, TransformersDE and Controller
    :attr time_limit: rollouts have a maximum of time_limit timesteps
    :attr prev_latents_buffer: tensor of previous observations' latents limited at a sequence length of a 1000.
    """
    def __init__(self, mdir, device, time_limit, test_save_directory=None, latents_buffer_size=LBUFFER_SIZE):
        """ Build vae, rnn, controller and environment. """
        # Loading world model and vae
        vae_file, transformers_de_file, ctrl_file = \
            [join(mdir, m, 'best.tar') for m in ['vae', 'transformers_de', 'ctrl']]

        assert exists(vae_file) and exists(transformers_de_file),\
            "Either vae or transformers_de is untrained."

        vae_state, transformers_de_state = [
            torch.load(fname, map_location={'cuda:0': str(device)})
            for fname in (vae_file, transformers_de_file)]

        for m, s in (('VAE', vae_state), ('TransformersDE', transformers_de_state)):
            print("Loading {} at epoch {} "
                  "with test loss {}".format(
                      m, s['epoch'], s['precision']))

        self.vae = VAE(3, LSIZE).to(device)
        self.vae.load_state_dict(vae_state['state_dict'])

        self.transformers_de = TransformersDE(LSIZE, ASIZE, RSIZE, 5).to(device)
        self.transformers_de.load_state_dict(transformers_de_state['state_dict'])

        self.controller = Controller(LSIZE, RSIZE, ASIZE).to(device)

        # load controller if it was previously saved
        if exists(ctrl_file):
            ctrl_state = torch.load(ctrl_file, map_location={'cuda:0': str(device)})
            print("Loading Controller with reward {}".format(
                ctrl_state['reward']))
            self.controller.load_state_dict(ctrl_state['state_dict'])

        self.env = gym.make('CarRacing-v0')
        self.device = device

        self.time_limit = time_limit
        self.prev_latents_buffer = None
        self.prev_actions_buffer = None

        self.test_save_directory = test_save_directory
        self.latents_buffer_size = latents_buffer_size

    def get_action_and_transition(self, obs, hidden):
        """ Get action and transition.

        Encode obs to latent using the VAE, then obtain estimation for next
        latent and next hidden state using the TransformersDE and compute the controller
        corresponding action.

        :args obs: current observation (1 x 3 x 64 x 64) torch tensor
        :args hidden: current hidden state (1 x 256) torch tensor

        :returns: (action, next_hidden)
            - action: 1D np array
            - next_hidden (1 x 256) torch tensor
        """
        _, latent_mu, _ = self.vae(obs) # the view of the world right now, (1x32)
        action = self.controller(latent_mu, hidden) # hidden is the previous understanding of the world
        if self.prev_latents_buffer is None:
            self.prev_latents_buffer = latent_mu
            self.prev_actions_buffer = action
        else:
            self.prev_latents_buffer = torch.cat([self.prev_latents_buffer, latent_mu], dim=0)
            self.prev_actions_buffer = torch.cat([self.prev_actions_buffer, action], dim=0)

        if self.prev_latents_buffer.size(0) > self.latents_buffer_size:
            self.prev_latents_buffer = self.prev_latents_buffer[-self.latents_buffer_size:]
            self.prev_actions_buffer = self.prev_actions_buffer[-self.latents_buffer_size:]

        _, _, _, _, _, next_hidden = self.transformers_de(self.prev_actions_buffer.unsqueeze(1), self.prev_latents_buffer.unsqueeze(1)) # update the understanding given the view
        return action.squeeze().cpu().numpy(), next_hidden[-1, :, :].squeeze(1)

    def rollout(self, params, render=False):
        """ Execute a rollout and returns minus cumulative reward.

        Load :params: into the controller and execute a single rollout. This
        is the main API of this class.

        :args params: parameters as a single 1D np array

        :returns: minus cumulative reward
        """
        # copy params into the controller
        if params is not None:
            load_parameters(params, self.controller)

        obs = self.env.reset()

        # This first render is required !
        self.env.render()
        # initialize everything
        hidden = torch.zeros(1, RSIZE).to(self.device)
        self.prev_latents_buffer = None
        self.prev_actions_buffer = None
        
        cumulative = 0
        i = 0
        while True:
            if self.test_save_directory is not None:
                cv2.imwrite(join(self.test_save_directory, "{:04d}.png".format(i)), obs)
            obs = transform(obs).unsqueeze(0).to(self.device)
            action, hidden = self.get_action_and_transition(obs, hidden)
            obs, reward, done, _ = self.env.step(action)

            if render:
                self.env.render()

            cumulative += reward
            if done or i > self.time_limit:
                return - cumulative
            i += 1

