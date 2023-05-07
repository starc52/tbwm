""" Define controller """
import torch
import torch.nn as nn

class Controller(nn.Module):
    """ Controller
        :attr latents: latent space representation of current view, from VAE. 1xLSIZE
        :attr recurrents: current understanding of the world from the transformer 1xRSIZE
        :returns: (action)
            - action: (1 x ASIZE) torch tensor
    """
    def __init__(self, latents, recurrents, actions):
        super().__init__()
        self.fc = nn.Linear(latents + recurrents, actions)

    def forward(self, *inputs):
        cat_in = torch.cat(inputs, dim=1)
        return self.fc(cat_in)
