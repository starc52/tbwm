"""
Define MDRNN model, supposed to be used as a world model
on the latent space.
"""
import math
import torch
import torch.nn as nn
import torch.nn.functional as f
from torch.distributions.normal import Normal


def gmm_loss(batch, mus, sigmas, logpi, reduce=True):  # pylint: disable=too-many-arguments
    """ Computes the gmm loss.

    Compute minus the log probability of batch under the GMM model described
    by mus, sigmas, pi. Precisely, with bs1, bs2, ... the sizes of the batch
    dimensions (several batch dimension are useful when you have both a batch
    axis and a time step axis), gs the number of mixtures and fs the number of
    features.

    :args batch: (bs1, bs2, *, fs) torch tensor
    :args mus: (bs1, bs2, *, gs, fs) torch tensor
    :args sigmas: (bs1, bs2, *, gs, fs) torch tensor
    :args logpi: (bs1, bs2, *, gs) torch tensor
    :args reduce: if not reduce, the mean in the following formula is ommited

    :returns:
    loss(batch) = - mean_{i1=0..bs1, i2=0..bs2, ...} log(
        sum_{k=1..gs} pi[i1, i2, ..., k] * N(
            batch[i1, i2, ..., :] | mus[i1, i2, ..., k, :], sigmas[i1, i2, ..., k, :]))

    NOTE: The loss is not reduced along the feature dimension (i.e. it should scale ~linearily
    with fs).
    """
    batch = batch.unsqueeze(-2)
    normal_dist = Normal(mus, sigmas)
    g_log_probs = normal_dist.log_prob(batch)
    g_log_probs = logpi + torch.sum(g_log_probs, dim=-1)
    max_log_probs = torch.max(g_log_probs, dim=-1, keepdim=True)[0]
    g_log_probs = g_log_probs - max_log_probs

    g_probs = torch.exp(g_log_probs)
    probs = torch.sum(g_probs, dim=-1)

    log_prob = max_log_probs.squeeze() + torch.log(probs)
    if reduce:
        return - torch.mean(log_prob)
    return - log_prob


class VanillaPositionalEncoding(nn.Module):
    r"""Inject some information about the relative or absolute position of the tokens in the sequence.
        The positional encodings have the same dimension as the embeddings, so that the two can be summed.
        Here, we use sine and cosine functions of different frequencies.
    .. math:
        \text{PosEncoder}(pos, 2i) = sin(pos/10000^(2i/d_model))
        \text{PosEncoder}(pos, 2i+1) = cos(pos/10000^(2i/d_model))
        \text{where pos is the word position and i is the embed idx)
    Args:
        d_model: the embed dim (required).
        dropout: the dropout value (default=0.1).
        max_len: the max. length of the incoming sequence (default=5000).
    Examples:
        >>> pos_encoder = VanillaPositionalEncoding(d_model)
    """

    def __init__(self, d_model, dropout=0.1, max_len=5000):
        super(VanillaPositionalEncoding, self).__init__()
        self.dropout = nn.Dropout(p=dropout)

        position = torch.arange(max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model))
        pe = torch.zeros(max_len, 1, d_model)
        pe[:, 0, 0::2] = torch.sin(position * div_term)
        if d_model % 2:
            pe[:, 0, 1::2] = torch.cos(position * div_term[:-1])
        else:
            pe[:, 0, 1::2] = torch.cos(position * div_term)
        self.register_buffer('pe', pe)

    def forward(self, x):
        r"""Inputs of forward function
        Args:
            x: the batch of sequence fed to the positional encoder model (required).
        Shape:
            x: [sequence length, batch size, embed dim]
            output: [sequence length, batch size, embed dim]
        Examples:
            >>> output = pos_encoder(x)
        """

        x = x + self.pe[:x.size(0)]
        return self.dropout(x)


class _TransformersDEBase(nn.Module):
    def __init__(self, latents, actions, hiddens, gaussians):
        super().__init__()
        self.latents = latents
        self.actions = actions
        self.hiddens = hiddens
        self.gaussians = gaussians

        self.gmm_linear = nn.Linear(hiddens, (2 * latents + 1) * gaussians + 2)
        # input size = latents+action because the transformer encoder gives the same output size as the input
        # output size = (2*LSIZE+1)*#gaussians + 2 why? each gaussian outputs mu,
        # sigma and a probability of the data coming from a particular gaussian. next, the 2 is added for rewards and
        # dones.

    def forward(self, *inputs):
        pass


class TransformersDE(_TransformersDEBase):
    """ Transformer model for processing the whole sequence at once (-multi steps forward-) """

    def __init__(self, latents, actions, hiddens, gaussians):
        super().__init__(latents, actions, hiddens, gaussians)
        self.nhead = 8
        nlayers = 3  # 6 change if performance is bad
        dropout = 0.1
        self.src_mask = None
        self.latent_action_linear = nn.Linear(latents+actions, latents)
        self.pos_encoder = VanillaPositionalEncoding(latents, dropout)
        encoder_layers = nn.TransformerEncoderLayer(latents, self.nhead, dropout=dropout)
        self.transformer_encoder = nn.TransformerEncoder(encoder_layers, nlayers)
        self.conversion_linear = nn.Linear(latents, hiddens)
        # self.rnn = nn.LSTM(latents + actions, hiddens)

    def _generate_square_subsequent_mask(self, sz):
        mask = (torch.triu(torch.ones(sz, sz, device=next(self.latent_action_linear.parameters()).device)) == 1).transpose(0, 1)
        mask = mask.float().masked_fill(mask == 0, float('-inf')).masked_fill(mask == 1, float(0.0))
        return mask.to(next(self.latent_action_linear.parameters()).device)

    def forward(self, actions, latents):  # pylint: disable=arguments-differ
        """ MULTI STEPS forward.

        :args actions: (SEQ_LEN, BSIZE, ASIZE) torch tensor
        :args latents: (SEQ_LEN, BSIZE, LSIZE) torch tensor

        :returns: mu_nlat, sig_nlat, pi_nlat, rs, ds, parameters of the GMM
        prediction for the next latent, gaussian prediction of the reward and
        logit prediction of terminality.
            - mu_nlat: (SEQ_LEN, BSIZE, N_GAUSS, LSIZE) torch tensor
            - sigma_nlat: (SEQ_LEN, BSIZE, N_GAUSS, LSIZE) torch tensor
            - logpi_nlat: (SEQ_LEN, BSIZE, N_GAUSS) torch tensor
            - rs: (SEQ_LEN, BSIZE) torch tensor
            - ds: (SEQ_LEN, BSIZE) torch tensor
        """
        seq_len, bs = actions.size(0), actions.size(1)

        ins = torch.cat([actions, latents], dim=-1)

        ins = f.relu(self.latent_action_linear(ins))

        ins = self.pos_encoder(ins)
        if self.src_mask is None or seq_len != len(self.src_mask):
            self.src_mask = self._generate_square_subsequent_mask(seq_len)
            # src_mask = src_mask.expand(bs*self.nhead, seq_len, seq_len) # not sure if this is right or not. Test
            # and see.
        outs = self.transformer_encoder(ins, self.src_mask)
        hiddens = self.conversion_linear(outs)

        gmm_outs = self.gmm_linear(hiddens)

        stride = self.gaussians * self.latents

        mus = gmm_outs[:, :, :stride]
        mus = mus.view(seq_len, bs, self.gaussians, self.latents)

        sigmas = gmm_outs[:, :, stride:2 * stride]
        sigmas = sigmas.view(seq_len, bs, self.gaussians, self.latents)
        sigmas = torch.exp(sigmas)

        pi = gmm_outs[:, :, 2 * stride: 2 * stride + self.gaussians]
        pi = pi.view(seq_len, bs, self.gaussians)
        logpi = f.log_softmax(pi, dim=-1)

        rs = gmm_outs[:, :, -2]

        ds = gmm_outs[:, :, -1]

        return mus, sigmas, logpi, rs, ds, hiddens  # hiddens is of size SEQ_LEN*BSIZE*HIDDEN_SIZE

class TransformersDEXL(_TransformersDEBase):
    """ Transformer model for processing the whole sequence at once (-multi steps forward-) """

    def __init__(self, latents, actions, hiddens, gaussians):
        super().__init__(latents, actions, hiddens, gaussians)
        self.nhead = 8
        nlayers = 6
        dropout = 0.1
        self.src_mask = None
        self.latent_action_linear = nn.Linear(latents+actions, latents)
        self.pos_encoder = VanillaPositionalEncoding(latents, dropout)
        encoder_layers = nn.TransformerEncoderLayer(latents, self.nhead, dropout=dropout)
        self.transformer_encoder = nn.TransformerEncoder(encoder_layers, nlayers)
        self.conversion_linear = nn.Linear(latents, hiddens)
        # self.rnn = nn.LSTM(latents + actions, hiddens)

    def _generate_square_subsequent_mask(self, sz):
        mask = (torch.triu(torch.ones(sz, sz, device=next(self.latent_action_linear.parameters()).device)) == 1).transpose(0, 1)
        mask = mask.float().masked_fill(mask == 0, float('-inf')).masked_fill(mask == 1, float(0.0))
        return mask.to(next(self.latent_action_linear.parameters()).device)

    def forward(self, actions, latents):  # pylint: disable=arguments-differ
        """ MULTI STEPS forward.

        :args actions: (SEQ_LEN, BSIZE, ASIZE) torch tensor
        :args latents: (SEQ_LEN, BSIZE, LSIZE) torch tensor

        :returns: mu_nlat, sig_nlat, pi_nlat, rs, ds, parameters of the GMM
        prediction for the next latent, gaussian prediction of the reward and
        logit prediction of terminality.
            - mu_nlat: (SEQ_LEN, BSIZE, N_GAUSS, LSIZE) torch tensor
            - sigma_nlat: (SEQ_LEN, BSIZE, N_GAUSS, LSIZE) torch tensor
            - logpi_nlat: (SEQ_LEN, BSIZE, N_GAUSS) torch tensor
            - rs: (SEQ_LEN, BSIZE) torch tensor
            - ds: (SEQ_LEN, BSIZE) torch tensor
        """
        seq_len, bs = actions.size(0), actions.size(1)

        ins = torch.cat([actions, latents], dim=-1)

        ins = f.relu(self.latent_action_linear(ins))

        ins = self.pos_encoder(ins)
        if self.src_mask is None or seq_len != len(self.src_mask):
            self.src_mask = self._generate_square_subsequent_mask(seq_len)
            # src_mask = src_mask.expand(bs*self.nhead, seq_len, seq_len) # not sure if this is right or not. Test
            # and see.
        outs = self.transformer_encoder(ins, self.src_mask)
        hiddens = self.conversion_linear(outs)

        gmm_outs = self.gmm_linear(hiddens)

        stride = self.gaussians * self.latents

        mus = gmm_outs[:, :, :stride]
        mus = mus.view(seq_len, bs, self.gaussians, self.latents)

        sigmas = gmm_outs[:, :, stride:2 * stride]
        sigmas = sigmas.view(seq_len, bs, self.gaussians, self.latents)
        sigmas = torch.exp(sigmas)

        pi = gmm_outs[:, :, 2 * stride: 2 * stride + self.gaussians]
        pi = pi.view(seq_len, bs, self.gaussians)
        logpi = f.log_softmax(pi, dim=-1)

        rs = gmm_outs[:, :, -2]

        ds = gmm_outs[:, :, -1]

        return mus, sigmas, logpi, rs, ds, hiddens  # hiddens is of size SEQ_LEN*BSIZE*HIDDEN_SIZE

# class TransformerDECell(_TransformersDEBase):
#     """ MDRNN model for one step forward """
#
#     def __init__(self, latents, actions, hiddens, gaussians):
#         super().__init__(latents, actions, hiddens, gaussians)
#         self.nhead = 8
#         nlayers = 3  # 6 change if performance is bad
#         self.src_mask = None
#
#         self.pos_encoder = PositionalEncoding(latents + actions, dropout)
#         encoder_layers = TransformerEncoderLayer(latents + actions, nhead, dropout=dropout)
#         self.transformer_encoder = TransformerEncoder(encoder_layers, nlayers)
#         # self.rnn = nn.LSTMCell(latents + actions, hiddens)
#
#     def forward(self, action, latent):  # pylint: disable=arguments-differ
#         """ ONE STEP forward.
#
#         :args actions: (BSIZE, ASIZE) torch tensor
#         :args latents: (BSIZE, LSIZE) torch tensor
#
#         :returns: mu_nlat, sig_nlat, pi_nlat, r, d, next_hidden, parameters of
#         the GMM prediction for the next latent, gaussian prediction of the
#         reward, logit prediction of terminality and next hidden state.
#             - mu_nlat: (BSIZE, N_GAUSS, LSIZE) torch tensor
#             - sigma_nlat: (BSIZE, N_GAUSS, LSIZE) torch tensor
#             - logpi_nlat: (BSIZE, N_GAUSS) torch tensor
#             - rs: (BSIZE) torch tensor
#             - ds: (BSIZE) torch tensor
#         """
#         action = action.unsqueeze(0)
#         latent = latent.unsqueeze(0)
#         in_al = torch.cat([action, latent], dim=-1)
#
#         # next_hidden = self.rnn(in_al, hidden)
#         next_hidden = self.transformer_encoder(in_al)
#
#         out_rnn = next_hidden[0]
#
#         out_full = self.gmm_linear(out_rnn)
#
#         stride = self.gaussians * self.latents
#
#         mus = out_full[:, :stride]
#         mus = mus.view(-1, self.gaussians, self.latents)
#
#         sigmas = out_full[:, stride:2 * stride]
#         sigmas = sigmas.view(-1, self.gaussians, self.latents)
#         sigmas = torch.exp(sigmas)
#
#         pi = out_full[:, 2 * stride:2 * stride + self.gaussians]
#         pi = pi.view(-1, self.gaussians)
#         logpi = f.log_softmax(pi, dim=-1)
#
#         r = out_full[:, -2]
#
#         d = out_full[:, -1]
#
#         return mus, sigmas, logpi, r, d, next_hidden


# class TransformerModel(nn.Module):
#     """Container module with an encoder, a recurrent or transformer module, and a decoder.
#         !!!!! BE CAREFUL. THIS MODEL DOESN'T TAKE B*Seq_len*dim. Only Seq_len*dim.
#     """
#
#     def __init__(self, ntoken, ninp, nhead, nhid, nlayers, dropout=0.5):
#         """
#         ntoken: number of tokens in dictionary
#         ninp: input dimensionality
#         nhead: transformer hyperparameter number of attention heads
#         nhid: dimension of hidden/feed-forward layer
#         dropout: somewhere in the feed-forward layer
#         """
#         super(TransformerModel, self).__init__()
#         try:
#             from torch.nn import TransformerEncoder, TransformerEncoderLayer
#         except BaseException as e:
#             raise ImportError('TransformerEncoder module does not exist in PyTorch 1.1 or '
#                               'lower.') from e
#         self.model_type = 'Transformer'
#         self.src_mask = None
#         self.pos_encoder = PositionalEncoding(ninp, dropout)
#         encoder_layers = TransformerEncoderLayer(ninp, nhead, nhid, dropout)
#         self.transformer_encoder = TransformerEncoder(encoder_layers, nlayers)
#         self.encoder = nn.Embedding(ntoken, ninp)
#         self.ninp = ninp
#         self.decoder = nn.Linear(ninp, ntoken)
#
#         self.init_weights()
#
#     def _generate_square_subsequent_mask(self, sz):
#         mask = (torch.triu(torch.ones(sz, sz)) == 1).transpose(0, 1)
#         mask = mask.float().masked_fill(mask == 0, float('-inf')).masked_fill(mask == 1, float(0.0))
#         return mask
#
#     def init_weights(self):
#         initrange = 0.1
#         nn.init.uniform_(self.encoder.weight, -initrange, initrange)
#         nn.init.zeros_(self.decoder.bias)
#         nn.init.uniform_(self.decoder.weight, -initrange, initrange)
#
#     def forward(self, src, has_mask=True):
#         if has_mask:
#             device = src.device
#             if self.src_mask is None or self.src_mask.size(0) != len(src):
#                 mask = self._generate_square_subsequent_mask(len(src)).to(device)
#                 self.src_mask = mask
#         else:
#             self.src_mask = None
#
#         src = self.encoder(src) * math.sqrt(self.ninp)
#         src = self.pos_encoder(src)
#         output = self.transformer_encoder(src, self.src_mask)
#         output = self.decoder(output)
#         return F.log_softmax(output, dim=-1)
