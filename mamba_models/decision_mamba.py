# coding=utf-8
# Copyright 2018 The OpenAI Team Authors and HuggingFace Inc. team.
# Copyright (c) 2018, NVIDIA CORPORATION.  All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
"""PyTorch OpenAI GPT-2 model."""

import torch
import torch.nn as nn
import transformers
from transformers.modeling_utils import PreTrainedModel  #, Conv1D
from transformers.models.gpt2.configuration_gpt2 import GPT2Config
import math
import numpy as np
import torch.nn.functional as F
from torch import distributions as pyd

from mamba_models.layers import Block


class TanhTransform(pyd.transforms.Transform):
    domain = pyd.constraints.real
    codomain = pyd.constraints.interval(-1.0, 1.0)
    bijective = True
    sign = +1

    def __init__(self, cache_size=1):
        super().__init__(cache_size=cache_size)

    @staticmethod
    def atanh(x):
        return 0.5 * (x.log1p() - (-x).log1p())

    def __eq__(self, other):
        return isinstance(other, TanhTransform)

    def _call(self, x):
        return x.tanh()

    def _inverse(self, y):
        # We do not clamp to the boundary here as it may degrade the performance of certain algorithms.
        # one should use `cache_size=1` instead
        return self.atanh(y)

    def log_abs_det_jacobian(self, x, y):
        # We use a formula that is more numerically stable, see details in the following link
        # https://github.com/tensorflow/probability/commit/ef6bb176e0ebd1cf6e25c6b5cecdd2428c22963f#diff-e120f70e92e6741bca649f04fcd907b7
        return 2.0 * (math.log(2.0) - x - F.softplus(-2.0 * x))


class SquashedNormal(pyd.transformed_distribution.TransformedDistribution):
    """
    Squashed Normal Distribution(s)

    If loc/std is of size (batch_size, sequence length, d),
    this returns batch_size * sequence length * d
    independent squashed univariate normal distributions.
    """

    def __init__(self, loc, std):
        self.loc = loc
        self.std = std
        self.base_dist = pyd.Normal(loc, std)

        transforms = [TanhTransform()]
        super().__init__(self.base_dist, transforms)

    @property
    def mean(self):
        mu = self.loc
        for tr in self.transforms:
            mu = tr(mu)
        return mu

    def entropy(self, N=1):
        # sample from the distribution and then compute
        # the empirical entropy:
        x = self.rsample((N,))
        # clamp to avoid hitting exact boundary values ±1 that break _validate_sample on MPS
        x = x.clamp(-0.999999, 0.999999)
        log_p = self.log_prob(x)

        # log_p: (batch_size, context_len, action_dim),
        return -log_p.mean(axis=0).sum(axis=2)

    def log_likelihood(self, x):
        # log_prob(x): (batch_size, context_len, action_dim)
        # sum up along the action dimensions
        # Return tensor shape: (batch_size, context_len)
        return self.log_prob(x).sum(axis=2)


class DiagGaussianActor(nn.Module):
    """torch.distributions implementation of an diagonal Gaussian policy."""

    def __init__(self, hidden_dim, act_dim, log_std_bounds=[-5.0, 2.0]):
        super().__init__()

        self.mu = torch.nn.Linear(hidden_dim, act_dim)
        self.log_std = torch.nn.Linear(hidden_dim, act_dim)
        self.log_std_bounds = log_std_bounds

        def weight_init(m):
            """Custom weight init for Conv2D and Linear layers."""
            if isinstance(m, torch.nn.Linear):
                nn.init.orthogonal_(m.weight.data)
                if hasattr(m.bias, "data"):
                    m.bias.data.fill_(0.0)

        self.apply(weight_init)

    def forward(self, obs):
        mu, log_std = self.mu(obs), self.log_std(obs)
        log_std = torch.tanh(log_std)
        # log_std is the output of tanh so it will be between [-1, 1]
        # map it to be between [log_std_min, log_std_max]
        log_std_min, log_std_max = self.log_std_bounds
        log_std = log_std_min + 0.5 * (log_std_max - log_std_min) * (log_std + 1.0)
        std = log_std.exp()
        return SquashedNormal(mu, std)


class GPT2PreTrainedModel(PreTrainedModel):
    """
    An abstract class to handle weights initialization and a simple interface
    for downloading and loading pretrained models.
    """

    config_class = GPT2Config
    base_model_prefix = "transformer"

    def __init__(self, *inputs, **kwargs):
        super().__init__(*inputs, **kwargs)

    def _init_weights(self, module):
        """Initialize the weights."""
        if isinstance(module, (nn.Linear, nn.Embedding)):
            # Slightly different from the TF version which uses truncated_normal for initialization
            # cf https://github.com/pytorch/pytorch/pull/5617
            module.weight.data.normal_(mean=0.0, std=self.config.initializer_range)
            if isinstance(module, (nn.Linear)) and module.bias is not None:
                module.bias.data.zero_()
        elif isinstance(module, nn.LayerNorm):
            module.bias.data.zero_()
            module.weight.data.fill_(1.0)
            # module.weight.data.fill_(.01)  # KL: Adapter change


class GPT2Model(GPT2PreTrainedModel):
    def __init__(self, config):
        super().__init__(config)

        self.drop = nn.Dropout(config.embd_pdrop)
        self.h = nn.ModuleList([Block(config, index) for index in range(config.n_layer)])  #, scale=True
        self.ln_f = nn.LayerNorm(config.n_embd, eps=config.layer_norm_epsilon)

        self.init_weights()

    def forward(self, inputs_embeds=None):
        hidden_states = inputs_embeds
        hidden_states = self.drop(hidden_states)

        for block in self.h:
            hidden_states = block(hidden_states)

        hidden_states = self.ln_f(hidden_states)
        return hidden_states


class DecisionMamba(nn.Module):
    """
    This model uses GPT to model (Return_1, state_1, action_1, Return_2, state_2, ...)
    """
    def __init__(
            self,
            state_dim,
            act_dim,
            hidden_size,
            max_length=None,
            max_ep_len=4096,
            action_tanh=True,
            remove_act_embs=False,
            stochastic_policy=True,
            init_temperature=0.1,
            target_entropy=None,
            **kwargs
    ):
        super().__init__()

        self.state_dim = state_dim
        self.act_dim = act_dim
        self.hidden_size = hidden_size
        self.max_length = max_length
        self.stochastic_policy = stochastic_policy
        # Store action range for evaluation clamping (like Decision Transformer)
        self.action_range = kwargs.get('action_range', [-1.0, 1.0])

        config = transformers.GPT2Config(
            vocab_size=1,  # doesn't matter -- we don't use the vocab
            n_embd=hidden_size,
            remove_act_embs=remove_act_embs,
            max_length=max_length,
            model_type='dmamba',  # Set to use Mamba blocks
            layer_norm_epsilon=1e-5,
            drop_p=kwargs.get('resid_pdrop', 0.1),
            **kwargs
        )
        #self.env_name = config.env_name

        # note: the only difference between this GPT2Model and the default Huggingface version
        # is that the positional embeddings are removed (since we'll add those ourselves)
        self.transformer = GPT2Model(config)
        self.remove_act_embs = remove_act_embs

        self.embed_timestep = nn.Embedding(max_ep_len, hidden_size)
        self.embed_return = torch.nn.Linear(1, hidden_size)
        self.embed_state = torch.nn.Linear(self.state_dim, hidden_size)
        self.embed_action = torch.nn.Linear(self.act_dim, hidden_size)

        self.embed_ln = nn.LayerNorm(hidden_size)

        # Add state and return prediction heads like Decision Transformer
        self.predict_state = torch.nn.Linear(hidden_size, self.state_dim)
        self.predict_return = torch.nn.Linear(hidden_size, 1)
        
        if stochastic_policy:
            self.predict_action = DiagGaussianActor(hidden_size, self.act_dim)
        else:
            self.predict_action = nn.Sequential(
                *([nn.Linear(hidden_size, self.act_dim)] + ([nn.Tanh()] if action_tanh else []))
            )

        if stochastic_policy:
            # Register log_temperature as a trainable parameter so it moves with .to(device)
            self.register_parameter(
                'log_temperature', nn.Parameter(torch.tensor(np.log(init_temperature), dtype=torch.float32)))
            self.target_entropy = target_entropy

    def temperature(self):
        if self.stochastic_policy:
            return self.log_temperature.exp()
        else:
            return torch.tensor(1.0, device=next(self.parameters()).device)

    def forward(self, states, actions, rewards, returns_to_go, timesteps, ordering, padding_mask=None):
        batch_size, seq_length = states.shape[0], states.shape[1]

        # time embeddings are treated similar to positional embeddings
        time_embeddings = self.embed_timestep(timesteps)
        state_embeddings = self.embed_state(states) + time_embeddings
        returns_embeddings = self.embed_return(returns_to_go) + time_embeddings
        if not self.remove_act_embs:
            action_embeddings = self.embed_action(actions) + time_embeddings

        # this makes the sequence look like (R_1, s_1, a_1, R_2, s_2, a_2, ...)
        # which works nice in an autoregressive sense since states predict actions
        if self.remove_act_embs:
            num_token_type = 2
            stacked_inputs = torch.stack(
                (returns_embeddings, state_embeddings), dim=1
            ).permute(0, 2, 1, 3).reshape(batch_size, num_token_type*seq_length, self.hidden_size)
            stacked_inputs = self.embed_ln(stacked_inputs)
        else:
            num_token_type = 3
            stacked_inputs = torch.stack(
                (returns_embeddings, state_embeddings, action_embeddings), dim=1
            ).permute(0, 2, 1, 3).reshape(batch_size, num_token_type*seq_length, self.hidden_size)
            stacked_inputs = self.embed_ln(stacked_inputs)

        # we feed in the input embeddings (not word indices as in NLP) to the model
        x = self.transformer(inputs_embeds=stacked_inputs)

        # reshape x so that the second dimension corresponds to the original
        # returns (0), states (1), or actions (2); i.e. x[:,1,t] is the token for s_t
        x = x.reshape(batch_size, seq_length, num_token_type, self.hidden_size).permute(0, 2, 1, 3)

        # Get predictions for all modalities like Decision Transformer
        # predict next action given state
        action_preds = self.predict_action(x[:, 1])  
        # predict next state given state and action
        state_preds = self.predict_state(x[:, 2] if x.shape[1] > 2 else x[:, 1])
        # predict next return given state and action  
        return_preds = self.predict_return(x[:, 2] if x.shape[1] > 2 else x[:, 1])
        
        return state_preds, action_preds, return_preds

    def get_action(self, states, actions, returns_to_go, timesteps):
        # we don't care about the past rewards in this model
        states = states.reshape(1, -1, self.state_dim)
        actions = actions.reshape(1, -1, self.act_dim)
        returns_to_go = returns_to_go.reshape(1, -1, 1)
        timesteps = timesteps.reshape(1, -1)

        states = states[:,-self.max_length:]
        actions = actions[:,-self.max_length:]
        returns_to_go = returns_to_go[:,-self.max_length:]
        timesteps = timesteps[:,-self.max_length:]

        # pad all tokens to sequence length
        states = torch.cat(
            [torch.zeros((states.shape[0], self.max_length-states.shape[1], self.state_dim), device=states.device), states],
            dim=1).to(dtype=torch.float32)
        actions = torch.cat(
            [torch.zeros((actions.shape[0], self.max_length - actions.shape[1], self.act_dim), device=actions.device), actions],
            dim=1).to(dtype=torch.float32)
        returns_to_go = torch.cat(
            [torch.zeros((returns_to_go.shape[0], self.max_length-returns_to_go.shape[1], 1), device=returns_to_go.device), returns_to_go],
            dim=1).to(dtype=torch.float32)
        timesteps = torch.cat(
            [torch.zeros((timesteps.shape[0], self.max_length-timesteps.shape[1]), device=timesteps.device), timesteps],
            dim=1).to(dtype=torch.long)

        state_preds, action_preds, return_preds = self.forward(states, actions, None, returns_to_go, timesteps, None)
        if self.stochastic_policy:
            return action_preds[0, -1]  # action_preds is already a distribution
        else:
            return action_preds[0, -1]  # action_preds is a tensor

    def get_predictions(self, states, actions, rewards, returns_to_go, timesteps, num_envs=1, **kwargs):
        """Get predictions for evaluation - similar to Decision Transformer interface"""
        # Reshape inputs to match expected format
        batch_size = states.shape[0]
        
        # Process sequences to get predictions  
        state_preds, action_preds, return_preds = self.forward(
            states, actions, rewards, returns_to_go, timesteps, None
        )
        
        if self.stochastic_policy:
            return state_preds[:, -1], action_preds, return_preds[:, -1]  # (state_preds, action_dist, return_preds)
        else:
            return state_preds[:, -1], action_preds[:, -1], return_preds[:, -1]  # (state_preds, action_preds, return_preds)
