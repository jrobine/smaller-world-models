import math
from typing import Any, Dict, Optional, Tuple

import torch
from torch import nn, Tensor
from torch.distributions import Distribution, Categorical, Independent

from layer_utils.distributions import independent_continuous_bernoullis
from layer_utils.modes import mode
from layers import ConvLSTMCell, ResidualBlock31, VQVAE
from rl.modelbased import *
from rl.networks import *
from rl.spaces import *
from rl.utils import *

__all__ = ['AtariStateRepresentation', 'AtariDynamics', 'AtariDiscreteLatentSpaceWorldModel',
           'AtariLatentSpaceActorCriticNetwork']


class AtariStateRepresentation(VQVAE):

    def __init__(self, in_channels: int, num_embeddings: int, embedding_size: int) -> None:
        super().__init__(num_embeddings, embedding_size, latent_height=6, latent_width=6,
                         exponential_moving_averages=True)

        self.encoder = nn.Sequential(
            nn.Conv2d(in_channels, 32, kernel_size=4, stride=2, padding=1, bias=False),
            nn.BatchNorm2d(32),
            nn.LeakyReLU(),
            nn.Conv2d(32, 64, kernel_size=4, stride=2, padding=1, bias=False),
            nn.BatchNorm2d(64),
            nn.LeakyReLU(),
            nn.Conv2d(64, 128, kernel_size=4, stride=2, padding=1, bias=False),
            nn.BatchNorm2d(128),
            nn.LeakyReLU(),
            nn.Conv2d(128, 256, kernel_size=4, stride=2, padding=1, bias=False),
            nn.BatchNorm2d(256),
            nn.LeakyReLU(),
            ResidualBlock31(256, 256, batch_norm=True),
            nn.LeakyReLU(),
            ResidualBlock31(256, 256, batch_norm=True),
            nn.LeakyReLU(),
            # TODO kernel_size 3 or 1?
            nn.Conv2d(256, embedding_size, kernel_size=1, stride=1, padding=0, bias=False),
            nn.BatchNorm2d(embedding_size)
        )

        self.decoder = nn.Sequential(
            # TODO kernel_size 3 or 1?
            nn.Conv2d(embedding_size, 256, kernel_size=3, stride=1, padding=1, bias=True),
            nn.LeakyReLU(),
            ResidualBlock31(256, 256, batch_norm=False),
            nn.LeakyReLU(),
            ResidualBlock31(256, 256, batch_norm=False),
            nn.LeakyReLU(),
            nn.ConvTranspose2d(256, 128, kernel_size=4, stride=2, padding=1, bias=True),
            nn.LeakyReLU(),
            nn.ConvTranspose2d(128, 64, kernel_size=4, stride=2, padding=1, bias=True),
            nn.LeakyReLU(),
            nn.ConvTranspose2d(64, 32, kernel_size=4, stride=2, padding=1, bias=True),
            nn.LeakyReLU(),
            nn.ConvTranspose2d(32, in_channels, kernel_size=4, stride=2, padding=1, bias=True)
        )

    def encode(self, x: Tensor) -> Tensor:
        return self.encoder(x).permute(0, 2, 3, 1)

    def decode(self, z: Tensor) -> Distribution:
        logits = self.decoder(z.permute(0, 3, 1, 2))
        return independent_continuous_bernoullis(logits)


class AtariDynamics(nn.Module):

    def __init__(
            self,
            embedding_size: int,
            num_embeddings: int,
            num_actions: int,
            conv_lstm_hidden_channels: Tuple[int, int] = (128, 128)) -> None:
        super().__init__()

        cell1_channels, cell2_channels = conv_lstm_hidden_channels
        self.cell1 = ConvLSTMCell(
            embedding_size + num_actions, hidden_channels=cell1_channels,
            height=6, width=6, kernel_size=5, bias=True, layer_norm=True)
        self.cell2 = ConvLSTMCell(
            cell1_channels + num_actions, hidden_channels=cell2_channels,
            height=6, width=6, kernel_size=5, bias=True, layer_norm=True)

        self.next_latent_head = nn.Conv2d(
            cell2_channels + num_actions, num_embeddings,
            kernel_size=1, stride=1, padding=0, bias=True)

        self.reward_head = nn.Sequential(
            nn.Conv2d(cell2_channels + num_actions, 32, kernel_size=1, stride=1, padding=0, bias=True),
            nn.LayerNorm([32, 6, 6]),
            nn.LeakyReLU(),
            nn.Flatten(),
            nn.Linear(32 * 6 * 6, 512),
            nn.LeakyReLU(),
            nn.Linear(512, 3)
        )

        # nn.functional.one_hot seems to be slow (at least PyTorch 1.6.0), so we
        # cache the results and index the variable
        self.register_buffer(
            '_one_hot_cache', nn.functional.one_hot(
                torch.arange(num_actions), num_actions).float().view(num_actions, num_actions, 1, 1)
                .expand(num_actions, num_actions, 6, 6), persistent=False)

    def init_recurrent_state(self, batch_size: int) -> Tensor:
        # TODO keep contiguous (see agent model recurrent states)
        h1, c1 = self.cell1.init_recurrent_state(batch_size)
        h2, c2 = self.cell2.init_recurrent_state(batch_size)
        recurrent_state = torch.cat((h1, c1, h2, c2), 1)
        return recurrent_state

    def reset_recurrent_state(self, recurrent_state: Tensor, terminal: Tensor) -> Tensor:
        mask = ~terminal.reshape(-1, 1, 1, 1)  # for broadcasting
        return recurrent_state * mask

    def forward(
            self,
            latent_quantized: Tensor,
            action: Tensor,
            recurrent_state: Tensor
    ) -> Tuple[Distribution, Distribution, Tensor]:
        latent_quantized = latent_quantized.permute(0, 3, 1, 2)

        action_enc = self._one_hot_cache[action]

        cell1_channels = self.cell1.hidden_channels
        cell2_channels = self.cell2.hidden_channels
        # TODO keep contiguous (see agent model recurrent states)
        h1, c1, h2, c2 = torch.split(
            recurrent_state, (cell1_channels, cell1_channels, cell2_channels, cell2_channels), 1)
        h1, c1 = self.cell1(torch.cat((latent_quantized, action_enc), 1), (h1, c1))
        h2, c2 = self.cell2(torch.cat((h1, action_enc), 1), (h2, c2))
        recurrent_state = torch.cat((h1, c1, h2, c2), 1)

        # TODO nonlinearity?
        y = torch.cat((h2, action_enc), 1)

        next_latent_logits = self.next_latent_head(y).permute(0, 2, 3, 1)
        next_latent_distribution = Independent(Categorical(logits=next_latent_logits), 2)

        reward_logits = self.reward_head(y)
        reward_distribution = Categorical(logits=reward_logits)

        return next_latent_distribution, reward_distribution, recurrent_state


class AtariDiscreteLatentSpaceWorldModel(EnvModel[Tensor]):

    def __init__(self,
                 in_channels: int,
                 num_embeddings: int,
                 num_actions: int,
                 device: torch.device,
                 real_trajectory_buffer: TrajectoryBuffer) -> None:
        super().__init__()
        self._real_observation_space = TensorBox(
            low=torch.zeros((in_channels, 96, 96), dtype=torch.uint8, device=device),
            high=torch.full((in_channels, 96, 96), 255, dtype=torch.uint8, device=device),
            device=device)
        self._real_action_space = TensorDiscrete(num_actions, device)

        self.real_trajectory_buffer = real_trajectory_buffer

        embedding_size = 32  # TODO 64?

        self._simulated_observation_space = TensorBox(
            low=torch.full((6, 6, embedding_size), -math.inf, dtype=torch.float, device=device),
            high=torch.full((6, 6, embedding_size), math.inf, dtype=torch.float, device=device))
        self._simulated_action_space = self._real_action_space

        self.state_repr = AtariStateRepresentation(in_channels, num_embeddings, embedding_size)
        self.dynamics = AtariDynamics(embedding_size, num_embeddings, num_actions, (128, 128))

        # only for evaluation
        self._latent_index_trace = torch.zeros(num_embeddings, device=device)

        self.to(device)

    @property
    def real_observation_space(self) -> TensorSpace:
        return self._real_observation_space

    @property
    def real_action_space(self) -> TensorSpace:
        return self._real_action_space

    @property
    def simulated_observation_space(self) -> TensorSpace:
        return self._simulated_observation_space

    @property
    def simulated_action_space(self) -> TensorSpace:
        return self._simulated_action_space

    def simulate_reset(self, batch_size: int) -> Tuple[Tensor, Tensor]:
        with torch.no_grad():
            # sample initial observation from real data
            assert self.real_trajectory_buffer.num_trajectories > 0
            observation = next(self.real_trajectory_buffer.generate_minibatches(
                Obs, minibatch_size=batch_size, minibatch_steps=1,
                max_random_offset=0, shuffle=True, drop_last_batch=True,
                drop_last_time=False)).observation.squeeze(0)
            observation_pre = self._prepare_observations(observation)
            _, latent_quantized = self.state_repr.encode_and_quantize(observation_pre)
            recurrent_state = self.dynamics.init_recurrent_state(batch_size)
            return latent_quantized, recurrent_state

    def simulate_step(
            self,
            last_latent_quantized: Tensor,
            last_reward: Tensor,
            last_terminal: Tensor,
            action: Tensor,
            recurrent_state: Tensor
    ) -> Tuple[Tensor, Tensor, Tensor, Tuple[Dict[str, Any], ...], Tensor]:
        with torch.no_grad():
            next_latent_distribution, reward_distribution, recurrent_state = \
                self.dynamics(last_latent_quantized, action, recurrent_state)

            pred_latent_indices = next_latent_distribution.sample()
            pred_latent_quantized = self.state_repr.lookup(pred_latent_indices)

            # map rewards from categorical support {0, 1, 2} to [-1, 1]
            reward = reward_distribution.sample().float() - 1.
            terminal = torch.zeros_like(reward, dtype=torch.bool)
            info = tuple({} for _ in range(action.shape[0]))
            return pred_latent_quantized, reward, terminal, info, recurrent_state

    def encode(
            self,
            observation: Optional[Tensor],
            reward: Optional[Tensor],
            terminal: Optional[Tensor],
            info: Optional[Tuple[Dict[str, Any], ...]],
    ) -> Tuple[Optional[Tensor], Optional[Tensor], Optional[Tensor], Optional[Tuple[Dict[str, Any], ...]]]:
        with torch.no_grad():
            observation_pre = self._prepare_observations(observation)
            _, latent_quantized = self.state_repr.encode_and_quantize(observation_pre)
        return latent_quantized, reward, terminal, info

    def _prepare_observations(self, observations: Tensor) -> Tensor:
        observations = observations.to(self._simulated_observation_space.device)
        observations = observations.float() / 255.
        return observations

    def compute_state_representation_loss(self, observation: Tensor) -> Tuple[Tensor, Dict[str, Tensor]]:
        observation_pre = self._prepare_observations(observation)
        z_e, z, z_q, x_posterior = self.state_repr(observation_pre)
        loss, stats = self.state_repr.compute_loss(observation_pre, z_e, z, z_q, x_posterior)

        self._latent_index_trace *= 0.99
        unique_z, counts = torch.unique(z, return_counts=True)
        self._latent_index_trace[unique_z] += counts
        latent_index_distribution = self._latent_index_trace / torch.sum(self._latent_index_trace)
        latent_index_distribution = torch.round(latent_index_distribution * 100.) / 100.
        latent_index_distribution = latent_index_distribution / torch.sum(latent_index_distribution)

        num_used_indices = torch.unique(torch.nonzero(latent_index_distribution, as_tuple=False)).shape[0]
        entropy = torch.distributions.Categorical(probs=latent_index_distribution).entropy()
        stats['avg_entropy'] = entropy
        stats['avg_embedding_usage'] = torch.tensor(num_used_indices / self.state_repr.num_embeddings)

        return loss, stats

    def compute_dynamics_loss(
            self,
            observations: Tensor,
            actions: Tensor,
            rewards: Tensor,
            terminals: Tensor,
            context_size: int,
            reward_loss_weight: float) -> Tuple[Tensor, Dict[str, Tensor]]:
        num_steps, batch_size = observations.shape[:2]
        recurrent_state = self.dynamics.init_recurrent_state(batch_size)

        assert context_size >= 1
        with torch.no_grad():
            observations_pre = self._prepare_observations(observations)
            latents_enc = self.state_repr.encode(observations_pre.view(-1, *observations_pre.shape[2:]))
            latents_enc = latents_enc.view(*observations_pre.shape[:2], *latents_enc.shape[1:])
            true_latents_indices, true_latents_quantized = self.state_repr.quantize(latents_enc)
            del observations, observations_pre, latents_enc

            # map to match support of categorical: {0, ..., num_actions - 1}
            if rewards.is_floating_point():
                rewards_sup = rewards.clamp(-1., 1.).round().long() + 1
            else:
                rewards_sup = rewards.long().clamp(-1, 1) + 1
            del rewards

        next_latent_loss_sum = 0.
        next_latent_loss_count = 0
        reward_loss_sum = 0.

        pred_latent_num_correct = torch.zeros(1, dtype=torch.long, device=true_latents_indices.device)
        reward_num_correct = torch.zeros(3, dtype=torch.long, device=rewards_sup.device)  # per class
        reward_num_total = [0, 0, 0]

        sampled_latents_quantized = None
        # TODO check time indices (e.g. num_steps - 1?)
        for t in range(num_steps - 1):
            if t < context_size:
                observation_in = true_latents_quantized[t]
            else:
                observation_in = sampled_latents_quantized

            next_latent_distribution, reward_distribution, recurrent_state = \
                self.dynamics(observation_in, actions[t], recurrent_state)
            del observation_in

            next_non_terminal = ~terminals[t + 1]  # ignore next terminal obs
            next_non_terminal_count = next_non_terminal.sum()
            if next_non_terminal_count > 0:
                # negative log likelihood
                nll = -next_latent_distribution.log_prob(true_latents_indices[t + 1])
                next_latent_loss = (nll * next_non_terminal).sum()
                next_latent_loss_sum = next_latent_loss_sum + next_latent_loss
                next_latent_loss_count += next_non_terminal_count
                del nll, next_latent_loss
            del next_non_terminal_count

            reward_sup = rewards_sup[t]
            reward_loss = -reward_distribution.log_prob(reward_sup).sum()
            reward_loss_sum = reward_loss_sum + reward_loss
            del reward_loss

            recurrent_state = self.dynamics.reset_recurrent_state(recurrent_state, terminals[t])

            with torch.no_grad():
                pred_latent_indices = next_latent_distribution.sample()
                sampled_latents_quantized = self.state_repr.lookup(pred_latent_indices)

                pred_latent_num_correct += torch.sum(
                    pred_latent_indices[next_non_terminal] == true_latents_indices[t + 1][next_non_terminal])
                del pred_latent_indices, next_latent_distribution, next_non_terminal

                reward_mode = mode(reward_distribution)
                r0 = reward_mode[reward_sup == 0]
                r1 = reward_mode[reward_sup == 1]
                r2 = reward_mode[reward_sup == 2]
                reward_num_correct[0] += torch.sum(r0 == 0)
                reward_num_correct[1] += torch.sum(r1 == 1)
                reward_num_correct[2] += torch.sum(r2 == 2)
                reward_num_total[0] += r0.numel()
                reward_num_total[1] += r1.numel()
                reward_num_total[2] += r2.numel()
                del reward_mode, reward_distribution, reward_sup, r0, r1, r2

        # TODO / batch_size?
        total_next_latent_loss = next_latent_loss_sum / next_latent_loss_count
        total_reward_loss = reward_loss_sum / ((num_steps - 1) * batch_size)
        loss = total_next_latent_loss + reward_loss_weight * total_reward_loss
        stats = {
            'next_latent_loss': total_next_latent_loss.detach().clone(),
            'next_latent_acc':
                pred_latent_num_correct.float() /
                (next_latent_loss_count * self.state_repr.latent_height * self.state_repr.latent_width),
            'reward_loss': total_reward_loss.detach().clone()
        }
        # TODO total reward accuracy
        if reward_num_total[0] > 0:
            stats['reward_-1_acc'] = reward_num_correct[0].float() / reward_num_total[0]
        if reward_num_total[1] > 0:
            stats['reward_0_acc'] = reward_num_correct[1].float() / reward_num_total[1]
        if reward_num_total[2] > 0:
            stats['reward_+1_acc'] = reward_num_correct[2].float() / reward_num_total[2]

        return loss, stats


class AtariLatentSpaceActorCriticNetwork(SharedActorCriticNetwork):

    def __init__(
            self,
            embedding_size: int,
            num_actions: int,
            recurrent: bool,
            device: torch.device) -> None:
        super().__init__()
        self._observation_space = TensorBox(
            low=torch.full((6, 6, embedding_size), -math.inf, dtype=torch.float, device=device),
            high=torch.full((6, 6, embedding_size), math.inf, dtype=torch.float, device=device),
            device=device)
        self._action_space = TensorDiscrete(num_actions, device)
        self.hidden = nn.Sequential(
            nn.Conv2d(embedding_size, 64, kernel_size=1, stride=1, padding=0, bias=False),
            nn.LayerNorm([64, 6, 6]),
            nn.LeakyReLU(),
            nn.Conv2d(64, 64, kernel_size=5, stride=1, padding=2, bias=False),
            nn.LayerNorm([64, 6, 6]),
            nn.LeakyReLU(),
            nn.Flatten(),
            nn.Linear(64 * 6 * 6, 512),
            nn.LeakyReLU()
        )
        if recurrent:
            self.lstm = nn.LSTM(512, 128)
            self.action_logits = nn.Linear(128, num_actions)
            self.value = nn.Linear(128, 1)
        else:
            self.action_logits = nn.Linear(512, num_actions)
            self.value = nn.Linear(512, 1)
        self.register_buffer('_temperature', torch.tensor(1.0))
        self.to(device)

    @property
    def temperature(self) -> float:
        return self._temperature.item()

    @temperature.setter
    def temperature(self, value: float) -> None:
        self._temperature.copy_(torch.tensor(value))

    @property
    def observation_space(self) -> TensorSpace:
        return self._observation_space

    @property
    def action_space(self) -> TensorSpace:
        return self._action_space

    @property
    def is_recurrent(self) -> bool:
        return hasattr(self, 'lstm')

    def init_recurrent_state(self, batch_size: int) -> Tensor:
        zeros = torch.zeros(1, batch_size, 128, device=next(self.parameters()).device)
        h, c = zeros, zeros
        recurrent_state = torch.cat((h, c), 0).transpose(0, 1)  # transpose to keep contiguous
        return recurrent_state

    def mask_recurrent_state(self, recurrent_state: Optional[Tensor], terminal: Tensor) -> Optional[Tensor]:
        assert recurrent_state.shape[0] == terminal.shape[0]
        # transpose to keep contiguous
        recurrent_state = (recurrent_state.transpose(0, 1) * (~terminal).reshape(1, -1, 1)).transpose(0, 1)
        return recurrent_state

    def forward(self, observation: Tensor, recurrent_state: Optional[Tensor] = None) -> Tuple[Tensor, Optional[Tensor]]:
        hidden = self.hidden(observation.permute(0, 3, 1, 2))
        if self.is_recurrent:
            assert recurrent_state is not None
            h, c = recurrent_state.transpose(0, 1)
            h, c = h.unsqueeze(0).contiguous(), c.unsqueeze(0).contiguous()
            hidden_seq = hidden.unsqueeze(0)
            hidden_seq, (h, c) = self.lstm(hidden_seq, (h, c))
            hidden = hidden_seq.squeeze(0)
            recurrent_state = torch.cat((h, c), 0).transpose(0, 1)  # transpose to keep contiguous
        return hidden, recurrent_state

    def compute_hidden(
            self,
            observation: Tensor,
            recurrent_state: Optional[Tensor] = None,
            train: bool = False) -> Tuple[Tensor, Optional[Tensor]]:
        return self.forward(observation, recurrent_state)

    def compute_action_distribution_shared(
            self,
            observation: Tensor,
            hidden: Tensor,
            recurrent_state: Optional[Tensor] = None,
            train: bool = False) -> Tuple[Distribution, Optional[Tensor]]:
        logits = self.action_logits(hidden)
        return Categorical(logits=logits / self._temperature), recurrent_state

    def compute_value_shared(
            self,
            observation: Tensor,
            hidden: Tensor,
            recurrent_state: Optional[Tensor] = None,
            train: bool = False) -> Tuple[Tensor, Optional[Tensor]]:
        return self.value(hidden).squeeze(-1), recurrent_state
