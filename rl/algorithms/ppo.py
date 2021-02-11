from collections import defaultdict
from typing import Any, Callable, Dict, NamedTuple, Optional, Sequence, Tuple, Union

import torch
from torch import Tensor

from rl.algorithms.base import Algorithm
from rl.networks.actor_critic import ActorCriticNetwork
from rl.utils.return_utils import compute_return, compute_gae_return
from rl.utils.sampler import EnvSampler

__all__ = ['ProximalPolicyOptimization', 'PPOSample', 'PPOMinibatch']

PPOSample = NamedTuple(
    'PPOSample',
    [('observation', Tensor), ('reward', Tensor), ('terminal', Tensor), ('info', Sequence[Tuple[Dict[str, Any], ...]]),
     ('action', Tensor), ('log_prob', Tensor), ('value', Tensor), ('next_observation', Tensor)])

PPORecurrentSample = NamedTuple(
    'PPORecurrentSample',
    [('observation', Tensor), ('reward', Tensor), ('terminal', Tensor), ('info', Sequence[Tuple[Dict[str, Any], ...]]),
     ('action', Tensor), ('log_prob', Tensor), ('value', Tensor), ('next_observation', Tensor),
     ('recurrent_state', Tensor)])

PPOMinibatch = NamedTuple(
    'PPOMinibatch',
    [('observation', Tensor), ('action', Tensor), ('log_prob', Tensor), ('value', Tensor), ('return_', Tensor)])

PPORecurrentMinibatch = NamedTuple(
    'PPOMinibatch',
    [('observation', Tensor), ('action', Tensor), ('log_prob', Tensor), ('value', Tensor), ('return_', Tensor),
     ('recurrent_state', Tensor)])


class ProximalPolicyOptimization(Algorithm):
    """TODO docstring"""

    def __init__(
            self,
            network: ActorCriticNetwork,
            optimize_fn: Callable[[Tensor], None],
            logging_fn: Callable[[PPOSample, Dict[str, Tensor]], None],
            num_steps: int,
            num_epochs: int,
            minibatch_size: int,
            discount_rate: float,
            gae_lambda: Optional[float],
            clip: float,
            value_loss_clip: float,
            value_loss_coef: float,
            entropy_coef: float,
            normalize_advantage: bool) -> None:
        super().__init__()
        self.network = network
        self.optimize_fn = optimize_fn
        self.logging_fn = logging_fn
        self.num_steps = num_steps
        self.num_epochs = num_epochs
        self.minibatch_size = minibatch_size
        self.discount_rate = discount_rate
        self.gae_lambda = gae_lambda
        self.clip = clip
        self.value_loss_clip = value_loss_clip
        self.value_loss_coef = value_loss_coef
        self.entropy_coef = entropy_coef
        self.normalize_advantage = normalize_advantage

    def start(self) -> None:
        """TODO docstring"""
        pass

    def update(self, sampler: EnvSampler) -> None:
        """TODO docstring"""
        if self.network.training:
            self.network.eval()

        batch = sampler.sample(
            self.num_steps, PPORecurrentSample if self.network.is_recurrent else PPOSample,
            train=True, batch_first=False)

        with torch.no_grad():
            bootstrap_observation = batch.next_observation.squeeze(0)
            bootstrap_recurrent_state = batch.recurrent_state[-1] if self.network.is_recurrent else None
            bootstrap_value, _ = \
                self.network.compute_value(bootstrap_observation, bootstrap_recurrent_state, train=False)
            if self.gae_lambda is not None:
                return_ = compute_gae_return(
                    batch.reward, batch.terminal, batch.value, bootstrap_value, self.discount_rate, self.gae_lambda,
                    batch_first=False)
            else:
                return_ = compute_return(
                    batch.reward, batch.terminal, bootstrap_value, self.discount_rate, batch_first=False)

        if self.network.is_recurrent:
            flat_batch = PPORecurrentMinibatch(
                batch.observation.flatten(0, 1), batch.action.flatten(0, 1), batch.log_prob.flatten(0, 1),
                batch.value.flatten(0, 1), return_.flatten(0, 1), batch.recurrent_state.flatten(0, 1))
        else:
            flat_batch = PPOMinibatch(
                batch.observation.flatten(0, 1), batch.action.flatten(0, 1), batch.log_prob.flatten(0, 1),
                batch.value.flatten(0, 1), return_.flatten(0, 1))

        if not self.network.training:
            self.network.train()

        accumulated_stats = defaultdict(lambda: [])
        num_updates = 0

        for epoch in range(self.num_epochs):
            epoch_minibatch_indices = torch.randperm(flat_batch.observation.shape[0]).split(self.minibatch_size)

            for minibatch_indices in epoch_minibatch_indices:
                if self.network.is_recurrent:
                    minibatch = PPORecurrentMinibatch(
                        flat_batch.observation[minibatch_indices],
                        flat_batch.action[minibatch_indices],
                        flat_batch.log_prob[minibatch_indices],
                        flat_batch.value[minibatch_indices],
                        flat_batch.return_[minibatch_indices],
                        flat_batch.recurrent_state[minibatch_indices])
                else:
                    minibatch = PPOMinibatch(
                        flat_batch.observation[minibatch_indices],
                        flat_batch.action[minibatch_indices],
                        flat_batch.log_prob[minibatch_indices],
                        flat_batch.value[minibatch_indices],
                        flat_batch.return_[minibatch_indices])

                loss, stats = self.compute_loss(minibatch)
                self.optimize_fn(loss)
                for (key, stat) in stats.items():
                    accumulated_stats[key].append(stat)
                num_updates += 1
                del loss, stats, minibatch

        algo_stats = {key: sum(accumulated) / num_updates for (key, accumulated) in accumulated_stats.items()}
        self.logging_fn(batch, algo_stats)

    def compute_loss(self, minibatch: Union[PPOMinibatch, PPORecurrentMinibatch]) -> Tuple[Tensor, Dict[str, Tensor]]:
        """TODO docstring"""
        recurrent_state = minibatch.recurrent_state if self.network.is_recurrent else None
        new_action_distribution, new_value, _ = \
            self.network.compute_action_distribution_and_value(minibatch.observation, recurrent_state)
        new_log_prob = new_action_distribution.log_prob(minibatch.action)

        advantage = minibatch.return_ - minibatch.value
        if self.normalize_advantage:
            advantage = (advantage - advantage.mean()) / (advantage.std() + 1e-8)

        ratio = torch.exp(new_log_prob - minibatch.log_prob)
        surr1 = ratio * advantage
        surr2 = torch.clamp(ratio, 1.0 - self.clip, 1.0 + self.clip) * advantage
        entropy = new_action_distribution.entropy().mean(0)
        pg_objective = torch.min(surr1, surr2).mean(0) + self.entropy_coef * entropy
        pg_loss = -pg_objective

        if self.value_loss_clip is not None:
            clipped_value = \
                minibatch.value + (new_value - minibatch.value).clamp(-self.value_loss_clip, self.value_loss_clip)
            vf_loss = 0.5 * torch.max(
                (minibatch.return_ - new_value).square(),
                (minibatch.return_ - clipped_value).square()).mean()
        else:
            vf_loss = 0.5 * (minibatch.return_ - new_value).square().mean()

        loss = pg_loss + self.value_loss_coef * vf_loss
        stats = {
            'loss': loss.detach().clone(),
            'pg_loss': pg_loss.detach().clone(),
            'vf_loss': vf_loss.detach().clone(),
            'entropy': entropy.detach().clone()
        }
        return loss, stats
