import torch
from torch import Tensor

__all__ = ['compute_return', 'compute_gae_return']


def compute_return(
        rewards: Tensor,
        terminals: Tensor,
        bootstrap_value: Tensor,
        discount_rate: float,
        batch_first: bool = False) -> Tensor:
    """Computes the return based on sequences of rewards, terminals and the bootstrapped values of the next observation.

    Arguments:
        rewards (Tensor): Float tensor of shape `(time, batch)` that contains the reward sequence.
        terminals (Tensor): Boolean tensor of shape `(time, batch)` that contains the terminal sequence.
        bootstrap_value (Tensor): Float tensor of shape `(batch)` that contains the bootstrapped values of the next
            obsrvation.
        discount_rate (float): The discount rate that is used to compute the return.
        batch_first (bool, optional): Whether the rewards and terminals are of shape `(batch, time)` instead. The
            returns will also be of that shape. Defaults to ``False``.

    Returns:
        Tensor of shape `(time, batch)` containing the computed returns (or of shape `(batch, time)`, if `batch_first`
        is ``True``).
    """
    return_ = torch.zeros_like(rewards)
    next_return = bootstrap_value
    num_steps = rewards.shape[1 if batch_first else 0]

    for t in reversed(range(num_steps)):
        index_t = (slice(None), t) if batch_first else (t, slice(None))
        non_terminal = ~terminals[index_t]
        return_[index_t] = rewards[index_t]
        return_[index_t][non_terminal] += discount_rate * next_return[non_terminal]
        next_return = return_[index_t]

    return return_


def compute_gae_return(
        rewards: Tensor,
        terminals: Tensor,
        values: Tensor,
        bootstrap_value: Tensor,
        discount_rate: float,
        gae_lambda: float,
        batch_first: bool = False) -> Tensor:
    """TODO docstring

    Implementation adopted from [1].

    References:
        [1] https://stable-baselines.readthedocs.io/en/master/_modules/stable_baselines/ppo2/ppo2.html#PPO2
    """
    return_ = torch.zeros_like(rewards)
    next_value = bootstrap_value
    old_gae = torch.zeros_like(next_value)
    num_steps = rewards.shape[1 if batch_first else 0]
    for t in reversed(range(num_steps)):
        index_t = (slice(None), t) if batch_first else (t, slice(None))
        non_terminal = ~terminals[index_t]
        value_t = values[index_t]
        delta = rewards[index_t] - value_t
        delta[non_terminal] += discount_rate * next_value[non_terminal]
        gae = delta.clone()
        gae[non_terminal] += discount_rate * gae_lambda * old_gae[non_terminal]
        old_gae = gae
        return_[index_t] = gae + value_t
        next_value = value_t

    return return_
