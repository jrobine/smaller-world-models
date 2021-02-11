from typing import Any, Dict, NamedTuple, Sequence, Tuple, Type, TypeVar, \
    Optional

import torch
from gym.vector import VectorEnv
from torch import Tensor

from rl.agents.base import Agent
from rl.spaces.base import TensorSpace

__all__ = ['EnvSampler', 'Obs', 'Rew', 'Term', 'Act', 'Info',
           'ObsRew', 'ObsRewInfo', 'ObsRewAct', 'ObsRewActInfo', 'ObsRewTerm',
           'ObsRewTermInfo', 'ObsRewTermAct', 'ObsRewTermActInfo']

T = TypeVar('T', bound=NamedTuple)


# TODO Generic[H] for recurrent state?
class EnvSampler:
    """Utility class to sample experience from environments. It is stateful and stores the last observation, which
    allows to easily continue sampling from the last step. Before sampling, reset() has to be called.

    Arguments:
        env (VectorEnv): The environment used for sampling. It has to be a vector environment, since only batches of
            experience are supported. It also needs to have :class:`rl.spaces.TensorSpace` observation and action
            spaces, because only tensors are supported.
        agent (Agent): The agent that selects the actions to take. The observation and action space have to be the same
            as the spaces of the environment.
        last_observation (Tensor, optional): The last observation returned by the environment. E.g., this can be used to
            create a new sampler without resetting the environment. Defaults to ``None``.
        recurrent_state(Tensor, optional): TODO docstring
    """

    def __init__(
            self,
            env: VectorEnv,
            agent: Agent,
            last_observation: Optional[Tensor] = None,
            recurrent_state: Optional[Tensor] = None) -> None:
        assert isinstance(env.unwrapped, VectorEnv)
        self._env = env
        self._agent = agent

        assert isinstance(env.observation_space, TensorSpace)
        assert isinstance(env.action_space, TensorSpace)
        assert isinstance(env.single_observation_space, TensorSpace)
        assert isinstance(env.single_action_space, TensorSpace)
        assert env.single_observation_space == agent.observation_space
        assert env.single_action_space == agent.action_space

        if last_observation is not None:
            assert last_observation.shape == env.observation_space.shape

        self._last_observation = last_observation
        self._recurrent_state = recurrent_state

    @property
    def env(self) -> VectorEnv:
        """Returns the environment used for sampling."""
        return self._env

    @property
    def agent(self) -> Agent:
        """Returns the agent used for sampling."""
        return self._agent

    @property
    def batch_size(self) -> int:
        """Returns the size of the batches that are returned."""
        return self._env.num_envs

    @property
    def last_observation(self) -> Optional[Tensor]:
        """Returns the last observation returned by the environment."""
        return self._last_observation

    @property
    def recurrent_state(self) -> Optional[Tensor]:
        """TODO docstring"""
        return self._recurrent_state

    def reset(self) -> None:
        """Resets the sampler."""
        self._last_observation = self._env.reset()
        self.reset_recurrent_state()

    def reset_recurrent_state(self) -> None:
        """TODO docstring"""
        self._recurrent_state = self._agent.init_recurrent_state(self._last_observation.shape[0])

    def sample(
            self,
            num_steps: int,
            named_tuple_cls: Type[T],
            train: bool = False,
            batch_first: bool = False) -> T:
        """Samples a sequence of experience from the environment using the agent.

        Args:
            num_steps (int): The number of steps to sample.
            named_tuple_cls (Type[T]): The class of the named tuple that should be returned. The field names determine
                the values which are collected. This has two advantages:
                    1. It can be specified which values should be collected.
                    2. The values can be easily accessed from the return value.

                The default supported field names for a named tuple are:
                'observation', 'reward', 'terminal', 'info', 'action', 'next_observation'.

                The return value of the agent is a dictionary, which can contain more than the action, e.g. the
                log-probabilities. Those can also be collected by adding the corresponding fields to the named tuple.

                All values will be tensors of shape (batch_size, num_steps), or (num_steps, batch_size) if `batch_first`
                is ``True``. The only exception is the value of 'info', which will be a list of tuples containing the
                info dictionaries (each tuple is a batch, and the list contains the batches of each time step).
            train (bool, optional): Whether the agent will be used for training. See :method:``rl.agents.Agent.act()``
                for more information. Defaults to ``False``.
            batch_first (bool, optional): Whether the returned tensors are of shape `(batch, time)` instead. The info
                dicts will not be changed. Defaults to ``False``.

        Returns:
            A named tuple with the type specified by `named_tuple_cls`
            containing the collected values. Tensors will be of shape
            `(num_steps, batch)` (or `(batch, num_steps)`, if `batch_first` is
            set to ``True``).

        Example:
            MyTuple = namedtuple('MyTuple', ['observation'], ['action'])

            When the class MyTuple is passed to a sampler, the return value of
            the sampling methods will be of type MyTuple and will only contain
            the collected observations and actions.
        """
        assert num_steps > 0
        if self._last_observation is None:
            raise ValueError('Call sampler.reset() before sampling.')

        fields = named_tuple_cls._fields
        result = {}

        for t in range(num_steps):
            agent_dict, recurrent_state = self.agent.act(self._last_observation, self._recurrent_state, train)
            assert 'action' in agent_dict, \
                'The return value of agent.act() has to contain an item with' \
                'the key \'action\'.'
            action = agent_dict['action']
            observation, reward, terminal, info = self.env.step(action)

            update_dict = {}
            update_dict.update(agent_dict)
            update_dict['observation'] = self._last_observation
            update_dict['reward'] = reward
            update_dict['terminal'] = terminal
            update_dict['recurrent_state'] = self._recurrent_state.detach() if recurrent_state is not None else None
            update_dict['info'] = info

            for (field, value) in update_dict.items():
                if field in fields:
                    if field not in result:
                        result[field] = []
                    result[field].append(value)

            self._last_observation = observation
            self._recurrent_state = self.agent.mask_recurrent_state(recurrent_state, terminal)

        for field in fields:
            if field != 'info' and field != 'next_observation':
                tensors = result[field]
                seq = torch.stack(tensors, 0)
                if batch_first:  # try to keep batches contiguous
                    seq = seq.transpose(0, 1)
                result[field] = seq

        if 'next_observation' in fields:
            seq_dim = 1 if batch_first else 0
            result['next_observation'] = self._last_observation.unsqueeze(seq_dim)

        named_tuple = named_tuple_cls(**result)
        return named_tuple

    def sample_episodes(
            self,
            num_episodes: int,
            named_tuple_cls: Type[T],
            train: bool = False,
            batch_first: bool = False,
            max_episode_steps: int = -1) -> Tuple[T, ...]:
        """Samples sequences of experience from the environment using the agent.
        Each sequence contains a full episode. The environment will be restarted
        before collecting.

        Args:
            num_episodes (int): The number of episodes to sample.
            named_tuple_cls (Type[T]): The class of the named tuple that should
                be returned. See `sample()` for more information.
            train (bool, optional): Whether the agent will be used for training.
                See :method:``rl.agents.Agent.act()`` for more information.
                Defaults to ``False``.
            batch_first (bool, optional): Whether the returned tensors are of
                shape `(batch, time)` instead. The info dicts will not be
                changed. Defaults to ``False``.
            max_episode_steps (int): TODO docstring

        Returns:
            A tuple of named tuples with the type specified by `named_tuple_cls`
            containing the collected episodes. Tensors will be of shape
            `(num_steps, 1)` (or `(1, num_steps)`, if `batch_first` is
            set to ``True``).
        """
        assert num_episodes > 0

        last_observation = self._env.reset()
        recurrent_state = self.agent.init_recurrent_state(last_observation.shape[0])

        fields = named_tuple_cls._fields
        assert 'next_observation' not in fields
        results = {field: tuple([] for _ in range(self.batch_size)) for field in fields}

        episodes = torch.zeros(self.batch_size, dtype=torch.long)
        unfinished = episodes < num_episodes
        steps = 0
        while unfinished.any():
            # act also in finished environments, but ignore values
            agent_dict, new_recurrent_state = self.agent.act(last_observation, recurrent_state, train)
            assert 'action' in agent_dict, \
                'The return value of agent.act() has to contain an item with' \
                'the key \'action\'.'
            action = agent_dict['action']
            observation, reward, terminal, info = self.env.step(action)

            update_dict = {}
            update_dict.update(agent_dict)
            update_dict['observation'] = last_observation
            update_dict['reward'] = reward
            update_dict['terminal'] = terminal
            update_dict['info'] = info
            update_dict['recurrent_state'] = recurrent_state.detach() if recurrent_state is not None else None

            unfinished_indices = torch.nonzero(unfinished, as_tuple=True)[0]
            for (field, tensor) in update_dict.items():
                if field in fields:
                    result_batch = results[field]
                    if field == 'info':
                        for i in unfinished_indices:
                            result_batch[i].append((tensor[i],))
                    else:
                        for i in unfinished_indices:
                            result_batch[i].append(tensor[i])
                    del result_batch

            last_observation = observation
            recurrent_state = self.agent.mask_recurrent_state(new_recurrent_state, terminal)

            episodes[terminal] += 1
            unfinished = episodes < num_episodes

            steps += 1
            if 0 <= max_episode_steps <= steps:
                break

        named_tuples = []
        batch_dim = 0 if batch_first else 1
        for i in range(self.batch_size):
            result = {}
            for field in fields:
                if field == 'info':
                    result[field] = results[field][i]
                else:
                    result[field] = torch.stack(results[field][i]).unsqueeze(batch_dim)
            named_tuple = named_tuple_cls(**result)
            named_tuples.append(named_tuple)
        return tuple(named_tuples)


Obs = NamedTuple('Obs', [('observation', Tensor)])
Rew = NamedTuple('Rew', [('reward', Tensor)])
Term = NamedTuple('Term', [('terminal', Tensor)])
Act = NamedTuple('Act', [('action', Tensor)])
Info = NamedTuple('Info', [('info', Sequence[Tuple[Dict[str, Any], ...]])])
ObsRew = NamedTuple(
    'ObsRew',
    [('observation', Tensor),
     ('reward', Tensor)])
ObsRewInfo = NamedTuple(
    'ObsRewInfo',
    [('observation', Tensor),
     ('reward', Tensor),
     ('info', Sequence[Tuple[Dict[str, Any], ...]])])
ObsRewAct = NamedTuple(
    'ObsRewAct',
    [('observation', Tensor),
     ('reward', Tensor),
     ('action', Tensor)])
ObsRewActInfo = NamedTuple(
    'ObsRewActInfo',
    [('observation', Tensor),
     ('reward', Tensor),
     ('action', Tensor),
     ('info', Sequence[Tuple[Dict[str, Any], ...]])])
ObsRewTerm = NamedTuple(
    'ObsRewTerm',
    [('observation', Tensor),
     ('reward', Tensor),
     ('terminal', Tensor)])
ObsRewTermInfo = NamedTuple(
    'ObsRewTermInfo',
    [('observation', Tensor),
     ('reward', Tensor),
     ('terminal', Tensor),
     ('info', Sequence[Tuple[Dict[str, Any], ...]])])
ObsRewTermAct = NamedTuple(
    'ObsRewTermAct',
    [('observation', Tensor),
     ('reward', Tensor),
     ('terminal', Tensor),
     ('action', Tensor)])
ObsRewTermActInfo = NamedTuple(
    'ObsRewTermActInfo',
    [('observation', Tensor),
     ('reward', Tensor),
     ('terminal', Tensor),
     ('action', Tensor),
     ('info', Sequence[Tuple[Dict[str, Any], ...]])])
