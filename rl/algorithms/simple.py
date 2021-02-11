from typing import Callable, Generic, NamedTuple, TypeVar

from gym.vector import VectorEnv

from rl.agents.base import Agent
from rl.algorithms.base import Algorithm
from rl.modelbased.env_model import EnvModel
from rl.modelbased.simulated_env import SimulatedEnv
from rl.utils.sampler import EnvSampler

__all__ = ['SimulatedPolicyLearning']

M = TypeVar('M', bound=EnvModel)
B = TypeVar('B')
T = TypeVar('T', bound=NamedTuple)


class SimulatedPolicyLearning(Algorithm, Generic[M]):
    """TODO docstring"""

    def __init__(
            self,
            env_model: M,
            real_env: VectorEnv,
            simulated_env: SimulatedEnv,
            real_agent: Agent,
            simulated_agent: Agent,
            real_data_buffer: B,
            collect_real_data_fn: Callable[[EnvSampler, B, int], None],
            train_supervised_fn: Callable[[M, B, int], None],
            train_rl_fn: Callable[[EnvSampler, int], None]) -> None:
        self._env_model = env_model
        self._real_env = real_env
        self._real_agent = real_agent
        self._simulated_env = simulated_env
        self._simulated_agent = simulated_agent

        assert real_env.single_observation_space == env_model.real_observation_space
        assert real_env.single_action_space == env_model.real_action_space
        assert simulated_env.single_observation_space == env_model.simulated_observation_space
        assert simulated_env.single_action_space == env_model.simulated_action_space

        if real_agent.observation_space == env_model.simulated_observation_space:
            # agent in simulation space
            assert real_agent.action_space == env_model.simulated_action_space
        elif real_agent.observation_space == env_model.real_observation_space:
            # agent in real space
            assert real_agent.action_space == env_model.real_action_space

        assert simulated_agent.observation_space == env_model.simulated_observation_space
        assert simulated_agent.action_space == env_model.simulated_action_space

        self.real_data_buffer = real_data_buffer
        self.collect_real_data_fn = collect_real_data_fn
        self.train_supervised_fn = train_supervised_fn
        self.train_rl_fn = train_rl_fn

        self._real_sampler = None
        self._simulated_sampler = None
        self.iteration = 0

    @property
    def env_model(self) -> M:
        """TODO docstring"""
        return self._env_model

    @property
    def real_env(self) -> VectorEnv:
        """TODO docstring"""
        return self._real_env

    @property
    def real_agent(self) -> Agent:
        """TODO docstring"""
        return self._real_agent

    @real_agent.setter
    def real_agent(self, agent: Agent) -> None:
        """TODO docstring"""
        if agent.observation_space == self._env_model.simulated_observation_space:
            # agent in simulation space
            assert agent.action_space == self._env_model.simulated_action_space
        elif agent.observation_space == self._env_model.real_observation_space:
            # agent in real space
            assert agent.action_space == self._env_model.real_action_space

        self._real_agent = agent
        if self._real_sampler is not None:
            self._setup_real_sampler()

    @property
    def simulated_agent(self) -> Agent:
        """TODO docstring"""
        return self._simulated_agent

    @simulated_agent.setter
    def simulated_agent(self, agent) -> None:
        """TODO docstring"""
        assert agent.observation_space == self._env_model.simulated_observation_space
        assert agent.action_space == self._env_model.simulated_action_space

        self._simulated_agent = agent
        if self._simulated_sampler is not None:
            self._setup_simulated_sampler()

    def _setup_real_sampler(self) -> None:
        last_observation, recurrent_state = (None, None) if self._real_sampler is None else \
            (self._real_sampler.last_observation, self._real_sampler.recurrent_state)
        self._real_sampler = EnvSampler(self._real_env, self._real_agent, last_observation, recurrent_state)

    def _setup_simulated_sampler(self) -> None:
        last_observation, recurrent_state = (None, None) if self._simulated_sampler is None else \
            (self._simulated_sampler.last_observation, self._simulated_sampler.recurrent_state)
        self._simulated_sampler = EnvSampler(
            self._simulated_env, self._simulated_agent, last_observation, recurrent_state)

    def start(self, initial_iteration: int = 0) -> None:
        """TODO docstring"""
        self.iteration = initial_iteration
        self._setup_real_sampler()
        self._setup_simulated_sampler()

    def update(self) -> None:
        """TODO docstring"""
        # sample from real environment and store in buffer
        self.collect_real_data_fn(self._real_sampler, self.real_data_buffer, self.iteration)

        # train model supervised using collected data
        self.train_supervised_fn(self.env_model, self.real_data_buffer, self.iteration)

        # train agent model-free with simulated experience
        self.train_rl_fn(self._simulated_sampler, self.iteration)

        self.iteration += 1
