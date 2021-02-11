from math import ceil
from os import makedirs
from os.path import join
from typing import Dict

import gym
import torch
from gym import Env
from gym.wrappers import AtariPreprocessing, FrameStack
from torch import nn, Tensor
from torch.utils.tensorboard import SummaryWriter

from experiments.ppo.models import AtariActorCriticNetwork
from experiments.utils import *
from rl.agents import *
from rl.algorithms import *
from rl.envs import *
from rl.spaces import *
from rl.utils import *


def main() -> None:
    game = 'Pong'
    device = torch.device('cuda:0')

    # TODO args + seed

    out_dir = f'out/ppo/{game.lower()}'
    makedirs(out_dir, exist_ok=True)

    def make_atari_env(game: str) -> Env:
        env = gym.make(f'{game}NoFrameskip-v4')
        env = RemoveALEInfo(env)
        env = AtariPreprocessing(
            env, frame_skip=4, screen_size=84,
            terminal_on_life_loss=False, grayscale_obs=True)
        env = FrameStack(env, num_stack=4)
        return env

    num_envs = 8
    env = ImprovedAsyncVectorEnv([lambda: make_atari_env(game)] * num_envs)
    env = TensorWrapper(env, device)
    env = RecordEpisodeStatistics(env)

    network = AtariActorCriticNetwork(
        num_actions=env.single_action_space.n, device=device)

    agent = ActorCriticAgent(network)
    sampler = EnvSampler(env, agent)

    optimizer = torch.optim.Adam(network.parameters(), lr=0.00025)

    def optimize(loss: Tensor) -> None:
        optimizer.zero_grad()
        loss.backward()
        nn.utils.clip_grad_norm_(network.parameters(), max_norm=0.5)
        optimizer.step()

    summary_writer = SummaryWriter(out_dir)
    logger = StatsLogger(summary_writer)

    def log_stats(batch: PPOSample, algo_stats: Dict[str, Tensor]) -> None:
        stats = {'env': get_env_stats(batch.info), 'train': algo_stats}
        batch_shape = batch.observation.shape
        logger.log(stats, num_steps=batch_shape[0] * batch_shape[1])

    algo = ProximalPolicyOptimization(
        network, optimize_fn=optimize, logging_fn=log_stats,
        num_steps=128, num_epochs=4, minibatch_size=256,
        discount_rate=0.99, gae_lambda=0.95, clip=0.1, value_loss_clip=0.1,
        value_loss_coef=0.5, entropy_coef=0.01, normalize_advantage=True)

    num_interactions = int(1e7)
    interactions_per_update = env.num_envs * algo.num_steps
    num_updates = ceil(num_interactions / interactions_per_update)

    sampler.reset()
    algo.start()
    for _ in range(num_updates):
        algo.update(sampler)

    summary_writer.close()

    torch.save(network.state_dict(), join(out_dir, 'network.pt'))


if __name__ == '__main__':
    main()
