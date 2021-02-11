from argparse import ArgumentParser
from os import makedirs, listdir
from os.path import exists, isdir
from typing import Dict, Optional

import gym
import torch
from gym import Env
from gym.wrappers import AtariPreprocessing, FrameStack
from matplotlib.colors import hsv_to_rgb
from torch import nn, Tensor, optim
from torch.utils.tensorboard import SummaryWriter
from torchvision.utils import make_grid, save_image

from experiments.smaller_world_models.models import *
from experiments.utils import *
from layer_utils.modes import mode
from rl.agents import *
from rl.algorithms import *
from rl.envs import *
from rl.modelbased import *
from rl.spaces import *
from rl.utils import *


def main() -> None:
    parser = ArgumentParser()
    parser.add_argument('game', type=str)
    parser.add_argument('output_dir', type=str)
    parser.add_argument('--device', nargs='?', type=str, const=None, default=None)
    parser.add_argument('--seed', type=int, default=None)
    parser.add_argument('--force', action='store_true', default=False)
    args = parser.parse_args()

    device = set_current_device(args.device)
    if args.seed is not None:
        set_seed(args.seed)

    makedirs(args.output_dir, exist_ok=True)
    if exists(args.output_dir) and isdir(args.output_dir):
        if len(listdir(args.output_dir)) > 0:
            if not args.force:
                print(f'The specified output directory "{args.output_dir}" is not empty.')
                print('Add --force to allow a non-empty directory.')
                exit(1)

    summary_writer = SummaryWriter(args.output_dir)

    # setup hyperparameters
    state_repr_batch_size = 40
    dynamics_batch_size = 4
    dynamics_num_steps = 11
    dynamics_context_size = 4
    reward_loss_weight = 1e-6
    num_real_steps = (12800,) + (6400,) * 14
    num_warmup_epochs = 50
    num_epochs = (140, 32, 24, 20, 16, 14, 12, 11, 10, 9, 8, 8, 7, 7, 6)
    state_repr_intervals = (2, 2, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3)
    num_simulated_updates = (1000, 1000, 1000, 1000, 1000, 1000, 1000, 2000, 1000, 1000, 1000, 2000, 1000, 1000, 3000)
    num_real_eval_envs = 32
    framestack = 1
    num_embeddings = 128
    recurrent_agent = False
    eval_last_only = True

    debug = True
    profile = False
    if debug:
        state_repr_batch_size = 40
        dynamics_batch_size = 4
        num_real_steps = (40,) * 15
        num_warmup_epochs = 5
        num_epochs = (5,) * 15
        num_simulated_updates = (1,) * 15
        num_real_eval_envs = 3
    elif profile:
        num_real_steps = (640,) * 15
        num_warmup_epochs = 5
        num_epochs = (5,) * 15
        num_simulated_updates = (20,) * 15
        num_real_eval_envs = 8

    # setup real environments
    def make_atari_env(game: str, seed: Optional[int] = None) -> Env:
        env = gym.make(f'{game}NoFrameskip-v4')
        if seed is not None:
            env.seed(seed)
        env = RemoveALEInfo(env)
        env = AtariPreprocessing(env, frame_skip=4, screen_size=96, terminal_on_life_loss=False, grayscale_obs=True)
        env = FrameStack(env, num_stack=framestack)
        return env

    real_env = SingleVectorEnv(make_atari_env(args.game, seed=args.seed))
    real_env = TensorWrapper(real_env, device)
    real_env = RecordEpisodeStatistics(real_env)
    real_env = RemoveEmptyInfo(real_env)

    # setup world model
    keep_trajectories_on_gpu = True
    real_trajectory_buffer = TrajectoryBuffer(
        ['observation', 'reward', 'terminal', 'action'],
        device=None if keep_trajectories_on_gpu else torch.device('cpu'))

    num_actions = real_env.single_action_space.n
    env_model = AtariDiscreteLatentSpaceWorldModel(
        framestack, num_embeddings, num_actions, device, real_trajectory_buffer)

    def get_num_params(module):
        return sum([param.numel() for param in module.parameters()])

    # setup world model optimizers
    state_repr_optimizer = optim.Adam(env_model.state_repr.parameters(), lr=1e-4)

    state_repr_scheduler = optim.lr_scheduler.LambdaLR(
        state_repr_optimizer,
        lambda epoch: 10.0 if epoch < num_warmup_epochs else
        1.0 - 0.0 * ((epoch - num_warmup_epochs) / (sum(num_epochs) - 1)))

    state_repr_logger = StatsLogger(summary_writer, defer_synchronize_steps=10)

    def train_state_repr_model(observation: Tensor) -> None:
        state_repr_optimizer.zero_grad()
        loss, stats = env_model.compute_state_representation_loss(observation)
        loss.backward()
        state_repr_optimizer.step()
        state_repr_logger.log({'train_state_representation': stats}, num_steps=observation.shape[0])
        del loss, stats

    dynamics_optimizer = optim.Adam([
        {'params':
             tuple(env_model.dynamics.cell1.parameters()) +
             tuple(env_model.dynamics.cell2.parameters()) +
             tuple(env_model.dynamics.next_latent_head.parameters()),
         'lr': 5e-4},
        {'params': env_model.dynamics.reward_head.parameters(),
         'lr': 1e-3}
    ])

    dynamics_logger = StatsLogger(summary_writer, defer_synchronize_steps=10)

    def train_dynamics_model(observation: Tensor, action: Tensor, reward: Tensor, terminal: Tensor) -> None:
        dynamics_optimizer.zero_grad()
        loss, stats = env_model.compute_dynamics_loss(
            observation, action, reward, terminal, dynamics_context_size, reward_loss_weight)
        loss.backward()
        dynamics_optimizer.step()

        num_steps = observation.shape[0] * observation.shape[1]
        dynamics_logger.log({'train_dynamics': stats}, num_steps)

    # setup agent
    agent_network = AtariLatentSpaceActorCriticNetwork(
        env_model.state_repr.embedding_size, num_actions=real_env.single_action_space.n,
        recurrent=recurrent_agent, device=device)

    agent = ActorCriticAgent(agent_network)
    wrapped_agent = ModelAgentWrapper(agent, env_model)
    real_random_agent = RandomAgent.for_env(real_env)
    if args.seed is not None:
        real_random_agent.action_space.seed(args.seed)

    # setup agent optimizer
    agent_optimizer = optim.Adam(agent_network.parameters(), lr=0.00025)

    def optimize_agent_network(loss: Tensor) -> None:
        # since PPO optimizes multiple times per batch, logging is done in
        # a separate function, log_ppo_stats
        agent_optimizer.zero_grad()
        loss.backward()
        nn.utils.clip_grad_norm_(agent_network.parameters(), max_norm=0.5)
        agent_optimizer.step()

    # setup PPO
    ppo_logger = StatsLogger(summary_writer, defer_synchronize_steps=5)

    def log_ppo_stats(batch: PPOSample, algo_stats: Dict[str, Tensor]) -> None:
        stats = {'simulated_env': get_env_stats(batch.info), 'train_ppo': algo_stats}
        batch_shape = batch.observation.shape
        ppo_logger.log(stats, num_steps=batch_shape[0] * batch_shape[1])

    ppo = ProximalPolicyOptimization(
        agent_network, optimize_fn=optimize_agent_network, logging_fn=log_ppo_stats, num_steps=50, num_epochs=4,
        minibatch_size=256, discount_rate=0.99, gae_lambda=0.95, clip=0.1, value_loss_clip=0.1, value_loss_coef=0.5,
        entropy_coef=0.01, normalize_advantage=True)

    # setup SimPLe
    simulated_env_logger = StatsLogger(summary_writer)

    def collect_real_data(real_sampler: EnvSampler, buffer: TrajectoryBuffer, iteration: int) -> None:
        print('Collect real experience')

        agent_network.temperature = 1.5  # TODO
        if iteration == 0:
            real_sampler.reset()
        else:
            real_sampler.reset_recurrent_state()
        real_sample = real_sampler.sample(num_real_steps[iteration], ObsRewTermActInfo, train=False)
        agent_network.temperature = 1.0

        # store collected real experience in buffer
        buffer.add_trajectory(real_sample)

        # log statistics of real experience
        stats = {'collect_real_env': get_env_stats(real_sample.info)}
        batch_shape = real_sample.observation.shape
        simulated_env_logger.log(stats, num_steps=batch_shape[0] * batch_shape[1])

    def train_supervised(
            env_model: AtariDiscreteLatentSpaceWorldModel,
            buffer: TrajectoryBuffer,
            iteration: int) -> None:
        # train the world model
        if not env_model.training:
            env_model.train()

        if iteration == 0:
            print('Warm up state representation model')

            for epoch in range(num_warmup_epochs):
                minibatch_gen = buffer.generate_minibatches(
                    Obs, minibatch_size=state_repr_batch_size, minibatch_steps=1, max_random_offset=0, shuffle=True,
                    drop_last_batch=False, drop_last_time=False)
                for minibatch in minibatch_gen:
                    train_state_repr_model(minibatch.observation.squeeze(0))
                    del minibatch
                del minibatch_gen

                state_repr_scheduler.step()

            if not eval_last_only:
                run_latent_analysis('warmup')
                run_dynamics_analysis('warmup')

        print('Train world model')
        for epoch in range(num_epochs[iteration]):
            state_repr_minibatch_gen = buffer.generate_minibatches(
                Obs, minibatch_size=state_repr_batch_size, minibatch_steps=1, max_random_offset=0, shuffle=True,
                drop_last_batch=False, drop_last_time=False, last_trajectory_first=False, device=device)  # TODO =True?

            dynamics_minibatch_gen = buffer.generate_minibatches(
                ObsRewTermAct, minibatch_size=dynamics_batch_size, minibatch_steps=dynamics_num_steps,
                max_random_offset=dynamics_num_steps - 1, shuffle=True, drop_last_batch=False, drop_last_time=True,
                device=device)

            has_state_repr_minibatches = True
            has_dynamics_minibatches = True
            i = 0
            while has_dynamics_minibatches:
                if has_state_repr_minibatches and state_repr_intervals[iteration] > 0 and \
                        i % state_repr_intervals[iteration] == 0:
                    try:
                        minibatch = next(state_repr_minibatch_gen)
                        train_state_repr_model(minibatch.observation.squeeze(0))
                        del minibatch
                    except StopIteration:
                        has_state_repr_minibatches = False

                if has_dynamics_minibatches:
                    try:
                        minibatch = next(dynamics_minibatch_gen)
                        train_dynamics_model(
                            minibatch.observation, minibatch.action, minibatch.reward, minibatch.terminal)
                        del minibatch
                    except StopIteration:
                        has_dynamics_minibatches = False

                i += 1

            state_repr_scheduler.step()

            del state_repr_minibatch_gen, dynamics_minibatch_gen

        state_repr_logger.synchronize()
        dynamics_logger.synchronize()

        if not eval_last_only or iteration == len(num_epochs) - 1:
            run_latent_analysis(f'iteration-{iteration + 1}')
            run_dynamics_analysis(f'iteration-{iteration + 1}')

    def train_rl(simulated_sampler: EnvSampler, iteration: int) -> None:
        # train agent with PPO using simulated experience from the world model
        print('Train agent')
        for _ in range(num_simulated_updates[iteration]):
            simulated_sampler.reset()
            ppo.update(simulated_sampler)
        ppo_logger.synchronize()

    # setup simulated environment
    # TODO make variables
    simulated_env = SimulatedEnv(env_model, batch_size=16, max_episode_steps=50)
    simulated_env = RecordEpisodeStatistics(simulated_env)
    simulated_env = RemoveEmptyInfo(simulated_env)

    algo = SimulatedPolicyLearning(
        env_model,
        real_env=real_env,
        simulated_env=simulated_env,
        real_agent=real_random_agent,
        simulated_agent=agent,
        real_data_buffer=real_trajectory_buffer,
        collect_real_data_fn=collect_real_data,
        train_supervised_fn=train_supervised,
        train_rl_fn=train_rl
    )

    # setup evaluation environment
    real_eval_env = ImprovedAsyncVectorEnv([lambda: make_atari_env(args.game)] * num_real_eval_envs)
    real_eval_env = TensorWrapper(real_eval_env, device)
    real_eval_env = RecordEpisodeStatistics(real_eval_env)
    real_eval_env = RemoveEmptyInfo(real_eval_env)
    real_eval_sampler = EnvSampler(real_eval_env, wrapped_agent)

    def generate_colors(num_colors: int) -> Tensor:
        num_saturations = 4
        min_saturation = 0.6
        colors = []
        for i in range(num_colors):
            h = i / (num_colors - 1)
            s = min_saturation + ((i % num_saturations) / (num_saturations - 1)) * (1 - min_saturation)
            v = 1
            color = hsv_to_rgb((h, s, v))
            color = torch.from_numpy(color * 255).to(torch.uint8)
            colors.append(color)
        colors = torch.stack(colors, dim=0)
        return colors

    embedding_index_colors = generate_colors(env_model.state_repr.num_embeddings).to(device)

    def run_latent_analysis(tag: str) -> None:
        # only for evaluation
        with torch.no_grad():
            x = real_eval_sampler.sample_episodes(1, Obs, max_episode_steps=400)[0].observation.squeeze(1)
            x = env_model._prepare_observations(x)
            _, z, _, x_posterior = env_model.state_repr(x)
            x_recon = mode(x_posterior)

            z = embedding_index_colors[z].permute(0, 3, 1, 2)
            z = z.float() / 255.
            z = torch.nn.functional.interpolate(z, (96, 96), mode='nearest')

            X = make_grid(torch.max(x, dim=1, keepdim=True)[0].repeat(1, 3, 1, 1), nrow=x.shape[0], padding=2)
            X_recon = make_grid(torch.max(x_recon, dim=1, keepdim=True)[0].repeat(1, 3, 1, 1),
                                nrow=x_recon.shape[0], padding=2)
            Z = make_grid(z, nrow=z.shape[0], padding=2)
            save_image(torch.cat((X, Z, X_recon), dim=1), f'{args.output_dir}/latent-analysis-{tag}.png')

    simulated_eval_sampler = EnvSampler(SimulatedEnv(env_model, batch_size=1, max_episode_steps=200), agent)

    def run_dynamics_analysis(tag: str) -> None:
        # only for evaluation
        with torch.no_grad():
            latents_quantized = simulated_eval_sampler.sample_episodes(1, Obs)[0].observation.squeeze(1)
            z, _ = env_model.state_repr.quantize(latents_quantized)
            x_posterior = env_model.state_repr.decode(latents_quantized)
            x_recon = mode(x_posterior)

            z = embedding_index_colors[z].permute(0, 3, 1, 2)
            z = z.float() / 255.
            z = torch.nn.functional.interpolate(z, (96, 96), mode='nearest')

            Z = make_grid(z, nrow=z.shape[0], padding=2)
            X_recon = make_grid(torch.max(x_recon, dim=1, keepdim=True)[0].repeat(1, 3, 1, 1),
                                nrow=x_recon.shape[0], padding=2)
            save_image(torch.cat((Z, X_recon), dim=1), f'{args.output_dir}/dynamics-analysis-{tag}.png')

    eval_logger = StatsLogger(summary_writer)

    def eval_real_env() -> None:
        print('Evaluate agent')
        eval_episodes = list(real_eval_sampler.sample_episodes(1, Info))
        eval_infos = []
        while len(eval_episodes) > 0:
            eval_infos.extend(eval_episodes.pop(0).info)
        env_stats = get_env_stats(eval_infos, std=True)
        eval_stats = {'eval_real_env': env_stats}
        eval_logger.log(
            eval_stats, num_steps=num_real_steps[algo.iteration - 1])
        print(f'Mean episode return: {env_stats["episode_return_mean"]:.2f}')

    algo.start()
    ppo.start()
    while algo.iteration < len(num_real_steps):
        # collect real experience, train the world model and train the agent
        print(f'Start iteration {algo.iteration + 1}/{len(num_real_steps)}')
        algo.update()

        # evaluate in real environment
        eval_real_env()
        print()

        if algo.iteration == 1:
            # switch to the trained agent in the second iteration
            algo.real_agent = wrapped_agent

    # TODO save model to file

    # cleanup
    summary_writer.close()


if __name__ == '__main__':
    main()
