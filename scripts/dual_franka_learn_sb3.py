import argparse
from distutils.log import debug

import gym
from importlib_metadata import entry_points
import numpy as np
import torch

from stable_baselines3 import PPO
from stable_baselines3.common.env_util import (
    make_vec_env, DummyVecEnv, SubprocVecEnv)
from sb3_contrib import ARS

from gym.envs.registration import register

max_episode_steps = 400

register(   id='DualFrankaPandaObjectsBulletEnv-v0', 
            entry_point='consensus_normflow.envs:DualFrankaPandaObjectsBulletEnv',
            max_episode_steps=max_episode_steps)

from consensus_normflow.consensus_normflow.normflow_ds import ConsensusNormalizingFlowDynamics, ConsensusDuoNormalizingFlowDynamics
from consensus_normflow.consensus_normflow.consensus_policy_sb3 import ConsensusNormalizingflowACPolicy


import os
import pickle

import cv2
import torch

from consensus_normflow.utils.train_sb3_utils import CustomCallback, init_train

def main(args):
    np.random.seed(args.seed)
    torch.random.manual_seed(args.seed)

    #default parameters for creating envs
    env_args = argparse.Namespace(viz=False, force_ctrl=True, debug=False)

    args.logdir, args.device = init_train(args.rl_algo, args)

    #sb3 cannot use vector envs
    vec_env = gym.make('DualFrankaPandaObjectsBulletEnv-v0', args=env_args)
    n_envs = 1


    eval_env = gym.make('DualFrankaPandaObjectsBulletEnv-v0', args=env_args)

    #note this is only useful for on policy algo
    # n_envs = args.num_envs
    # vec_env = make_vec_env(
    #     'DualFrankaPandaObjectsBulletEnv-v0', n_envs=n_envs,
    #     vec_env_cls=SubprocVecEnv if n_envs > 1 else DummyVecEnv,
    #     env_kwargs={'args': env_args})
    # vec_env.seed(args.seed)
    # print('Created dual-arm task with observation_space',
    #       vec_env.observation_space.shape, 'action_space',
    #       vec_env.action_space.shape)

    # train_env.seed(args.seed)
    eval_env.seed(args.seed)

    n_episode_per_epoch = 10
    n_epoch = 100

    #prepare policy
    if args.rl_policy == 'NN':
        policy = "MlpPolicy"    #need more specifications for its arch
    else:
        if args.rl_algo == 'PPO':
            #create an AC policy
            policy = ConsensusNormalizingflowACPolicy
        elif args.rl_alg == 'ARS':
            #create an ARS policy
            raise NotImplementedError
        else:
            raise NotImplementedError

    if args.rl_algo == 'PPO':
        #preferred to use policy_kwargs for custom policies see https://github.com/DLR-RM/stable-baselines3/issues/168
        #and https://github.com/DLR-RM/stable-baselines3/issues/99
        policy_kwargs = dict(activation_fn=torch.nn.ReLU, net_arch=[16, 16], log_std_init=1.)
        agent = PPO(policy, vec_env, learning_rate=args.lr, 
                    policy_kwargs=policy_kwargs, 
                    n_steps=max_episode_steps * n_episode_per_epoch, 
                    batch_size=max_episode_steps, 
                    n_epochs=n_episode_per_epoch, verbose=1)
    elif args.rl_algo == 'ARS':
        raise NotImplementedError
    else:
        raise NotImplementedError

    
    tol_timesteps = max_episode_steps * n_episode_per_epoch * n_epoch    #rollout length * episode per epoch * epoch
    num_steps_between_save = tol_timesteps // 10    #every 10 epoch to save a checkpoint

    cb = CustomCallback(eval_env, args.logdir, n_envs, args,
                num_steps_between_save=num_steps_between_save,
                viz=args.viz, debug=args.debug)

    agent.learn(total_timesteps=tol_timesteps, callback=cb)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='args', add_help=True)
    parser.add_argument('--rl_policy', type=str, default='NN',
                        choices=['NN', 'CNF', 'CNF-TI'],
                        help='Name of policy used to train')
    parser.add_argument('--rl_algo', type=str, default='PPO',
                        choices=['PPO', 'ARS'],
                        help='Name of RL algo from Stable Baselines to train')
    parser.add_argument('--lr', type=float, default=1e-4,
                        help='Learning rate for training')
    parser.add_argument('--seed', type=int, default=1,
                        help='Random seed to train')
    parser.add_argument('--num_envs', type=int, default=8,
                        help='Number of parallel envs.')
    parser.add_argument('--play', type=bool, default=0,
                        help='If play the trained policy')
    parser.add_argument('--load_checkpt', type=str, default='.',
                        help='Path to load checkpoint')
    parser.add_argument('--logdir', type=str, default=None,
                        help='Path for logs')
    parser.add_argument('--disable_logging_video', action='store_true',
                        help='Whether to disable dumping video to logger')
    parser.add_argument('--use_wandb', action='store_true',
                        help='Whether to enable logging to wandb.ai')
    parser.add_argument('--device', type=str, default='cuda:0',
                    help='Name/ID of the device for training.')
    parser.add_argument('--viz', action='store_true',
                        help='Whether to visualize')    
    parser.add_argument('--debug', action='store_true',
                        help='Whether to print debug info')
    args, _ = parser.parse_known_args()
    main(args)
