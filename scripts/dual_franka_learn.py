import argparse
from distutils.log import debug

import gym
from importlib_metadata import entry_points
import numpy as np
import torch

from stable_baselines3 import PPO
from sb3_contrib import ARS

from gym.envs.registration import register

register(   id='DualFrankaPandaObjectsBulletEnv-v0', 
            entry_point='consensus_normflow.envs:DualFrankaPandaObjectsBulletEnv',
            max_episode_steps=250)

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
    train_env = gym.make('DualFrankaPandaObjectsBulletEnv-v0', args=env_args)
    eval_env = gym.make('DualFrankaPandaObjectsBulletEnv-v0', args=env_args)

    train_env.seed(args.seed)
    eval_env.seed(args.seed)

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
        policy_kwargs = dict(activation_fn=torch.nn.ReLU)
        agent = PPO(policy, train_env, policy_kwargs=policy_kwargs, n_steps=2500, batch_size=2500, n_epochs=10, verbose=1)
    elif args.rl_algo == 'ARS':
        raise NotImplementedError
    else:
        raise NotImplementedError

    tol_timesteps = 250000
    num_steps_between_save = tol_timesteps // 10

    cb = CustomCallback(eval_env, args.logdir, 1, args,
                num_steps_between_save=num_steps_between_save,
                viz=args.viz, debug=args.debug)

    agent.learn(total_timesteps=250000, callback=cb)


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
