import torch
import torch.nn as nn

from garage import wrap_experiment
# from garage.envs import GymEnv, normalize
from garage.envs.bullet import BulletEnv
from garage.envs import normalize

from garage.experiment.deterministic import set_seed

from garage.torch.algos import PPO
from garage.torch.policies import GaussianMLPPolicy
from garage.torch.value_functions import GaussianMLPValueFunction
from garage.trainer import Trainer
from garage.sampler import DefaultWorker, LocalSampler, VecWorker, RaySampler

from gym.envs.registration import register

from consensus_normflow.consensus_normflow.consensus_policy_garage import GaussianConsensusNormalizingFlowPolicy

max_episode_steps = 250

register(   id='DualFrankaPandaObjectsBulletEnv-v0', 
            entry_point='consensus_normflow.envs:DualFrankaPandaObjectsBulletEnv',
            max_episode_steps=max_episode_steps)

@wrap_experiment
def dualfranka_bullet_tests(ctxt=None, config=None):
    assert(config is not None)
    seed = config.seed
    policy_type = config.rl_policy

    isRendering = False

    set_seed(seed)
    trainer = Trainer(snapshot_config=ctxt)

    env = normalize(BulletEnv('DualFrankaPandaObjectsBulletEnv-v0'))
    # print(env._env.isRendering)
    # env._env.isRendering = isRendering

    #need a separate seed for gym environment for full determinism
    env.seed(seed)
    env.action_space.seed(seed)

    #original hidden size 256
    hidden_size = 32

    if policy_type == 'NN':
        print('Using Vanilla NN Policy')
        policy = GaussianMLPPolicy(env.spec,
                                hidden_sizes=[hidden_size, hidden_size],
                                hidden_nonlinearity=torch.relu,
                                output_nonlinearity=None,
                                init_std=1.0
                                )

    elif policy_type == 'CNF-N':
        print('Using Consensus NormalizingFlow Policy')
        policy = GaussianConsensusNormalizingFlowPolicy(env.spec,
                                                        nfds=None,  #use default nfds parameters, see the policy implementation
                                                        use_ti=False,
                                                        init_std=1.0)
    else:
        print('Using Translational Invariant Consensus NormalizingFlow Policy')
        policy = GaussianConsensusNormalizingFlowPolicy(env.spec,
                                                        nfds=None,  #use default nfds parameters, see the policy implementation
                                                        init_std=1.0)



    #shared settings
    value_function = GaussianMLPValueFunction(env_spec=env.spec,
                                            hidden_sizes=(hidden_size, hidden_size),
                                            hidden_nonlinearity=torch.relu,
                                            output_nonlinearity=None)


    # sampler = MultiprocessingSampler(agents=policy,
    #                     envs=env,
    #                     max_episode_length=env.spec.max_episode_length,
    #                     worker_class=DefaultWorker)
    sampler = LocalSampler(agents=policy,
                        envs=env,
                        max_episode_length=env.spec.max_episode_length,
                        worker_class=DefaultWorker)

    algo = PPO(env_spec=env.spec,
            policy=policy,
            value_function=value_function,
            sampler=sampler,
            discount=0.99,
            gae_lambda=0.95,
            lr_clip_range=0.1,
            center_adv=False)

    for g in algo._policy_optimizer._optimizer.param_groups:
        g['lr'] = args.lr


    # if torch.cuda.is_available():
    #     set_gpu_mode(True)
    # else:
    #     set_gpu_mode(False)
    # algo.to()
    trainer.setup(algo, env)
    trainer.train(n_epochs=200, batch_size=5000, plot=isRendering)   
    return

import os
import argparse
import time
import wandb

def main(args):
    wandb_tensorboard_patched = False

            
    #build experiment name
    timestr = time.strftime("%Y%m%d-%H%M%S")
    log_dir = os.path.join('garage_data/local/experiment', '{0}_{1}_{2}_seed{3}_{4}'.format(args.rl_algo, 'DualArm', args.rl_policy, args.seed, timestr))
    
    if args.use_wandb:
        wandb_run = wandb.init(config=args, project='consensus_normflow', name=log_dir, reinit=True, sync_tensorboard=False)
        if not wandb_tensorboard_patched:
            wandb.tensorboard.patch(tensorboardX=True, pytorch=True)
            wandb_tensorboard_patched = True

    ctxt = dict(log_dir=log_dir, snapshot_mode='last', archive_launch_repo=False, use_existing_dir=True)       

    dualfranka_bullet_tests(ctxt, config=args)
    
    if args.use_wandb:
        wandb_run.finish()


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='args', add_help=False)
    parser.add_argument('--rl_policy', type=str, default='NN',
                        choices=['NN', 'CNF-N', 'CNF-TI'],
                        help='Name of policy used to train')
    parser.add_argument('--rl_algo', type=str, default='PPO',
                        choices=['PPO'],
                        help='Name of RL algo from Stable Baselines to train')
    parser.add_argument('--lr', type=float, default=1e-4,
                        help='Learning rate for training')
    parser.add_argument('--seed', type=int, default=1,
                        help='Random seed to train')
    parser.add_argument('--use_wandb', action='store_true',
                        help='Whether to enable logging to wandb.ai')
    args, unknown = parser.parse_known_args()
    main(args)