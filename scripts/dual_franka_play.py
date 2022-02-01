"""
a test of dual arm franka env
"""
import argparse
import numpy as np
import torch

from consensus_normflow.envs import DualFrankaPandaBulletEnv, DualFrankaPandaTaskTranslationBulletEnv, DualFrankaPandaObjectsBulletEnv
from consensus_normflow.consensus_normflow.normflow_ds import ConsensusNormalizingFlowDynamics, ConsensusDuoNormalizingFlowDynamics

from consensus_normflow.consensus_normflow.consensus_policy_sb3 import ConsensusNormalizingflowACPolicy

def main(args):
    # nfds = ConsensusNormalizingFlowDynamics(n_dim=3, n_agents=2, n_flows=1, K=4)
    nfds = ConsensusDuoNormalizingFlowDynamics(n_dim=3, n_flows=1, K=20, D=0.1)
    
    env = DualFrankaPandaObjectsBulletEnv(args)
    env.reset()

    policy = ConsensusNormalizingflowACPolicy(observation_space=env.observation_space, action_space=env.action_space)

    r = 0
    while env.t < 5000:
        obs = env.get_obs() #[agent1_pos, agent1_vel, agent2_pos, agent2_vel]
        # pos = torch.from_numpy(np.concatenate((obs[:3], obs[6:9]))).unsqueeze(0).float()
        # pos.requires_grad = True
        # vel = torch.from_numpy(np.concatenate((obs[3:6], obs[9:]))).unsqueeze(0).float()
        # a = nfds.forward_2ndorder(pos, vel).detach().squeeze(0).numpy()

        obs_np = np.expand_dims(obs, axis=0).astype(np.float32)
        action, state_ = policy.predict(obs_np, deterministic=True)
        a = action[0]   #only one batch size

        obs, reward, done, info = env.step(a)
        r += reward
    print('Return:', r)
    return


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='args', add_help=True)
    parser.add_argument('--viz', action='store_true',
                    help='Whether to visualize')
    parser.add_argument('--force_ctrl', action='store_true',
                    help='Whether to use force control as action')
    parser.add_argument('--debug', action='store_true',
                    help='Whether to use debug visualization')
    args, _ = parser.parse_known_args()
    main(args)
