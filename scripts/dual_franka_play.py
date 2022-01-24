"""
a test of dual arm franka env
"""
import argparse
import numpy as np
import torch

from consensus_normflow.envs import DualFrankaPandaBulletEnv, DualFrankaPandaTaskTranslationBulletEnv, DualFrankaPandaObjectsBulletEnv
from consensus_normflow.consensus_normflow.normflow_ds import ConsensusNormalizingFlowDynamics, ConsensusDuoNormalizingFlowDynamics

def main(args):
    # nfds = ConsensusNormalizingFlowDynamics(n_dim=3, n_agents=2, n_flows=1, K=4)
    nfds = ConsensusDuoNormalizingFlowDynamics(n_dim=3, n_flows=1, K=4)

    env = DualFrankaPandaObjectsBulletEnv(args)
    env.reset()
    while True:
        obs = env.get_obs() #[agent1_pos, agent1_vel, agent2_pos, agent2_vel]
        pos = torch.from_numpy(np.concatenate((obs[:3], obs[6:9]))).unsqueeze(0).float()
        pos.requires_grad = True
        vel = torch.from_numpy(np.concatenate((obs[3:6], obs[9:]))).unsqueeze(0).float()
        a = nfds.forward_2ndorder(pos, vel).detach().squeeze(0).numpy()
        env.step(a)
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
