from typing import Any, Callable, Dict, List, Optional, Tuple, Type, Union
from cv2 import mean

import gym
import torch as th
from torch import nn

from stable_baselines3.common.policies import ActorCriticPolicy, BasePolicy

from sb3_contrib.ars.ars import ARSPolicy
from stable_baselines3.common.torch_layers import FlattenExtractor, BaseFeaturesExtractor
from stable_baselines3.common.type_aliases import Schedule

from .normflow_ds import ConsensusNormalizingFlowDynamics, ConsensusDuoNormalizingFlowDynamics

class ConsensusNormalizingflowACPolicy(ActorCriticPolicy):

    def __init__(self,
        #parameters for base policy
        observation_space: gym.spaces.Space,
        action_space: gym.spaces.Space,
        lr_schedule: Schedule = BasePolicy._dummy_schedule,
        net_arch: Optional[List[Union[int, Dict[str, List[int]]]]] = None,
        activation_fn: Type[nn.Module] = nn.Tanh,
        *args,
        **kwargs,
        ) -> None:
        
        super(ConsensusNormalizingflowACPolicy, self).__init__(
              observation_space,
              action_space,
              lr_schedule,
              net_arch,
              activation_fn,
            #   use_sde = False,                              #force some parameters because we don't need the full flexibility here
            #   features_extractor_class=FlattenExtractor,    #should stick to DiagGaussian so the parent class creates log_std for us
            #   features_extractor_kwargs=None,
              # Pass remaining arguments to base class
              *args,
              **kwargs,
        )

        #replace action_net with normflow_ds
        self.normflow_ds = ConsensusDuoNormalizingFlowDynamics(n_dim=self.features_dim//4, n_flows=2, K=30, D=1)
        #also remember to overwrite optimizer to ensure normflow parameters are added
        self.optimizer = self.optimizer_class(self.parameters(), lr=lr_schedule(1), **self.optimizer_kwargs)

    def _get_latent(self, obs: th.Tensor) -> Tuple[th.Tensor, th.Tensor, th.Tensor]:
        """
        Get the latent code (i.e., activations of the last layer of each network)
        for the different networks.
        :param obs: Observation
        :return: Latent codes
            for the actor, the value function and for gSDE function
        """
        # Preprocess the observation if needed
        features = self.extract_features(obs)
        latent_pi, latent_vf = self.mlp_extractor(features)

        # Features for sde
        latent_sde = latent_pi
        if self.sde_features_extractor is not None:
            latent_sde = self.sde_features_extractor(features)

        #surppress latent_pi because we don't allow feature extractor for policy here
        return obs, latent_vf, latent_sde
    
    def _get_action_dist_from_latent(self, obs: th.Tensor, latent_sde: Optional[th.Tensor] = None):
        #latent pi is actually obs here [n_batch, (n_pos_dim+n_vel_dim)*n]    n=2 for now
        #print(obs.shape)
        obs_agent = obs.view(obs.shape[0], self.normflow_ds.n_agents, self.normflow_ds.n_dim*2)
        x = obs_agent[:, :, :self.normflow_ds.n_dim].reshape(obs.shape[0], self.normflow_ds.n_agents*self.normflow_ds.n_dim)
        x_dot = obs_agent[:, :, self.normflow_ds.n_dim:].reshape(obs.shape[0], self.normflow_ds.n_agents*self.normflow_ds.n_dim)
        x.requires_grad_()

        with th.enable_grad():
            mean_actions = self.normflow_ds.forward_2ndorder(x, x_dot)

        #we should only get the case of DiagGaussianDistribution here
        return self.action_dist.proba_distribution(mean_actions, self.log_std)

    def forward(self, obs: th.Tensor, deterministic: bool = False) -> Tuple[th.Tensor, th.Tensor, th.Tensor]:
        """
        Forward pass in all the networks (actor and critic)
        :param obs: Observation
        :param deterministic: Whether to sample or use deterministic actions
        :return: action, value and log probability of the action
        """
        # Preprocess the observation if needed
        features = self.extract_features(obs)
        latent_pi, latent_vf = self.mlp_extractor(features)
        # Evaluate the values for the given observations
        values = self.value_net(latent_vf)
        distribution = self._get_action_dist_from_latent(obs)   #use raw obs as we dont need mlp processed feature for policy
        actions = distribution.get_actions(deterministic=deterministic)
        log_prob = distribution.log_prob(actions)
        return actions, values, log_prob

    def evaluate_actions(self, obs: th.Tensor, actions: th.Tensor) -> Tuple[th.Tensor, th.Tensor, th.Tensor]:
        """
        Evaluate actions according to the current policy,
        given the observations.
        :param obs:
        :param actions:
        :return: estimated value, log likelihood of taking those actions
            and entropy of the action distribution.
        """
        # Preprocess the observation if needed
        features = self.extract_features(obs)
        latent_pi, latent_vf = self.mlp_extractor(features)
        distribution = self._get_action_dist_from_latent(obs)   #use raw obs as we dont need mlp processed feature for policy
        log_prob = distribution.log_prob(actions)
        values = self.value_net(latent_vf)
        return values, log_prob, distribution.entropy()
    
    def get_distribution(self, obs: th.Tensor):
        """
        Get the current policy distribution given the observations.
        :param obs:
        :return: the action distribution.
        """
        # features = self.extract_features(obs)
        # latent_pi = self.mlp_extractor.forward_actor(features)
        return self._get_action_dist_from_latent(obs)   #use raw obs as we dont need mlp processed feature for policy


