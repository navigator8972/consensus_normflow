
import torch
from torch import nn
from torch.distributions import Normal
from torch.distributions.independent import Independent

from garage.torch.distributions import TanhNormal

from garage.torch.policies.stochastic_policy import StochasticPolicy

from garage.torch import global_device

from dowel import logger, tabular

from .normflow_ds import ConsensusNormalizingFlowDynamics, ConsensusDuoNormalizingFlowDynamics

class GaussianConsensusNormalizingFlowPolicy(StochasticPolicy):
    def __init__(self,
                env_spec,
                nfds=None,
                normal_distribution_cls=Normal,
                init_std=1.0,
                name='GaussianConsensusNormalizingFlowPolicy'):
        super().__init__(env_spec, name)

        self._obs_dim = env_spec.observation_space.flat_dim
        self._action_dim = env_spec.action_space.flat_dim

        if nfds is None:
            nfds = ConsensusDuoNormalizingFlowDynamics(n_dim=self._obs_dim//4, n_flows=2, hidden_dim=16, K=25, D=1)
        
        self.normflow_ds = nfds

        self._normal_distribution_cls=normal_distribution_cls
        #this is probably slightly different from GaussianMLP that has only one param for variance
        init_std_param = torch.Tensor([init_std]).log()
        self._init_std = torch.nn.Parameter(init_std_param)
    
    def forward(self, observations):
        """Compute the action distributions from the observations.
        Args:
            observations (torch.Tensor): Batch of observations on default
                torch device.
        Returns:
            torch.distributions.Distribution: Batch distribution of actions.
            dict[str, torch.Tensor]: Additional agent_info, as torch Tensors
        """
        # logger.log('Obervations shape: {0}, {1}'.format(observations.shape[0], observations.shape[1]))
        #first flatten observations because jacobian_in_batch can only handle one batch dimension
        #should we use view to avoid create new tensors?
        obs_flatten = torch.reshape(observations, (-1, self._obs_dim))
        # logger.log('Obervations flatten shape: {0}, {1}'.format(obs_flatten.shape[0], obs_flatten.shape[1]))
        #might need to figure out a way for more axes

        obs_agent = obs_flatten.view(obs_flatten.shape[0], self.normflow_ds.n_agents, self.normflow_ds.n_dim*2)
        x = obs_agent[:, :, :self.normflow_ds.n_dim].reshape(obs_flatten.shape[0], self.normflow_ds.n_agents*self.normflow_ds.n_dim)
        x_dot = obs_agent[:, :, self.normflow_ds.n_dim:].reshape(obs_flatten.shape[0], self.normflow_ds.n_agents*self.normflow_ds.n_dim)
        x.requires_grad_()

        with torch.enable_grad():
            mean_flatten = self.normflow_ds.forward_2ndorder(x, x_dot)

        #restore mean shape
        broadcast_shape = list(observations.shape[:-1]) + [self._action_dim]
        mean = torch.reshape(mean_flatten, broadcast_shape)

        uncentered_log_std = torch.zeros(*broadcast_shape).to(
                    global_device()) + self._init_std

        std = uncentered_log_std.exp()

        dist = self._normal_distribution_cls(mean, std)

        if not isinstance(dist, TanhNormal):
            # Makes it so that a sample from the distribution is treated as a
            # single sample and not dist.batch_shape samples.
            dist = Independent(dist, 1)

        return (dist, dict(mean=dist.mean, log_std=(dist.variance**.5).log()))