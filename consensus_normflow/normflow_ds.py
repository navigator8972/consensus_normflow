import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np

import flow

class QuadraticPotentialFunction(nn.Module):

    def __init__(self, feature=None):
        super().__init__()

        self.feature = feature
    
    def forward(self, x, x_star):
        if self.feature is not None:
            x = self.feature(x)
            x_star = self.feature(x_star)
        
        return (x - x_star).pow(2).sum(1)
    
    def forward_grad_feature(self, x, x_star):
        if self.feature is not None:
            x = self.feature(x)
            x_star = self.feature(x_star)
        
        return (x - x_star)*2

#https://gist.github.com/apaszke/226abdf867c4e9d6698bd198f3b45fb7
def jacobian(y, x, create_graph=False):                                                               
    jac = []                                                                                          
    flat_y = y.reshape(-1)                                                                            
    grad_y = torch.zeros_like(flat_y)                                                                 
    for i in range(len(flat_y)):                                                                      
        grad_y[i] = 1.                                                                                
        grad_x, = torch.autograd.grad(flat_y, x, grad_y, retain_graph=True, create_graph=create_graph)
        jac.append(grad_x.reshape(x.shape))                                                           
        grad_y[i] = 0.                                                                                
    return torch.stack(jac).reshape(y.shape + x.shape)


#batch version jacobian
#https://github.com/pytorch/pytorch/issues/23475
def jacobian_in_batch(y, x):
    '''
    Compute the Jacobian matrix in batch form.
    Return (B, D_y, D_x)
    '''
    batch = y.shape[0]
    single_y_size = np.prod(y.shape[1:])
    y = y.view(batch, -1)
    vector = torch.ones(batch).to(y)

    # Compute Jacobian row by row.
    # dy_i / dx -> dy / dx
    # (B, D) -> (B, 1, D) -> (B, D, D)
    jac = [torch.autograd.grad(y[:, i], x, 
                               grad_outputs=vector, 
                               retain_graph=True,
                               create_graph=True)[0].view(batch, -1)
                for i in range(single_y_size)]
    jac = torch.stack(jac, dim=1)
    
    return jac                                                
                                                                                                      

class NormalizingFlowDynamicalSystem(nn.Module):
    
    def __init__(self, dim=2, n_flows=3, hidden_dim=8, K=None, D=None, device='cpu'):
        super().__init__()
        self.flows = [flow.RealNVP(dim, hidden_dim=hidden_dim, base_network=flow.FCNN) for i in range(n_flows)]
        self.phi = nn.Sequential(*self.flows)
        self.potential = QuadraticPotentialFunction(feature=self.phi)
        self.dim = dim
        self.device = device

        if device == 'cpu':
            self.phi.cpu()
            self.potential.cpu()
        else:
            self.phi.cuda()
            self.potential.cuda()
        
        if K is None:
            self.K = torch.eye(self.dim, device=device)
        elif isinstance(K, (int, float)):
            self.K = torch.eye(self.dim, device=device) * K
        else:
            self.K = K

        if D is None:
            self.D = torch.eye(self.dim, device=device)
        elif isinstance(D, (int, float)):
            self.D = torch.eye(self.dim, device=device) * D
        else:
            self.D = D
    
    def forward(self, x, x_star, inv=False):
        '''
        x:          state pos
        x_star:     equilibrium pos
        inv:        use inverse of Jacobian or not. works as change of coordinate if True
        '''
        y = self.phi(x)
        phi_jac = jacobian_in_batch(y, x)
        potential_grad = -self.potential.forward_grad_feature(x, x_star).unsqueeze(-1)
        if inv:
            return torch.solve(potential_grad, phi_jac)[0].squeeze(-1)
        else:
            return torch.bmm(phi_jac.transpose(1, 2), potential_grad).squeeze(-1)
    
    def forward_with_damping(self, x, x_star, x_dot, inv=False, jac_damping=True):
        '''
        same as forward
        D:              damping matrix
        x_dot:          time derivative of x
        jac_damping:    apply jacobian to damping matrix?
        '''
        y = self.phi(x)
        # print(y.requires_grad, x.requires_grad)
        phi_jac = jacobian_in_batch(y, x)
        potential_grad = -self.potential.forward_grad_feature(x, x_star).unsqueeze(-1)
        potential_grad_K = torch.matmul(self.K,potential_grad)

        if jac_damping:
            damping_acc = -torch.bmm(
                torch.bmm(
                    torch.bmm(phi_jac.transpose(1, 2), self.D.expand(x_dot.shape[0], -1, -1)), 
                    phi_jac), 
                x_dot.unsqueeze(-1)).squeeze(-1)
        else:
            damping_acc = -torch.bmm(self.D.expand(x_dot.shape[0], -1, -1), x_dot.unsqueeze(-1)).squeeze(-1)

        if inv:
            return torch.solve(potential_grad_K, phi_jac)[0].squeeze(-1) + damping_acc
        else: 
            return torch.bmm(phi_jac.transpose(1, 2), potential_grad_K).squeeze(-1) + damping_acc
    
    def potential_with_damping(self, x, x_star, x_dot, M):
        #M: batched version of mass, could be spd depending on x
        x_potential = 0.5*self.potential.forward(x, x_star)
        x_dot_potential = 0.5*torch.bmm(torch.bmm(x_dot.unsqueeze(1), M), x_dot.unsqueeze(-1)).squeeze()
        # print(x_potential.shape, x_dot_potential.shape)
        return x_potential + x_dot_potential

    def null_space_proj(self, x, plane_norm):
        '''
        project x to the plane defined by plane_norm, batch-wise processing
        x:          batch of vectors with dim length
        plane_norm: batch of norms
        '''
        norm_dir = F.normalize(plane_norm, dim=1)
        proj_len = torch.bmm(x.view(x.shape[0], 1, x.shape[1]), norm_dir.view(norm_dir.shape[0], norm_dir.shape[1], 1)).squeeze(-1)
        return x - proj_len*norm_dir
    
    def null_space(self, x_dot):
        '''
        get nullspace of given batch of x_dot such that
        torch.bmm(nullspace, x_dot) == 0

        return (batch_size, x_dot_dim, x_dot_dim)
        '''
        #note we can avoid matrix inversion because x_dot are vectors so we actually just need the inverse of norm
        norm_square_inv = 1./torch.sum(x_dot**2, dim=1, keepdim=True).clamp(min=1e-6)
        # print('x_dot', x_dot)
        I = torch.eye(x_dot.shape[1], device=self.device).unsqueeze(0).repeat(x_dot.shape[0], 1, 1)
        return I - norm_square_inv.unsqueeze(-1)*torch.bmm(x_dot.unsqueeze(-1), x_dot.unsqueeze(1))

    def init_phi(self):

        def param_init(m):
            if isinstance(m, nn.Conv2d) or isinstance(m, nn.Linear):
                torch.nn.init.xavier_uniform_(m.weight)
                if m.bias is not None:
                    torch.nn.init.zeros_(m.bias)
        
        self.phi.apply(param_init)
        return


class DuoNormalizingFlowDynamicalSystem(nn.Module):
    def __init__(self, dim=2, n_flows=3, hidden_dim=8, K=None, D=None, device='cpu'):
        super().__init__()
        self.flows = [flow.RealNVP(dim, hidden_dim=hidden_dim, base_network=flow.FCNN) for i in range(n_flows)]
        self.phi = nn.Sequential(*self.flows)
        # self.phi = nn.Identity()
        self.potential = QuadraticPotentialFunction(feature=self.phi)

        if device == 'cpu':
            self.phi.cpu()
            self.potential.cpu()
        else:
            self.phi.cuda()
            self.potential.cuda()
        

        #prepare two normalizingflow ds for two agents
        self.normflow_ds = [NormalizingFlowDynamicalSystem(dim, n_flows, hidden_dim, K, D, device) for _ in range(2)]
        #override modules
        for ds in self.normflow_ds:
            ds.phi = self.phi
            ds.potential = self.potential

        # assert(list(self.normflow_ds[0].phi.parameters()) == list(self.normflow_ds[1].phi.parameters()))
        
    
    def forward(self, x, x_dot):
        """
        x, x_dot are lists of batched tensors
        x = [batched_pos_1, batched_pos_2]
        x_dot = [batched_vel_1, batched_vel_2]
        """
        u_1 = self.normflow_ds[0].forward_with_damping(x[0], x[1], x_dot[0])
        u_2 = self.normflow_ds[1].forward_with_damping(x[1], x[0], x_dot[1])
        return [u_1, u_2]

class ConsensusNormalizingFlowDynamics(nn.Module):
    def __init__(self, n_dim=2, n_agents=2, n_flows=3, hidden_dim=8, L=None, K=None, D=None):
        super().__init__()
        flows = [flow.RealNVP(n_dim, hidden_dim=hidden_dim, base_network=flow.FCNN) for i in range(n_flows)]
        self.phi = nn.Sequential(*flows)

        if L is None:
            #fully connected Laplacian by default
            L = -torch.ones(n_agents, n_agents)
            L += torch.eye(n_agents)*2

        laplacian = torch.kron(L, torch.eye(n_dim))
        self.register_buffer('laplacian', laplacian)

        #K and D must be a list of positive numbers, fix them not for learning for now
        if K is None:
            K = torch.ones(n_dim)
        elif isinstance(K, (int, float)):
            K = torch.ones(n_dim) * K

        self.register_buffer('K', torch.diag(K).unsqueeze(0).repeat(n_agents, 1, 1))

        if D is None:
            D = torch.ones(n_dim)
        elif isinstance(D, (int, float)):
            D = torch.ones(n_dim) * D

        self.register_buffer('D', torch.diag(K).unsqueeze(0).repeat(n_agents, 1, 1))

        self.n_dim = n_dim
        self.n_agents = n_agents

    def forward(self, x):
        """
        x are batched tensors
        x = [batch_dims, n_agents*n_dim]
        """

        return NotImplementedError
    
    def forward_2ndorder(self, x, x_dot, jac_damping=True):
        """
        x, x_dot are batched tensors
        x = [batch_dims, n_agents*n_dim]
        x_dot = [batch_dims, n_agents*n_dim]

        u = J^T@L@K@phi(L@x) - J^TDJx_dot
        """
        batch_dims = x.size()[:-1]
        Lx = torch.matmul(self.laplacian, x.unsqueeze(-1)).squeeze(-1)
        Lx_agent = Lx.view(*batch_dims, self.n_agents, self.n_dim)
        phi = self.phi(Lx_agent.view(-1, self.n_dim))  #(batch_dims * n_agents, n_dim)
        phi_agent = phi.view(*batch_dims, self.n_agents, self.n_dim)
        # print(x.shape, Lx_agent.shape, phi_agent.shape)
        # print(phi_agent.view(-1, self.n_dim).shape, Lx_agent.view(-1, self.n_dim).shape)
        #agent-wise jacobian (batch_dims, n_agents, n_dim, n_dim)
        phi_jac_agent = jacobian_in_batch(phi, Lx).view(*batch_dims, self.n_agents, self.n_dim, self.n_dim)

        if jac_damping:
            damping = phi_jac_agent.transpose(-2, -1) @ self.D @ phi_jac_agent
        damping_u = (damping @ (x_dot.view(*batch_dims, self.n_agents, self.n_dim).unsqueeze(-1))).squeeze(-1)

        #L@K@phi(Lx) and convert it to agent-wise tensor (batch_dims, n_agents, n_dim)
        LKphi_agent = (self.laplacian @ (self.K @ phi_agent).view_as(x).unsqueeze(-1)).squeeze(-1).view(*batch_dims, self.n_agents, self.n_dim)
        u =  (phi_jac_agent.transpose(-2, -1) @ LKphi_agent.unsqueeze(-1)).squeeze(-1) - damping_u

        return u.view(*batch_dims, self.n_agents*self.n_dim)