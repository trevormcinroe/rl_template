import torch
from torch import nn, FloatTensor, Tensor, LongTensor
import torch.nn.functional as F
import torch.distributions as td
import numpy as np
from networks.utils.activations import get_activation
from networks.distributions import SquashedNormal
from typing import Union, List, Tuple, Optional
import wandb


class Actor(nn.Module):
    def __init__(self, obs_dim: int, action_dim: int, hidden_dims: List[int], activation: str, norm: bool,
                 logvar_min: int, logvar_max: int) -> None:
        super().__init__()
        self.logvar_min = logvar_min
        self.logvar_max = logvar_max

        self.layers = nn.ModuleList([nn.Linear(obs_dim, hidden_dims[0])])

        for i in range(1, len(hidden_dims)):
            if norm:
                self.layers.append(nn.LayerNorm(hidden_dims[i - 1]))

            self.layers.append(get_activation(activation))
            self.layers.append(nn.Linear(hidden_dims[i - 1], hidden_dims[i]))

        self.layers.append(get_activation(activation))
        self.layers.append(nn.Linear(hidden_dims[-1], action_dim * 2))

    def forward(self, x: FloatTensor) -> td.Distribution:
        for layer in self.layers:
            x = layer(x)

        mu, logvar = x.chunk(2, dim=-1)
        logvar = torch.tanh(logvar)
        logvar = self.logvar_min + 0.5 * (self.logvar_max - self.logvar_min) * (logvar + 1)
        var = logvar.exp()
        dist = td.Independent(SquashedNormal(mu, var), 0)
        return dist


class Critic(nn.Module):
    def __init__(self, obs_dim: int, action_dim: int, hidden_dims: List[int], activation: str, norm: bool) -> None:
        super().__init__()

        self.layers = nn.ModuleList([nn.Linear(obs_dim + action_dim, hidden_dims[0])])

        for i in range(1, len(hidden_dims)):
            if norm:
                self.layers.append(nn.LayerNorm(hidden_dims[i - 1]))

            self.layers.append(get_activation(activation))
            self.layers.append(nn.Linear(hidden_dims[i - 1], hidden_dims[i]))

        if norm:
            self.layers.append(nn.LayerNorm(hidden_dims[i - 1]))
        self.layers.append(get_activation(activation))
        self.layers.append(nn.Linear(hidden_dims[-1], 1))

    def forward(self, x: FloatTensor) -> FloatTensor:
        for layer in self.layers:
            x = layer(x)
        return x


class DoubleQ(nn.Module):
    def __init__(self, obs_dim: int, action_dim: int, hidden_dims: List[int], activation: str, norm: bool) -> None:
        super().__init__()

        self.q1 = Critic(obs_dim, action_dim, hidden_dims, activation, norm)
        self.q2 = Critic(obs_dim, action_dim, hidden_dims, activation, norm)

    def forward(self, x: FloatTensor) -> Tuple[FloatTensor, FloatTensor]:
        return self.q1(x), self.q2(x)


class SAC:
    def __init__(self, obs_dim: int, action_dim: int, hidden_dims: List[int], activation: str, norm: bool,
                 logvar_min: int, logvar_max: int, actor_lr: float, critic_lr: float, alpha_lr: float,
                 init_temperature: float, gamma: float, tau: float, action_range: List[float], batch_size: int,
                 actor_update_freq: int, critic_target_update_freq: int, logger: wandb, device: str,
                 grad_clip: Union[int, bool]) -> None:
        self.actor = Actor(obs_dim, action_dim, hidden_dims, activation, False, logvar_min, logvar_max).to(device)

        self.critic = DoubleQ(obs_dim, action_dim, hidden_dims, activation, norm).to(device)
        self.target_critic = DoubleQ(obs_dim, action_dim, hidden_dims, activation, norm).to(device)

        self.target_critic.load_state_dict(self.critic.state_dict())

        self.log_alpha = torch.tensor(np.log(init_temperature)).to(device)
        self.log_alpha.requires_grad = True

        self.actor_optim = torch.optim.Adam(self.actor.parameters(), lr=actor_lr)
        self.critic_optim = torch.optim.Adam(self.critic.parameters(), lr=critic_lr)

        self.log_alpha_optimizer = torch.optim.Adam([self.log_alpha], lr=alpha_lr)
        self.target_entropy = -action_dim

        self.gamma = gamma
        self.tau = tau
        self.action_range = action_range
        self.batch_size = batch_size
        self.actor_update_freq = actor_update_freq
        self.critic_target_update_freq = critic_target_update_freq
        self.logger = logger
        self.device = device
        self.grad_clip = grad_clip

    @property
    def alpha(self) -> FloatTensor:
        return self.log_alpha.exp()

    def act(
            self, obs: Union[np.array, FloatTensor], sample: bool = False, return_dist: bool = False
    ) -> Tuple[np.array, Optional[td.Distribution]]:
        # if not isinstance(obs, torch.Tensor):
        obs = torch.FloatTensor(obs).to(self.device)
        obs = obs.unsqueeze(0)
        dist = self.actor(obs)
        action = dist.sample() if sample else dist.mean
        action = action.clamp(*self.action_range)

        if not return_dist:
            return action[0].detach().cpu().numpy()
        else:
            return action[0].detach().cpu().numpy(), dist

    def update_critic(self, obs: FloatTensor, action: FloatTensor, reward: FloatTensor, next_obs: FloatTensor,
                      not_done: LongTensor) -> None:
        # self.logger.log({'batch_reward': reward.mean().item()})

        with torch.no_grad():
            dist = self.actor(next_obs)
            next_action = dist.rsample()
            log_prob = dist.log_prob(next_action).sum(-1, keepdim=True)
            target_q1, target_q2 = self.target_critic(torch.cat([next_obs, next_action], dim=-1))
            target_V = torch.min(target_q1, target_q2) - self.alpha.detach() * log_prob
            target_Q = reward + (not_done * self.gamma * target_V)

        q1, q2 = self.critic(torch.cat([obs, action], dim=-1))
        self.logger.log({'batch_q': q1.mean().item()})
        critic_loss = F.mse_loss(q1, target_Q, reduction='none') + F.mse_loss(q2, target_Q, reduction='none')

        # self.logger.log({
        #     'id_td_error': critic_loss[:int(q1.shape[0] * 0.5)].mean().item(),
        #     'model_td_error': critic_loss[int(q1.shape[0] * 0.5):].mean().item()
        # })

        critic_loss = critic_loss.mean()

        self.critic_optim.zero_grad()
        critic_loss.backward()
        self.logger.log({'critic_loss': critic_loss.item()})

        # Clip dat grad
        torch.nn.utils.clip_grad_norm_(self.critic.parameters(), self.grad_clip)

        # Grabbing the norm of the parameters
        parameters = [p for p in self.critic.parameters() if p.grad is not None and p.requires_grad]
        device = parameters[0].grad.device
        total_norm = torch.norm(torch.stack([torch.norm(p.grad.detach()).to(device) for p in parameters]),
                                2.0).item()
        self.logger.log({'critic_grad_norm': total_norm})

        self.critic_optim.step()

    def update_actor_and_alpha(self, obs: FloatTensor) -> None:
        dist = self.actor(obs)
        action = dist.rsample()
        log_prob = dist.log_prob(action).sum(-1, keepdim=True)

        q1, q2 = self.critic(torch.cat([obs, action], dim=-1))
        actor_Q = torch.min(q1, q2)

        actor_loss = (self.alpha.detach() * log_prob - actor_Q).mean()

        self.actor_optim.zero_grad()
        actor_loss.backward()
        self.logger.log({'actor_loss': actor_loss.item()})

        # Clip dat grad
        torch.nn.utils.clip_grad_norm_(self.actor.parameters(), self.grad_clip)

        # Grabbing the norm of the parameters
        parameters = [p for p in self.actor.parameters() if p.grad is not None and p.requires_grad]
        device = parameters[0].grad.device
        total_norm = torch.norm(torch.stack([torch.norm(p.grad.detach()).to(device) for p in parameters]),
                                2.0).item()
        self.logger.log({'actor_grad_norm': total_norm})

        self.actor_optim.step()

        self.log_alpha_optimizer.zero_grad()
        alpha_loss = (self.alpha * (-log_prob - self.target_entropy).detach()).mean()
        alpha_loss.backward()
        # self.logger.log({'alpha_loss': alpha_loss.item()})
        self.log_alpha_optimizer.step()

        # self.logger.log({'alpha': self.alpha.item()})

    def update(self, batch: Tuple[FloatTensor, FloatTensor, FloatTensor, FloatTensor, LongTensor], step: int) -> None:
        obs, action, next_obs, reward, not_done = batch

        self.update_critic(obs, action, reward, next_obs, not_done)

        if step % self.actor_update_freq == 0:
            self.update_actor_and_alpha(obs)

        if step % self.critic_target_update_freq == 0:
            for param, target_param in zip(self.critic.parameters(), self.target_critic.parameters()):
                target_param.data.copy_(self.tau * param.data + (1.0 - self.tau) * target_param.data)

    def save(self, filename: str) -> None:
        """

        Args:
            filename:

        Returns:

        """
        torch.save(self.critic.state_dict(), f'{filename}_critic.pt')
        torch.save(self.target_critic.state_dict(), f'{filename}_target_critic.pt')
        torch.save(self.actor.state_dict(), f'{filename}_actor.pt')
        torch.save(self.log_alpha, f'{filename}_alpha.pt')
        torch.save(self.actor_optim.state_dict(), f'{filename}_actor_optim.pt')
        torch.save(self.critic_optim.state_dict(), f'{filename}_critic_optim.pt')
        torch.save(self.log_alpha_optimizer.state_dict(), f'{filename}_alpha_optim.pt')

    def load(self, filename: str) -> None:
        """

        Returns:

        """
        self.critic.load_state_dict(torch.load(f'{filename}_critic.pt'))
        self.target_critic.load_state_dict(torch.load(f'{filename}_target_critic.pt'))
        self.actor.load_state_dict(torch.load(f'{filename}_actor.pt'))
        self.log_alpha = torch.load(f'{filename}_alpha.pt')
        # self.actor_optim.load_state_dict(torch.load(f'{filename}_actor_optim.pt'))
        # self.critic_optim.load_state_dict(torch.load(f'{filename}_critic_optim.pt'))
        # self.log_alpha_optimizer.load_state_dict(torch.load(f'{filename}_alpha_optim.pt'))
