import torch
from torch import nn


class ActorCritic(nn.Module):
  def __init__(self, obs_dim, act_dim, hidden=64):
    super().__init__()
    self.shared = nn.Sequential(
      nn.Linear(obs_dim, hidden),
      nn.Tanh(),
      nn.Linear(hidden, hidden),
      nn.Tanh(),
    )
    self.mu = nn.Linear(hidden, act_dim)  # Actor: 行動の平均
    self.v = nn.Linear(hidden, 1)  # Critic: 価値
    self.log_std = nn.Parameter(torch.zeros(act_dim))  # 観測に依存しない

  def forward(self, obs):
    h = self.shared(obs)
    return self.mu(h), self.v(h).squeeze(-1), self.log_std

  def act(self, obs):
    mu, v, log_std = self(obs)
    std = log_std.exp()
    dist = torch.distributions.Normal(mu, std)
    action = dist.sample()
    log_prob = dist.log_prob(action).sum(-1)
    return action, log_prob, v


torch.manual_seed(0)
net = ActorCritic(3, 1)
obs = torch.randn(4, 3)
action, log_prob, v = net.act(obs)
print("std     :", net.log_std.exp().detach().numpy())
print("action  :", action.detach().numpy().ravel().round(4))
print("log_prob:", log_prob.detach().numpy().round(4))


# net = ActorCritic(obs_dim=3, act_dim=1)
# obs = torch.randn(4, 3)
# mu, v, log_std = net(obs)
# print("mu:", tuple(mu.shape), " v:", tuple(v.shape), " log_std:", tuple(log_std.shape))
# print("パラメータ数:", sum(p.numel() for p in net.parameters()))
