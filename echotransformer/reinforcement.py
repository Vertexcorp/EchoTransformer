import torch
import torch.nn as nn
import torch.nn.functional as F

class ReinforcementModule(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.policy_net = nn.Sequential(
            nn.Linear(config.d_model, config.d_model // 2),
            nn.ReLU(),
            nn.Linear(config.d_model // 2, config.vocab_size)
        )
        self.value_net = nn.Sequential(
            nn.Linear(config.d_model, config.d_model // 2),
            nn.ReLU(),
            nn.Linear(config.d_model // 2, 1)
        )

    def forward(self, state):
        action_probs = F.softmax(self.policy_net(state), dim=-1)
        state_value = self.value_net(state)
        return action_probs, state_value

    def compute_loss(self, state, action, reward, next_state, done):
        action_probs, state_value = self.forward(state)
        _, next_state_value = self.forward(next_state)

        td_target = reward + (1 - done) * self.config.gamma * next_state_value
        td_error = td_target - state_value

        actor_loss = -torch.log(action_probs.gather(1, action)) * td_error.detach()
        critic_loss = F.mse_loss(state_value, td_target.detach())

        return actor_loss + critic_loss