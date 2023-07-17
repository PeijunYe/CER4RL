import argparse
import gym
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.distributions.categorical import Categorical


def acc(index):
    acc = 0
    a = torch.rand(1).to('cuda')

    if index == 0:
        acc = a * 0.3

    elif 0 < index <= 9:
        for i in range(10):
            if index == i:
                acc = a * 0.3 + index * 0.3
    elif index > 9:
        for i in range(10):
            if index == i + 10:
                acc = -(a * 0.3 + (index - 10) * 0.3)

    return acc


def steer(index):
    steer = 0
    b = torch.rand(1).to('cuda')
    if index == 0:
        steer = b * 0.03

    elif 0 < index <= 9:
        for i in range(10):
            if index == i:
                steer = b * 0.03 + index * 0.03
    elif index > 9:
        for i in range(10):
            if index == i + 10:
                steer = -(b * 0.03 + (index - 10) * 0.03)

    return steer


class PPOmemory:
    def __init__(self, mini_batch_size):
        self.states = []
        self.actions = []
        self.probs = []
        self.vals = []
        self.rewards = []
        self.dones = []
        self.mini_batch_size = mini_batch_size

    def sample(self):
        n_states = len(self.states)
        batch_start = np.arange(0, n_states, self.mini_batch_size)
        mini_batches = [batch_start[i:len(batch_start)] for i in batch_start]

        return np.array(self.states), np.array(self.actions), np.array(self.probs), \
            np.array(self.vals), np.array(self.rewards), np.array(self.dones), mini_batches

    def push(self, state, action, prob, val, reward, done):
        self.states.append(state)
        self.actions.append(action)
        self.probs.append(prob)
        self.vals.append(val)
        self.rewards.append(reward)
        self.dones.append(done)

    def clear(self):
        self.states = []
        self.actions = []
        self.probs = []
        self.vals = []
        self.rewards = []
        self.dones = []


class Actor(nn.Module):
    def __init__(self, n_states, index_actions, hidden_dim):
        super(Actor, self).__init__()
        self.actor = nn.Sequential(
            nn.Linear(n_states, hidden_dim),
            nn.Tanh(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.Tanh(),
            nn.Linear(hidden_dim, index_actions),
            nn.Softmax(dim=-1))

    def forward(self, state):
        dist = self.actor(state)
        dist = Categorical(dist)
        return dist


class Critic(nn.Module):
    def __init__(self, n_states, hidden_dim):
        super(Critic, self).__init__()
        self.critic = nn.Sequential(
            nn.Linear(n_states, hidden_dim),
            nn.Tanh(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.Tanh(),
            nn.Linear(hidden_dim, 1))

    def forward(self, state):
        value = self.critic(state)
        return value


class Agent:
    def __init__(self, n_states, n_actions, cfg):
        self.gamma = cfg.gamma
        self.n_epochs = cfg.n_epochs
        self.gae_lambda = cfg.gae_lambda
        self.policy_clip = cfg.policy_clip
        self.device = cfg.device
        self.is_load = False
        self.num_inputs = 4
        self.num_actions = 20
        if self.is_load == False:
            self.actor = Actor(n_states, n_actions, cfg.hidden_dim)
            self.actor.to(cfg.device)
            self.critic = Critic(n_states, cfg.hidden_dim)
            self.critic.to(cfg.device)
        else:
            self.actor = Actor(n_states, n_actions, cfg.hidden_dim)
            self.actor.to(cfg.device)
            state_dict_actor = torch.load('actor_params.pth')
            self.actor.load_state_dict(state_dict_actor)

            self.critic = Critic(n_states, cfg.hidden_dim)
            self.critic.to(cfg.device)
            state_dict_critic = torch.load('critic_params.pth')
            self.critic.load_state_dict(state_dict_critic)

        self.actor_optim = optim.Adam(self.actor.parameters(), lr=cfg.actor_lr)
        self.critic_optim = optim.Adam(self.critic.parameters(), lr=cfg.critic_lr)
        self.memory = PPOmemory(cfg.mini_batch_size)

    def choose_action(self, state):
        state = torch.tensor(state, dtype=torch.float).to(self.device)
        dist = self.actor(state)
        value = self.critic(state)

        action_index_acc = dist.sample()
        action_index_steer = dist.sample()
        prob = torch.squeeze(dist.log_prob(action_index_acc)).item()
        action_index = torch.squeeze(action_index_acc).item()
        action_acc = acc(action_index_acc)
        action_steer = steer(action_index_steer)
        action = [action_acc, action_steer]

        value = torch.squeeze(value).item()

        return action, prob, value, action_index

    def learn(self):
        states_arr, actions_arr, old_probs_arr, vals_arr, \
            rewards_arr, dones_arr, mini_batches = self.memory.sample()

        values = vals_arr[:]
        advantage = np.zeros(len(rewards_arr), dtype=np.float32)
        for t in range(len(rewards_arr) - 1):
            discount = 1.0
            a_t = 0
            for k in range(t, len(rewards_arr) - 1):
                a_t += discount * (
                        rewards_arr[k] + self.gamma * values[k + 1] * (1 - int(dones_arr[k])) - values[k])
                discount *= self.gamma * self.gae_lambda
            advantage[t] = a_t
        advantage = torch.tensor(advantage).to(self.device)

        values = torch.tensor(values).to(self.device)
        for batch in mini_batches:
            states = torch.tensor(states_arr[batch], dtype=torch.float).to(self.device)
            old_probs = torch.tensor(old_probs_arr[batch]).to(self.device)
            actions = torch.tensor(actions_arr[batch]).to(self.device)

            dist = self.actor(states)
            critic_value = torch.squeeze(self.critic(states))
            new_probs = dist.log_prob(actions)
            prob_ratio = new_probs.exp() / old_probs.exp()

            weighted_probs = advantage[batch] * prob_ratio
            weighted_clip_probs = torch.clamp(prob_ratio, 1 - self.policy_clip, \
                                              1 + self.policy_clip) * advantage[batch]
            actor_loss = -torch.min(weighted_probs, weighted_clip_probs).mean()
            returns = advantage[batch] + values[batch]
            critic_loss = (returns - critic_value) ** 2
            critic_loss = critic_loss.mean()
            total_loss = actor_loss + 0.5 * critic_loss
            self.actor_optim.zero_grad()
            self.critic_optim.zero_grad()
            total_loss.backward()
            self.actor_optim.step()
            self.critic_optim.step()
        self.memory.clear()


def get_args():
    parser = argparse.ArgumentParser(description="hyper parameters")
    parser.add_argument('--algo_name', default='PPO', type=str, help="name of algorithm")
    parser.add_argument('--env_name', default='CartPole-v1', type=str, help="name of environment")
    parser.add_argument('--train_eps', default=200, type=int, help="episodes of training")
    parser.add_argument('--test_eps', default=20, type=int, help="episodes of testing")
    parser.add_argument('--gamma', default=0.99, type=float, help="discounted factor")
    parser.add_argument('--mini_batch_size', default=1, type=int, help='mini batch size')
    parser.add_argument('--n_epochs', default=4, type=int, help='update number')
    parser.add_argument('--actor_lr', default=0.0003, type=float, help="learning rate of actor net")
    parser.add_argument('--critic_lr', default=0.0003, type=float, help="learning rate of critic net")
    parser.add_argument('--icm_lr', default=0.0003, type=float, help="learning rate of critic net")
    parser.add_argument('--gae_lambda', default=0.98, type=float, help='GAE lambda')
    parser.add_argument('--policy_clip', default=0.2, type=float, help='policy clip')
    parser.add_argument('-batch_size', default=20, type=int, help='batch size')
    parser.add_argument('--hidden_dim', default=256, type=int, help='hidden dim')
    parser.add_argument('--device', default='cuda', type=str, help="cpu or cuda")
    args = parser.parse_args()
    return args


def env_agent_config(cfg, seed=1):
    env = gym.make(cfg.env_name)
    n_states = env.observation_space.shape[0]
    n_actions = env.action_space.n
    agent = Agent(n_states, n_actions, cfg)
    if seed != 0:
        torch.manual_seed(seed)
        env.seed(seed)
        np.random.seed(seed)
    return env, agent
