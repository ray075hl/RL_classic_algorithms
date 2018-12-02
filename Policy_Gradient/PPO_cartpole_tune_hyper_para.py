import gym
import random
import math

import numpy as np

import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torch.distributions import Normal
from multiprocessing_env import SubprocVecEnv


def make_env():
    def _thunk():
        env = gym.make(env_name)
        return env

    return _thunk

use_cuda = torch.cuda.is_available()
device = torch.device("cuda" if use_cuda else "cpu")

env_num = 4
env_name = 'CartPole-v0'

env = gym.make(env_name)

envs = [make_env() for i in range(env_num)]
envs = SubprocVecEnv(envs)

def compute_gae(next_value, rewards, masks, values, gamma=0.99, tau=0.95):
    values = values + [next_value]
    gae = 0
    returns = []
    for step in reversed(range(len(rewards))):
        delta = rewards[step] + gamma * values[step + 1] * masks[step] - values[step]
        gae = delta + gamma * tau * masks[step] * gae
        returns.insert(0, gae + values[step])
    return returns


class ActorCritic(nn.Module):
    def __init__(self, num_inputs, num_outputs, hidden_size):
        super(ActorCritic, self).__init__()

        self.critic = nn.Sequential(
            nn.Linear(num_inputs, hidden_size),
            nn.ReLU(),
            nn.Linear(hidden_size, 1)
        )

        self.actor = nn.Sequential(
            nn.Linear(num_inputs, hidden_size),
            nn.ReLU(),
            nn.Linear(hidden_size, num_outputs)
        )

    def forward(self, x):
        value = self.critic(x)
        dis = self.actor(x)

        return dis, value



num_inputs = envs.observation_space.shape[0]
num_outputs = envs.action_space.n

#Hyper params:
hidden_size      = 32
lr               = 1e-3
num_steps        = 4
mini_batch_size  = 5
ppo_epochs       = 4
threshold_reward = 195

model = ActorCritic(num_inputs, num_outputs, hidden_size).to(device)
optimizer = optim.Adam(model.parameters(), lr=lr)



max_frames = 1500000
frame_idx  = 0
test_rewards = []


state = envs.reset()
early_stop = False


def ppo_iter(mini_batch_size, states, actions, log_probs, returns, advantage):
    batch_size = states.size(0)
    for _ in range(batch_size // mini_batch_size):
        rand_ids = np.random.randint(0, batch_size, mini_batch_size)
        yield states[rand_ids, :], actions[rand_ids, :], log_probs[rand_ids, :], returns[rand_ids, :], advantage[
                                                                                                       rand_ids, :]


def ppo_update(ppo_epochs, mini_batch_size, states, actions, log_probs, returns, advantages, clip_param=0.2):
    for _ in range(ppo_epochs):
        for state, action, old_log_probs, return_, advantage in ppo_iter(mini_batch_size, states, actions, log_probs,
                                                                         returns, advantages):
            dist, value = model(state)
            log_prob_ = F.log_softmax(dist, dim=1)
            entropy = (dist*log_prob_).sum().mean()
            new_log_probs = F.log_softmax(dist, dim=1) #dist.log_prob(action)


            ratio = (new_log_probs[range(mini_batch_size), action.squeeze()] - old_log_probs[range(mini_batch_size), action.squeeze()]).exp()
            surr1 = ratio * advantage
            surr2 = torch.clamp(ratio, 1.0 - clip_param, 1.0 + clip_param) * advantage

            actor_loss =  -torch.min(surr1, surr2).mean()
            critic_loss = (return_ - value).pow(2).mean()

            loss = 1.5*critic_loss + actor_loss + 0.0001 * entropy

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()


def test_env(vis=False):
    state = env.reset()
    if vis: env.render()
    done = False
    total_reward = 0
    while not done:
        state = torch.FloatTensor(state).unsqueeze(0).to(device)
        dist, _ = model(state)
        dist = F.softmax(dist, dim=1)[0]
        # print(dist)
        action = np.random.choice(2, p=dist.cpu().detach().numpy())
        next_state, reward, done, _ = env.step(action)
        state = next_state
        if vis: env.render()
        total_reward += reward
    return total_reward

test_rewards_list = []
while frame_idx < max_frames and not early_stop:

    log_probs = []
    values = []
    states = []
    actions = []
    rewards = []
    masks = []
    entropy = 0

    # first state ,  last state
    for _ in range(num_steps):
        state = torch.FloatTensor(state).to(device)
        logits, value = model(state)

        prob = F.softmax(logits, dim=1)
        action_list = []

        for i in range(len(prob)):
            action = np.random.choice(num_outputs, p=prob.cpu().detach().numpy()[i])
            action_list.append(action)
        # print(action_list)
        next_state, reward, done, _ = envs.step(action_list)

        log_prob = F.log_softmax(logits)

        log_probs.append(log_prob)
        values.append(value)
        rewards.append(torch.FloatTensor(reward).unsqueeze(1).to(device))   # 2D list
        masks.append(torch.FloatTensor(1 - done).unsqueeze(1).to(device))   # 2D list


        states.append(state)
        actions.append(torch.from_numpy(np.asarray(action_list)).unsqueeze(1))
        action_list.clear()
        state = next_state  # -----------------
        frame_idx += 1

    if frame_idx % 200 == 0:
        test_rewards = test_env()
        #print(test_rewards)
        test_rewards_list.append(test_rewards)
        print(sum(test_rewards_list[-50:])/50)
    if 1.0*sum(test_rewards_list[-50:])/50 > 195.0:
        print('solved')
        break

    next_state = torch.FloatTensor(next_state).to(device)
    _, next_value = model(next_state)
    returns = compute_gae(next_value, rewards, masks, values)

    returns = torch.cat(returns).detach()
    log_probs = torch.cat(log_probs).detach()
    values = torch.cat(values).detach()
    states = torch.cat(states)
    actions = torch.cat(actions)
    advantage = returns -values

    ppo_update(ppo_epochs, mini_batch_size, states, \
               actions, log_probs, returns, advantage)


last_reawards = test_env(vis=True)
print('xx: ', last_reawards)
