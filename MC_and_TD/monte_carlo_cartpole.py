import time
import gym
import numpy as np

import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F


GAMMA = 0.99
EPISODES_TO_TRAIN = 4

# if gpu is to be used
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def calc_qvals(rewards):
    res = []
    sum_r = 0.0
    for r in reversed(rewards):
        sum_r *= GAMMA
        sum_r += r
        res.append(sum_r)
    return list(reversed(res))


class Net(nn.Module):
    def __init__(self, input_size, n_actions):
        super(Net, self).__init__()

        self.net = nn.Sequential(
            nn.Linear(input_size, 128),
            nn.ReLU(),
            nn.Linear(128, n_actions)
        )

    def forward(self, x):
        return self.net(x)


class Agent:
    def __init__(self, env):
        self.net = Net(env.observation_space.shape[0], env.action_space.n).to(device)
        self.action_space = range(env.action_space.n)
        self.optimizer = optim.Adam(self.net.parameters(), lr=0.01)

    def select_action(self, state):
        state = torch.tensor(state, dtype=torch.float).to(device)
        state.unsqueeze(0)
        action_dis = F.softmax(self.net(state), dim=0)
        action = np.random.choice(self.action_space, p=action_dis.cpu().detach().numpy())
        return action

    def take_action(self, env, action):
        state, r, done, _ = env.step(action)
        return state, r, done

    def try_episode_datagen(self, env):
        state = env.reset()
        while True:
            action = self.select_action(state)

            next_state, reward, done =self.take_action(env, action)
            if done:
                next_state = None
                yield state, next_state, done, reward, action
                state = env.reset()
                continue
            yield state, next_state, done, reward, action
            state = next_state

    def training(self, data, rewards, actions):
        self.optimizer.zero_grad()
        inputs = torch.tensor(data, dtype=torch.float).to(device)
        logits = self.net(inputs)
        action_dis = F.log_softmax(logits, dim=1)

        rewards = torch.tensor(rewards, dtype=torch.float).to(device)
        actions = torch.tensor(actions, dtype=torch.long)

        action_v = rewards * action_dis[range(len(rewards)), actions]

        loss = -action_v.mean()

        loss.backward()
        self.optimizer.step()

    def play_games(self, env):  # test environment
        state = env.reset()
        while True:
            action = self.select_action(state)
            state, r, done = self.take_action(env, action)
            env.render()
            time.sleep(0.1)
            if done:
                break
        env.close()
        return None


if __name__ == '__main__':
    env = gym.make("CartPole-v0")
    agent = Agent(env)

    step = 0
    episode = 0

    total_rewards = []
    cur_rewards = []

    batch_states, batch_actions, batch_qvals = [], [], []
    data_source = agent.try_episode_datagen(env)  # Is a generator

    for step_idx, data in enumerate(data_source):
        state, last_state, done, reward, action = data

        batch_states.append(state)
        batch_actions.append(int(action))
        cur_rewards.append(reward)

        if last_state is None:
            batch_qvals.extend(calc_qvals(cur_rewards))
            total_rewards.append(np.sum(cur_rewards))
            cur_rewards.clear()
            episode += 1
        if episode < EPISODES_TO_TRAIN:
            continue

        states_v = torch.FloatTensor(batch_states)
        batch_actions_t = torch.LongTensor(batch_actions)
        batch_qvals_v = torch.FloatTensor(batch_qvals)

        agent.training(states_v, batch_qvals_v, batch_actions_t)

        step += 1
        print(len(batch_states))
        if float(np.mean(total_rewards[-100:])) > 195:  # 195 rewards is a baseline
            print('finished')
            break
        else:
            print('step : {} , Recent_100_episodes_mean: {}'.format(step, float(np.mean(total_rewards[-100:]))))
        episode = 0
        batch_states.clear()
        batch_actions.clear()
        batch_qvals.clear()

    agent.play_games(env)
