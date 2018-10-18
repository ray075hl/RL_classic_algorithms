# -*- coding: utf-8 -*-
"""
Data structures in this example are as follows:

(1) Reward table: A dictionary with the composite key "source state" + "action"
    + "target state", The value is obtained from the immediate reward.

(2) Transitions table: A dictionary keeping counters of the experienced transitions.
    The key is the composite "state" + "action" and the value is another dictionary
    that maps the target state into a count of times that we've seen it. For example,
    if in state 0 we execute action 1 ten times, after three times it leads us to
    state 4 and after seven times to state 5, Entry with the key (0,1) in this table
    will be a dict {4: 3, 5: 7}. We use this table to estimate the probabilities of
    our transitions.

(3) Value table: A dictionary that maps a state into the calculated value of this state.


The overall logic of our code is simple: in the loop, we play 100 random steps from
the environment, populating the reward and transition tables. After those 100 steps,
we perform a value iteration loop over all states, updating our value table. Then
we play several full episodes to check our improvements using the updated value table.
If the average reward for those test episodes is above the 0.8 boundary, then we stop
training. During test episodes, we also update our reward and transition tables to
use all data from the environment.
"""


import gym
import collections
from tensorboardX import SummaryWriter

ENV_NAME = "FrozenLake-v0"
GAMMA = 0.9
TEST_EPISODES = 20


class Agent:
    def __init__(self):
        self.env = gym.make(ENV_NAME)
        self.state = self.env.reset()
        self.rewards = collections.defaultdict(float)
        self.transits = collections.defaultdict(collections.Counter)
        self.values = collections.defaultdict(float)    # what we need to optimize

    def play_n_random_steps(self, count):
        for _ in range(count):
            action = self.env.action_space.sample()
            new_state, reward, is_done, _ = self.env.step(action)
            self.rewards[(self.state, action, new_state)] = reward
            self.transits[(self.state, action)][new_state] += 1
            self.state = self.env.reset() if is_done else new_state
        print('+++: ', self.transits)

    def select_action(self, state):  # greedy policy improve method
        best_action, best_value = None, None
        for action in range(self.env.action_space.n):
            action_value = self.values[(state, action)]
            if best_value is None or best_value < action_value:
                best_value = action_value
                best_action = action

        return best_action

    def play_episode(self, env):
        total_reward = 0.0
        state = env.reset()
        while True:
            action = self.select_action(state)
            new_state, reward, is_done, _ = env.step(action)
            self.rewards[(state, action, new_state)] = reward
            self.transits[(state, action)][new_state] += 1
            total_reward += reward
            if is_done:
                break

            state = new_state
        return total_reward

    def value_iteration(self):
        for state in range(self.env.observation_space.n):
            for action in range(self.env.action_space.n):
                action_value = 0.0
                target_counts = self.transits[(state, action)]
                total = sum(target_counts.values())
                for tgt_state, count in target_counts.items():
                    reward = self.rewards[(state, action, tgt_state)]
                    # select action base on state
                    best_action = self.select_action(tgt_state)
                    action_value += (count / total) * (reward +
                        GAMMA * self.values[(tgt_state, best_action)])

                # bellman update for Q value
                self.values[(state, action)] = action_value




if __name__ == "__main__":

    test_env = gym.make(ENV_NAME)
    agent = Agent()
    writer = SummaryWriter(comment="-v-iteration")

    iter_no = 0
    best_reward = 0.0

    while True:
        iter_no += 1
        agent.play_n_random_steps(100)
        agent.value_iteration()

        reward = 0.0
        for _ in range(TEST_EPISODES):
            reward += agent.play_episode(test_env)

        reward /= TEST_EPISODES
        writer.add_scalar("reward", reward, iter_no)
        if reward > best_reward:
            print("Best reward updated {:3f} -> {:3f}".format(best_reward, reward))
            best_reward = reward

        if reward > 0.80:
            print("Solved in {} iterations".format(iter_no))
            break
    writer.close()


