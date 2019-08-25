import numpy as np
import matplotlib.pyplot as plt


class Game:
    def __init__(self, true_reward, arm_num=10, epsilon=0.1, turns = 2000):
        self.turns = turns
        self.k = arm_num
        self.eps = epsilon

        self.true_reward = true_reward  # np.random.randn(self.k)

        self.q_estimation = np.zeros(self.k)
        self.action_numbers = np.zeros(self.k)
        self.total_reward = np.zeros(self.k)
        self.avg_reward = [0]

    def reset(self, eps_new):
        self.q_estimation = np.zeros(self.k)
        self.action_numbers = np.zeros(self.k)
        self.total_reward = np.zeros(self.k)
        self.avg_reward = [0]

        self.eps = eps_new

    def act(self):
        if np.random.rand() < self.eps:
            action = np.random.randint(self.k)
        else:
            action = np.argmax(self.q_estimation)

        return action


    def get_reward(self, action):
        return self.true_reward[action] + np.random.randn()   # gaussian_mean + variance=1

    def simulation(self):
        turn = 0

        while turn < self.turns:
            action = self.act()
            reward = self.get_reward(action)
            self.action_numbers[action] += 1
            self.total_reward[action] += reward
            self.q_estimation[action] = 1.0 * self.total_reward[action] / self.action_numbers[action]
            turn += 1

            self.avg_reward.append(sum(self.total_reward) / turn)


if __name__ == '__main__':
    arms = 10
    true_reward = np.random.randn(arms)
    game = Game(true_reward, arm_num=arms, epsilon=0.1)
    game.simulation()

    plt.figure(1)
    plt.plot(game.avg_reward, color='#FF0000' )  # red
    game.reset(0.0)
    game.simulation()
    plt.plot(game.avg_reward, color='#00FF00')   # green
    game.reset(0.02)
    game.simulation()
    plt.plot(game.avg_reward, color='#0000FF')   # blue
    plt.show()



