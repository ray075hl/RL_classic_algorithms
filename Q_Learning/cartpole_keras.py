import gym
import numpy as np
import random
from keras.layers import Dense, InputLayer
from keras.models import Sequential
from collections import deque
from keras.optimizers import Adam, SGD



model = Sequential()
model.add(InputLayer(batch_input_shape=(None, 4)))
model.add(Dense(10, activation='relu'))
model.add(Dense(10, activation='relu'))
model.add(Dense(2, activation='linear'))
model.compile(loss='mse', optimizer=Adam(lr=0.001), metrics=['mae'], )
model.summary()

class CartPole():

    def __init__(self, env):

        self.episode = 10000
        self.gamma = 0.95
        self.decay = 0.995
        self.total_reward = 0
        self.eps = 1.0
        self.min_eps = 0.01
        self.action_num = env.action_space.n
        self.model = model
        self.memory = deque(maxlen=20000)

    def select_action(self, state, time, flag):
        if flag == False:
            if np.random.random() < self.eps:
                return np.random.randint(0, self.action_num)
            else:
                return np.argmax(model.predict(state)[0])
        else:
            return np.argmax(model.predict(state)[0])

    def training(self, current_state, env, time):
        if self.eps > self.min_eps:
            self.eps *= self.decay
        else:
            self.eps = self.min_eps
        current_state = current_state.reshape((-1, 4))
        self.total_reward = 0
        flag_3000 = False
        while True:
            if flag_3000==True:
                env.render()
            # select_action
            action = self.select_action(current_state, time, flag_3000)
            # next_state
            next_state, reward, done, _ = env.step(action)
            next_state = next_state.reshape((-1, 4))
            self.total_reward += reward

            self.memory.append([current_state, next_state, reward, action, done])
            if self.total_reward > 3000:
                flag_3000 = True

            if done:
                flag_3000 = False
                break

            # update state
            current_state = next_state

        # print total reward in this episode
        print('time: {}, Reward: {}, eps: {}'.format(time, self.total_reward, self.eps))
        # replay
        if len(self.memory) >= 128:
            X = []
            Y = []
            batch_data = random.sample(self.memory, 128)
            for state_ , next_state_, reward_, action_, done_ in batch_data:
                if done_:
                    target_q_value = -10.#reward_ # reward_恒等于1.0  done_等于True的情况下，也学要学习， 学习惩罚？
                if not done_:   # 如果 done_ 为 False  reward
                    # Compute target q value
                    target_q_value = self.gamma * np.max(self.model.predict(next_state_)[0]) + reward_
                # Compute predict q value
                action_vec = self.model.predict(state_)
                action_vec[0][action_] = target_q_value
                X.append(state_[0])
                Y.append(action_vec.reshape(1, 2)[0])
            self.model.fit(np.array(X), np.array(Y), epochs=1, verbose=0)
            #self.memory.clear()

if __name__ == '__main__':
    env = gym.make("CartPole-v0").unwrapped
    cartpole = CartPole(env)
    for epi in range(cartpole.episode):
        init_state = env.reset()
        cartpole.training(init_state, env, epi)





