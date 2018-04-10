import numpy as np

if __name__ == '__main__':
    Q = np.zeros((6,6))
    R = -1* np.ones((6,6))
    R[0, 4] = 0
    R[1, 3] = 0;    R[1, 5] = 100
    R[2, 3] = 0
    R[3, 1] = 0;    R[3, 2] = 0;    R[3, 4] = 0
    R[4, 0] = 0;    R[4, 3] = 0;    R[4, 5] = 100
    R[5, 1] = 0;    R[5, 4] = 0;    R[5, 5] = 100

    lr = 0.8
    gamma = 0.95
    eps = 0.5
    decay_factor = 0.999

    episode = 1000
    for i in range(episode):    # 设置为100次尝试
        state = np.random.randint(0, 6)   # 每次尝试的起点为 state 0
        done = False
        eps *= decay_factor
        while not done:
            # 判断在此状态下  有几个可以选择的动作
            possible_state = [i for i, v in enumerate(R[state, :]) if v != -1]

            if np.random.random() < eps:                    # exploration
                new_state = np.random.choice(possible_state)
            else:                                           # exploitation
                temp = [R[state, x] for x in possible_state]
                index = np.argmax(temp)
                print('index', index)
                new_state = possible_state[index]
            # new_state = np.random.choice(possible_state)
            # update Q value table
            possible_new_state = [i for i, v in enumerate(R[new_state, :]) if v != -1]
            temp_1 = [Q[new_state, x] for x in possible_new_state]
            Q[state, new_state] += R[state, new_state] + lr*(gamma * np.max(temp_1) - Q[state, new_state])
            state = new_state
            if new_state == 5:
                done = True

    print(Q/np.max(Q))

    print(R)

