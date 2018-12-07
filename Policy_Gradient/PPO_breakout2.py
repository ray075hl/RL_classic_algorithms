import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F

from baselines.common.atari_wrappers import FrameStack, make_atari, wrap_deepmind
from baselines.common.vec_env.subproc_vec_env import SubprocVecEnv
from baselines.common.vec_env.vec_frame_stack import VecFrameStack

import gym
import numpy as np
import cv2

use_cuda = torch.cuda.is_available()
device = torch.device('cuda' if use_cuda else 'cpu')

env_name = 'BreakoutNoFrameskip-v0'
env_num = 8
ppo_epochs       = 3
mini_batch_size  = 256

max_frames = 150000000
frame_idx  = 0

action_space = 4
lr = 0.0001
gamma_ = 0.99
lambda_ = 0.95

num_steps = 128

def ortho_weights(shape, scale=1.):
    """ PyTorch port of ortho_init from baselines.a2c.utils """
    shape = tuple(shape)

    if len(shape) == 2:
        flat_shape = shape[1], shape[0]
    elif len(shape) == 4:
        flat_shape = (np.prod(shape[1:]), shape[0])
    else:
        raise NotImplementedError

    a = np.random.normal(0., 1., flat_shape)
    u, _, v = np.linalg.svd(a, full_matrices=False)
    q = u if u.shape == flat_shape else v
    q = q.transpose().copy().reshape(shape)

    if len(shape) == 2:
        return torch.from_numpy((scale * q).astype(np.float32))
    if len(shape) == 4:
        return torch.from_numpy((scale * q[:, :shape[1], :shape[2]]).astype(np.float32))


def atari_initializer(module):
    """ Parameter initializer for Atari models
    Initializes Linear, Conv2d, and LSTM weights.
    """
    classname = module.__class__.__name__

    if classname == 'Linear':
        module.weight.data = ortho_weights(module.weight.data.size(), scale=np.sqrt(2.))
        module.bias.data.zero_()

    elif classname == 'Conv2d':
        module.weight.data = ortho_weights(module.weight.data.size(), scale=np.sqrt(2.))
        module.bias.data.zero_()

    elif classname == 'LSTM':
        for name, param in module.named_parameters():
            if 'weight_ih' in name:
                param.data = ortho_weights(param.data.size(), scale=1.)
            if 'weight_hh' in name:
                param.data = ortho_weights(param.data.size(), scale=1.)
            if 'bias' in name:
                param.data.zero_()
def ortho_weights(shape, scale=1.):
    """ PyTorch port of ortho_init from baselines.a2c.utils """
    shape = tuple(shape)

    if len(shape) == 2:
        flat_shape = shape[1], shape[0]
    elif len(shape) == 4:
        flat_shape = (np.prod(shape[1:]), shape[0])
    else:
        raise NotImplementedError

    a = np.random.normal(0., 1., flat_shape)
    u, _, v = np.linalg.svd(a, full_matrices=False)
    q = u if u.shape == flat_shape else v
    q = q.transpose().copy().reshape(shape)

    if len(shape) == 2:
        return torch.from_numpy((scale * q).astype(np.float32))
    if len(shape) == 4:
        return torch.from_numpy((scale * q[:, :shape[1], :shape[2]]).astype(np.float32))

def compute_gae(next_value, rewards, masks, values, gamma=0.99, tau=0.95):
    #print('+++: ', values)
    #print('---: ', len(rewards))
    values = values + [next_value]
    gae = 0
    returns = []

    for step in reversed(range(len(rewards))):  # len(rewards) = 10
        delta = rewards[step] + gamma * values[step + 1] * masks[step] - values[step]
        gae = delta + gamma * tau * masks[step] * gae
        #xxx = gae.clone()
        returns.insert(0, gae + values[step])

        # print('----------')
        # print(values[step])

    return returns

def ppo_iter(mini_batch_size, states, actions, log_probs, returns, advantage):
    batch_size = states.size(0)
    for _ in range(batch_size // mini_batch_size):
        rand_ids = np.random.randint(0, batch_size, mini_batch_size)
        yield states[rand_ids, :], actions[rand_ids, :], log_probs[rand_ids, :], returns[rand_ids, :], advantage[
                                                                                                       rand_ids, :]
from torch.nn.utils import clip_grad_norm_

def ppo_update(ppo_epochs, mini_batch_size, states, actions, log_probs, returns, advantages, clip_param=0.2):
    for _ in range(ppo_epochs):
        perm = np.arange(states.size()[0])
        np.random.shuffle(perm)
        perm = torch.LongTensor(perm).to(device)
        states, actions, log_probs, returns, advantages = states[perm].clone(), actions[perm].clone(), \
                                                           log_probs[perm].clone(), returns[perm].clone(), advantages[perm].clone()
        for state, action, old_log_probs, return_, advantage in ppo_iter(mini_batch_size, states, actions, log_probs,
                                                                         returns, advantages):
            # advantage = (advantage - advantage.mean()) / advantage.std()
            dist, value = model(state)
            prob = F.softmax(dist, dim=-1)
            log_prob_ = F.log_softmax(dist, dim=-1)
            entropy = (prob*(-1.0*log_prob_)).mean()
            new_log_probs = F.log_softmax(dist, dim=1) #dist.log_prob(action)


            ratio = (new_log_probs[range(mini_batch_size), action.squeeze()] - old_log_probs[range(mini_batch_size), action.squeeze()]).exp()
            surr1 = ratio * advantage
            surr2 = torch.clamp(ratio, 1.0 - clip_param, 1.0 + clip_param) * advantage

            actor_loss =  -torch.min(surr1, surr2).mean()
            critic_loss = (return_ - value).pow(2).mean()
            #print('critic_loss: {}, actor_loss: {}, entropy_loss: {}'.format(critic_loss, actor_loss, entropy))
            loss = 0.5*critic_loss + actor_loss - 0.01 * entropy

            optimizer.zero_grad()
            loss.backward()
            clip_grad_norm_(model.parameters(), 0.5)
            optimizer.step()

def make_env(env_name, rank, seed):
    env = make_atari(env_name)
    env.seed(seed + rank)
    env = wrap_deepmind(env, episode_life=False, clip_rewards=False)
    return env

env_fns = []
for rank in range(env_num):
    env_fns.append(lambda: make_env(env_name, rank, 100 + rank))

envs = SubprocVecEnv(env_fns)
envs = VecFrameStack(envs, 4)

test_env = make_env(env_name, 0, 100)
test_env = FrameStack(test_env, 4)
class Model(nn.Module):
    def __init__(self, num_actions):
        """ Basic convolutional actor-critic network for Atari 2600 games
        Equivalent to the network in the original DQN paper.
        Args:
            num_actions (int): the number of available discrete actions
        """
        super().__init__()

        self.conv = nn.Sequential(nn.Conv2d(4, 32, 8, stride=4),
                                  nn.ReLU(inplace=True),
                                  nn.Conv2d(32, 64, 4, stride=2),
                                  nn.ReLU(inplace=True),
                                  nn.Conv2d(64, 64, 3, stride=1),
                                  nn.ReLU(inplace=True))

        self.fc = nn.Sequential(nn.Linear(64 * 7 * 7, 512),
                                nn.ReLU(inplace=True))

        self.pi = nn.Linear(512, num_actions)
        self.v = nn.Linear(512, 1)

        self.num_actions = num_actions

        # parameter initialization
        self.apply(atari_initializer)
        self.pi.weight.data = ortho_weights(self.pi.weight.size(), scale=.01)
        self.v.weight.data = ortho_weights(self.v.weight.size())

    def forward(self, conv_in):
        
        """ Module forward pass
        Args:
            conv_in (Variable): convolutional input, shaped [N x 4 x 84 x 84]
        Returns:
            pi (Variable): action probability logits, shaped [N x self.num_actions]
            v (Variable): value predictions, shaped [N x 1]
        """
        conv_in = conv_in.float() / 255.0
        N = conv_in.size()[0]

        conv_out = self.conv(conv_in).view(N, 64 * 7 * 7)

        fc_out = self.fc(conv_out)

        pi_out = self.pi(fc_out)
        v_out = self.v(fc_out)

        return pi_out, v_out
model = Model(4).to(device)


def play_test():
    state = test_env.reset()
    done = False
    ep_reward = 0.0
    last_action = np.array([-1])
    action_repeat = 0
    test_repeat_max = 100
    while not done:
        state = np.array(state)
        state = torch.from_numpy(state.transpose((2, 0, 1))).unsqueeze(0).to(device)

        pi, _ = model(state)
        _, action = torch.max(pi, dim=1)

        # abort after {self.test_repeat_max} discrete action repeats
        if action.data[0] == last_action.data[0]:
            action_repeat += 1
            if action_repeat == test_repeat_max:
                return ep_reward
        else:
            action_repeat = 0
        last_action = action

        state, reward, done, _ = test_env.step(action.data.cpu().numpy())

        ep_reward += reward

    return ep_reward




optimizer = optim.Adam(model.parameters(), lr=lr)

test_rewards_list = []

state = envs.reset()

print(state.shape)
while frame_idx < max_frames:
    log_probs = []
    values = []
    states = []
    actions = []
    rewards = []
    masks = []
    # first state ,  last state
    for _ in range(num_steps):
        state = torch.FloatTensor(state.transpose(0,3,1,2)).to(device)
        logits, value = model(state)



        u = torch.rand(logits.size()).to(device)
        _, action = torch.max(logits.data - (-u.log()).log(), 1)
        action_list = action.cpu().numpy()      
        '''
        prob = F.softmax(logits, dim=1)
        action_list = []

        for i in range(len(prob)):
            action = np.random.choice(action_space, p=prob.cpu().detach().numpy()[i])
            action_list.append(action)
        '''

        next_state, reward, done, _ = envs.step(action_list)
        log_prob = F.log_softmax(logits, dim=-1)
        log_probs.append(log_prob)


        values.append(value)
        rewards.append(torch.FloatTensor(reward).unsqueeze(1).to(device))   # 2D list
        masks.append(torch.FloatTensor(1 - done).unsqueeze(1).to(device))   # 2D list


        states.append(state)
        actions.append(torch.from_numpy(np.asarray(action_list)).unsqueeze(1))
        state = next_state  # -----------------
        frame_idx += 1
    if frame_idx % 1280 == 0:
        print(frame_idx, play_test())

    next_state = torch.FloatTensor(next_state.transpose(0,3,1,2)).to(device)
    _, next_value = model(next_state)
    returns = compute_gae(next_value, rewards, masks, values)
    #print('******')
    #print(test__)
    #print('******')
    returns = torch.cat(returns).detach()
    #print(returns)
    log_probs = torch.cat(log_probs).detach()
    values = torch.cat(values).detach()
    #print(values)
    # print(values.size())
    # exit()
    states = torch.cat(states)
    actions = torch.cat(actions)
    advantage = returns- values     # target_reward - predict_reward

    ppo_update(ppo_epochs, mini_batch_size, states, \
               actions, log_probs, returns, advantage)
