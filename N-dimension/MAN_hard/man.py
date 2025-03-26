import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import os
import collections
import random

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

class FullyConnectedModel(torch.nn.Module):
    def __init__(self, state_dim, action_dim):
        super().__init__()
        self.fc1 = nn.Linear(state_dim, 256)
        self.fc2 = nn.Linear(256, 512)
        self.fc3 = nn.Linear(512, 256)
        self.fc4 = nn.Linear(256, action_dim)

        nn.init.xavier_uniform_(self.fc1.weight)
        nn.init.xavier_uniform_(self.fc2.weight)
        nn.init.xavier_uniform_(self.fc3.weight)
        nn.init.xavier_uniform_(self.fc4.weight)

    def forward(self, x):
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = F.relu(self.fc3(x))
        x = self.fc4(x)
        return x


class MAN():
    def __init__(self, env, lr, logdir='./model'):
        self.env = env
        self.lr = lr
        self.logdir = logdir
        self.action_n = self.env.action_space
        self.obs_n = self.env.observation_space
        self.target_nets = []
        self.evaluate_nets = []
        self.optimizers = []

        previous_action = 0
        for num_action in self.action_n:
            target_net = FullyConnectedModel(state_dim=self.obs_n+previous_action, action_dim=num_action).to(device)
            evaluate_net = FullyConnectedModel(state_dim=self.obs_n+previous_action, action_dim=num_action).to(device)
            target_net.load_state_dict(evaluate_net.state_dict())
            target_net.eval()

            self.target_nets.append(target_net)
            self.evaluate_nets.append(evaluate_net)
            self.optimizers.append(torch.optim.Adam(evaluate_net.parameters(), lr=self.lr))

            previous_action += 1

    def save_model_weights(self):
        for i, model in enumerate(self.evaluate_nets):
            path = os.path.join(self.logdir, "model_man_")
            torch.save(model.state_dict(), path + str(i))
        return path

    def load_model(self, path):
        for i in range(self.env.n):
            self.evaluate_nets[i].load_state_dict(torch.load(path + str(i), map_location=device))

    def save_best_model(self):
        for i, model in enumerate(self.evaluate_nets):
            path = os.path.join(self.logdir, "model_man_" + str(i) + '_best')
            torch.save(model.state_dict(), path)
        return path

# Replay_Memory 类无需改动
class Replay_Memory():
    def __init__(self, memory_size=100000, burn_in=10000):

    # The memory essentially stores transitions recorder from the agent
    # taking actions in the environment.

    # Burn in episodes define the number of episodes that are written into the memory from the
    # randomly initialized agent. Memory size is the maximum size after which old elements in the memory are replaced.
    # A simple (if not the most efficient) was to implement the memory is as a list of transitions.

        self.memory_size = memory_size
        self.burn_in = burn_in
        self.mem_pool = collections.deque([], maxlen=self.memory_size)

    def sample_batch(self, batch_size=256):
    # This function returns a batch of randomly sampled transitions - i.e. state, action, reward, next state, terminal flag tuples.
    # You will feed this to your model to train.
        return random.sample(self.mem_pool, batch_size)

    def append(self, transition):
    # Appends transition to the memory.
        self.mem_pool.append(transition)