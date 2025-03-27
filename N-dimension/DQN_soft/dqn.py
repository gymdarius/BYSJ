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

class DQN():
    def __init__(self, env, n, lr, logdir='./model'):
        self.env = env
        self.lr = lr
        self.logdir = logdir
        self.action_n = 3**n
        self.obs_n = 2*n
        self.model = FullyConnectedModel(state_dim=self.obs_n, action_dim=self.action_n).to(device)

    def save_model_weights(self):
        self.path = os.path.join(self.logdir, "model_vanilla")
        torch.save(self.model.state_dict(), self.path)
        return self.path

    def load_model(self, path):
        return self.model.load_state_dict(torch.load(path))

    def save_best_model(self):
        self.path = os.path.join(self.logdir, "best_model_vanilla")
        torch.save(self.model.state_dict(), self.path)
        return self.path

class Replay_Memory():
    def __init__(self, memory_size=100000, burn_in=10000):
        self.memory_size = memory_size
        self.burn_in = burn_in
        self.mem_pool = collections.deque([], maxlen=self.memory_size)

    def sample_batch(self, batch_size=32):
        return random.sample(self.mem_pool, batch_size)

    def append(self, transition):
        self.mem_pool.append(transition)