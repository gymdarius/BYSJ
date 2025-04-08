# Replace redundant device assignments and ensure consistent GPU usage

from typing_extensions import Self
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import numpy as np
import random
from collections import namedtuple
from atari_wrappers import wrap_deepmind,make_atari
Experience = namedtuple('Experience',
                        ('s', 'a_I', 'a_II', 'r', 's_','d'))
BATCH_SIZE = 32
MEMORY_SIZE = 1000000
LEARN_START = 50000
TARGET_NET_UPDATE_FREQUENCY = 10000
LR = 0.0000625
GAMMA = 0.99
EPSILON_START  = 1
EPSILON_END = 0.1
seed = 1

# Set device once at the beginning
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")

class Net_I(nn.Module):
    def __init__(self, num_actions, seed):
        super(Net_I,self).__init__()
        self.seed = torch.manual_seed(seed)
        self.conv1 = nn.Conv2d(in_channels=4, out_channels=32, kernel_size=8 , stride=4)
        self.conv2 = nn.Conv2d(in_channels=32, out_channels=64, kernel_size=4 , stride=2)
        self.conv3 = nn.Conv2d(in_channels=64, out_channels=64, kernel_size=3 , stride=1)
        self.fc1 = nn.Linear(3136,512)
        self.fc2 = nn.Linear(512 ,num_actions)

    def forward(self,x):
        x = x/255.
        x = F.relu(self.conv1(x))
        x = F.relu(self.conv2(x))
        x = F.relu(self.conv3(x))
        x = x.view(-1,3136)
        x = F.relu(self.fc1(x))
        out = self.fc2(x)
        return out

    def init_weights(self, m):
        if type(m) == nn.Linear:
            torch.nn.init.kaiming_normal_(m.weight, nonlinearity='relu')
            m.bias.data.fill_(0.0)
        
        if type(m) == nn.Conv2d:
            torch.nn.init.kaiming_normal_(m.weight, nonlinearity='relu')

class Net_II(nn.Module):
    def __init__(self, num_actions, seed, add_state_dim=0):
        super(Net_II,self).__init__()
        self.add_state_dim = add_state_dim
        self.seed = torch.manual_seed(seed)
        self.conv1 = nn.Conv2d(in_channels=4, out_channels=32, kernel_size=8 , stride=4)
        self.conv2 = nn.Conv2d(in_channels=32, out_channels=64, kernel_size=4 , stride=2)
        self.conv3 = nn.Conv2d(in_channels=64, out_channels=64, kernel_size=3 , stride=1)
        self.fc1 = nn.Linear(3136 + add_state_dim,512)
        self.fc2 = nn.Linear(512 ,num_actions)

    def forward(self,x,action):
        x = x/255.
        x = F.relu(self.conv1(x))
        x = F.relu(self.conv2(x))
        x = F.relu(self.conv3(x))
        x = torch.cat((x.view(-1,3136),action), dim=1)
        x = F.relu(self.fc1(x))
        out = self.fc2(x)
        return out

    def init_weights(self, m):
        if type(m) == nn.Linear:
            torch.nn.init.kaiming_normal_(m.weight, nonlinearity='relu')
            m.bias.data.fill_(0.0)
        
        if type(m) == nn.Conv2d:
            torch.nn.init.kaiming_normal_(m.weight, nonlinearity='relu')

class MAN:
    def __init__(self,env):
        self.env_raw = make_atari(env)
        self.env =  wrap_deepmind(self.env_raw, frame_stack=False, episode_life=True, clip_rewards=True)

        self.action_space_I = np.arange(9)
        self.action_space_II = np.arange(2)
        self.action_map = np.array([[0,1],
                                    [2,10],
                                    [3,11],
                                    [4,12],
                                    [5,13],
                                    [6,14],
                                    [7,15],
                                    [8,16],
                                    [9,17]])

        self.num_actions_I = self.action_space_I.shape[0]               
        self.num_actions_II = self.action_space_II.shape[0]                

        # Create networks and immediately move them to the GPU
        self.target_net_I = Net_I(self.num_actions_I, seed).to(device)
        self.target_net_II = Net_II(self.num_actions_II, seed, add_state_dim=self.num_actions_I).to(device)

        self.eva_net_I = Net_I(self.num_actions_I, seed).to(device)
        self.eva_net_II = Net_II(self.num_actions_II, seed, add_state_dim=self.num_actions_I).to(device)
        
        self.eva_net_I.apply(self.eva_net_I.init_weights)
        self.eva_net_II.apply(self.eva_net_II.init_weights)

        self.target_net_I.load_state_dict(self.eva_net_I.state_dict())
        self.target_net_II.load_state_dict(self.eva_net_II.state_dict())

        self.target_net_I.eval()
        self.target_net_II.eval()

        self.memory_counter = 0
        self.state_counter = 0
        self.memory = []

        # Use cuda tensors for optimizers
        self.optimizer_I = optim.Adam(self.eva_net_I.parameters(), lr=LR, eps=1.5e-4)
        self.optimizer_II = optim.Adam(self.eva_net_II.parameters(), lr=LR, eps=1.5e-4)

        self.epsilon = EPSILON_START
        self.state_size = 5   #4+1
        self.i = 0
        
    def state_initialize(self):
        self.state_buffer = []
        img, _ = self.env.reset()

        for i in range(self.state_size):
            self.state_buffer.append(img)

        return self.state_buffer[1:5]

    def one_hot_encoder(self, x, num_classes):
        x = torch.flatten(x)
        # No need to call .to(device) again since x is already on device
        x_one_hot = F.one_hot(x, num_classes)
        return x_one_hot

    def choose_action_I(self, x, train=True):
        if train:          
            if len(self.memory) >= LEARN_START:
                self.epsilon -= (EPSILON_START-EPSILON_END)/MEMORY_SIZE
                self.epsilon = max(self.epsilon, EPSILON_END)
            epsilon = self.epsilon
        else:
            epsilon = 0.05
            
        if np.random.uniform() > epsilon/2:
            # Optimize tensor creation and device placement
            x = torch.tensor(np.array(x, dtype=np.float32), device=device).unsqueeze(0)
            with torch.no_grad():
                q_value = self.eva_net_I(x)
            action = torch.argmax(q_value).item()
        else:
            action = np.random.choice(self.action_space_I)

        return action

    def choose_action_II(self, x, action_I, train=True):
        if train:          
            if len(self.memory) >= LEARN_START:
                self.epsilon -= (EPSILON_START-EPSILON_END)/MEMORY_SIZE
                self.epsilon = max(self.epsilon, EPSILON_END)
            epsilon = self.epsilon
        else:
            epsilon = 0.05
            
        if np.random.uniform() > epsilon/2:
            action_one_hot = self.one_hot_encoder(action_I, self.num_actions_I)
            # Optimize tensor creation
            x = torch.tensor(np.array(x, dtype=np.float32), device=device).unsqueeze(0)
            with torch.no_grad():
                q_value = self.eva_net_II(x, action_one_hot)
            action = torch.argmax(q_value).item()
        else:
            action = np.random.choice(self.action_space_II)

        return action

    def choose_action(self, action_I, action_II):
        return self.action_map[action_I, action_II]

    def store_transition(self, s, a_I, a_II, r, s_, d):
        self.state_counter += 1
        exp = [s, a_I, a_II, r, s_, d]
        if len(self.memory) >= MEMORY_SIZE:
            self.memory.pop(0)
        self.memory.append(exp)

    def learn(self):
        sample = random.sample(self.memory, BATCH_SIZE)
        batch = Experience(*zip(*sample))

        # Batch tensors efficiently
        b_s = torch.tensor(np.array(batch.s, dtype=np.float32), device=device)
        b_a_I = torch.tensor(batch.a_I, device=device, dtype=torch.int64).unsqueeze(1)
        b_a_II = torch.tensor(batch.a_II, device=device, dtype=torch.int64).unsqueeze(1)
        b_r = torch.tensor(np.array(batch.r, dtype=np.float32), device=device).unsqueeze(1)
        b_s_ = torch.tensor(np.array(batch.s_, dtype=np.float32), device=device)
        b_d = torch.tensor(np.array(batch.d, dtype=np.float32), device=device).unsqueeze(1)

        # Forward pass for network I
        q_eval_I = torch.gather(self.eva_net_I(b_s), 1, b_a_I)
        
        # One-hot encoding for action I
        b_a_I_one_hot = self.one_hot_encoder(b_a_I, self.num_actions_I)
        
        # Forward pass for network II
        q_eval_II = torch.gather(self.eva_net_II(b_s, b_a_I_one_hot), 1, b_a_II)

        # Double DQN target computation
        with torch.no_grad():
            argmax_II = self.eva_net_I(b_s_).max(1)[1].long()
            q_next_II = self.target_net_I(b_s_).gather(1, argmax_II.unsqueeze(1))

            argmax_I = self.eva_net_II(b_s_, b_a_I_one_hot).max(1)[1].long()
            q_next_I = self.target_net_II(b_s_, b_a_I_one_hot).gather(1, argmax_I.unsqueeze(1))

        # Calculate target values
        q_target_I = b_r + GAMMA * q_next_I.max(1)[0].unsqueeze(1) * (-b_d + 1)
        q_target_II = b_r + GAMMA * q_next_II.max(1)[0].unsqueeze(1) * (-b_d + 1)
   
        # Compute losses
        loss_I = F.mse_loss(q_eval_I, q_target_I)
        loss_II = F.mse_loss(q_eval_II, q_target_II)

        # Backpropagation
        self.optimizer_I.zero_grad()
        self.optimizer_II.zero_grad()
        loss_I.backward()
        loss_II.backward()
        self.optimizer_I.step()
        self.optimizer_II.step()
        
        return loss_I.item(), loss_II.item()

    def evaluate(self, env, num_episode=15):
        env = wrap_deepmind(env)
        e_rewards = []
        for i in range(num_episode):
            img, _ = env.reset()
            sum_r = 0
            done = False
            state_buffer = []
            for i in range(5):
                state_buffer.append(img)
            s = state_buffer[1:5]
            while not done:
                a_i = self.choose_action_I(s, train=False)
                a_i = torch.tensor(a_i, device=device, dtype=torch.int64)
                a_ii = self.choose_action_II(s, a_i, train=False)
                
                a = self.choose_action(a_i.item(), a_ii)
    
                img, r, terminated, truncated, info = env.step(a)
                done = terminated or truncated

                sum_r += r
                state_buffer.pop(0)
                state_buffer.append(img)
                s_ = state_buffer[1:5]
                s = s_

            e_rewards.append(sum_r)
        return e_rewards