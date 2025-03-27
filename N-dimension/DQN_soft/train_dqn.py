import numpy as np
import torch
from reacher.reacher_env import ReacherEnv_v0
import matplotlib.pyplot as plt
import sys
import tqdm
import argparse
from dqn import Replay_Memory, DQN
import math
import pandas as pd

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

class DQN_Agent():
    def __init__(self, num_dimension, lr, render=False,tau=0.005):
        self.n = num_dimension
        self.env = ReacherEnv_v0(self.n)
        self.replay_memory = Replay_Memory()
        self.current_network = DQN(self.env, self.n, lr)
        self.target_network = DQN(self.env, self.n, lr)
        self.target_network.model.eval()

        self.tau = tau
        self.epsilon_start = 1
        self.epsilon_end = 0.1
        self.epsilon_decay = 10000
        self.gamma = 0.99
        self.batch_size = 32
        self.render = render
        self.c = 0
        self.update_feq = 200
        self.burn_in_memory()
        self.optimizer = torch.optim.Adam(self.current_network.model.parameters(), lr=self.current_network.lr)

    def soft_update(self):
    #"""软更新目标网络参数: θ_target = (1-τ)·θ_target + τ·θ_current"""
        for target_param, current_param in zip(self.target_network.model.parameters(), self.current_network.model.parameters()):
            target_param.data.copy_(
                (1.0 - self.tau) * target_param.data + self.tau * current_param.data
            )
    
    def epsilon_greedy_policy(self, q_values):
        #epsilon = self.epsilon_end + (self.epsilon_start - self.epsilon_end) * math.exp(-1. * self.c / self.epsilon_decay)
        epsilon = max(self.epsilon_end, self.epsilon_start - (self.epsilon_start - self.epsilon_end) * min(1.0, self.c / self.epsilon_decay))
        if np.random.rand() >= epsilon:
            return self.greedy_policy(q_values)
        else:
            return np.random.choice(self.env.action_space)

    def greedy_policy(self, q_values):
        return torch.argmax(q_values).item()

    def train(self):
        state = torch.tensor(self.env.reset()).view(1, -1).float().to(device)
        while True:
            self.optimizer.zero_grad()
            q_values = self.current_network.model(state)
            action = self.epsilon_greedy_policy(q_values)

            next_state, reward, done, _ = self.env.step(action)
            next_state = torch.tensor(next_state).view(1, -1).float().to(device)
            action = torch.tensor(action).view(1).long().to(device)
            reward = torch.tensor(reward).view(1).float().to(device)
            transition = [state, action, reward, next_state, done]

            self.replay_memory.append(transition)

            mini_batch = self.replay_memory.sample_batch(self.batch_size)
            y = torch.tensor([]).float().to(device)
            states = torch.tensor([]).float().to(device)
            actions = torch.tensor([]).long().to(device)

            for transition in mini_batch:
                sample_state = transition[0]
                sample_action = transition[1]
                sample_reward = transition[2]
                sample_next_state = transition[3]
                sample_done = transition[-1]
                states = torch.cat((states, sample_state))
                actions = torch.cat((actions, sample_action))
                if sample_done:
                    y = torch.cat((y, sample_reward))
                else:
                    y = torch.cat((y, sample_reward + self.gamma * torch.max(self.target_network.model(sample_next_state)).view(1)))
            q = torch.gather(self.current_network.model(states), 1, actions.view(-1, 1))

            loss = 1 / self.batch_size * torch.sum((y - q.view(1, -1))**2)
            loss.backward(retain_graph=True)
            self.optimizer.step()
            self.soft_update()

            self.c += 1

            #if self.c % self.update_feq == 0:
                #path = self.current_network.save_model_weights()
                #self.target_network.load_model(path)
            #    self.soft_update()
            #    self.target_network.model.eval()
            if not done:
                state = next_state
            else:
                break

    def test(self, model_file=None):
        with torch.no_grad():
            test_trial = 20
            self.current_network.model.eval()

            total_rewards = []
            for _ in range(test_trial):
                state = torch.tensor(self.env.reset()).view(1, -1).float().to(device)
                rewards = 0
                while True:
                    q_values = self.current_network.model(state)
                    if self.render:
                        self.env.render()

                    action = self.greedy_policy(q_values)
                    next_state, reward, done, _ = self.env.step(action)
                    next_state = torch.tensor(next_state).view(1, -1).float().to(device)
                    rewards += reward
                    state = next_state
                    if done:
                        break
                total_rewards.append(rewards)

            total_rewards = np.array(total_rewards)
            reward_mean = np.mean(total_rewards)
            reward_std = np.sqrt(np.mean(np.sum((total_rewards - reward_mean)**2) / test_trial))

            self.current_network.model.train()
        return reward_mean, reward_std

    def burn_in_memory(self):
        num = 0
        current_state = self.env.reset()

        while num < self.replay_memory.burn_in:
            action = np.random.choice(self.env.action_space)
            next_state, reward, done, _ = self.env.step(action)

            transition = [
                torch.tensor(current_state).view(1, -1).float().to(device),
                torch.tensor(action).view(1).long().to(device),
                torch.tensor(reward).view(1).float().to(device),
                torch.tensor(next_state).view(1, -1).float().to(device),
                done
            ]
            self.replay_memory.append(transition)
            num += 1
            if done:
                current_state = self.env.reset()
            if not done:
                current_state = next_state

def parse_arguments():
    parser = argparse.ArgumentParser(description='Deep Q Network Argument Parser')
    parser.add_argument('--env', dest='env', type=int, default=6)
    parser.add_argument('--render', dest='render', type=int, default=0)
    parser.add_argument('--train', dest='train', type=int, default=1)
    parser.add_argument('--model', dest='model_file', type=str)
    parser.add_argument('--lr', dest='lr', type=float, default=1e-5)
    parser.add_argument('--tau', dest='tau', type=float, default=0.005)  # 添加tau参数
    return parser.parse_args()

def main(args):
    args = parse_arguments()
    num_dimension = args.env
    lr = args.lr
    render = args.render
    tau = args.tau
    num_episodes = 1000
    num_trails = 10
    SAVE_STR = 'DQN-' + str(num_dimension) + 'd'

    reward_means_total = []
    reward_stds_total = []

    for trail in tqdm.tqdm(range(num_trails)):
        reward_means = []
        reward_stds = []
        DQN = DQN_Agent(num_dimension, lr, render=render, tau=tau)

        for epi in range(num_episodes):
            if epi % 10 == 0:
                reward_mean, reward_std = DQN.test()
                reward_means.append(reward_mean)
                reward_stds.append(reward_std)

                print("The test reward for episode %d is %.1f." % (epi, reward_means[-1]))
                print('The epsilon is:', DQN.epsilon_end + (DQN.epsilon_start - DQN.epsilon_end) * math.exp(-1. * DQN.c / DQN.epsilon_decay))

                if reward_means[-1] == max(reward_means):
                    path = DQN.current_network.save_best_model()
                    print("The best model is saved with reward %.1f." % (reward_means[-1]))
            DQN.train()
        reward_means_total.append(reward_means)
        reward_stds_total.append(reward_stds)

    pd.DataFrame(np.array(reward_means_total).mean(axis=0)).to_csv('./data/' + SAVE_STR + '_mean.csv', header=None, index=None)
    pd.DataFrame(np.array(reward_stds_total).mean(axis=0)).to_csv('./data/' + SAVE_STR + '_std.csv', header=None, index=None)

    plt.plot(np.array(reward_means_total).mean(axis=0))
    plt.savefig("./plots/reward_" + SAVE_STR)

if __name__ == '__main__':
    main(sys.argv)