import numpy as np
import torch
import gym
import stack
import matplotlib.pyplot as plt
import sys
import tqdm
import argparse
from MAN import Replay_Memory
from MAN import DQN
from MAN import DVN
import math
import pandas as pd

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

heights = torch.tensor([[0, 0.025, 0], [0.025, 0.025, 0.025], [0.05, 0.05, 0.05], [0.075, 0.075, 0.075]]).to(device)
SAVE_STR = 'MAN'

class Agent():

    def __init__(self, environment_name, alpha, beta, alpha_step, beta_step, lr, render=False):
        self.env = gym.make(environment_name)
        self.environment_name = environment_name
        self.current_network_Q = DQN(self.env, lr)
        self.target_network_Q = DQN(self.env, lr)
        self.current_network_V = DVN(self.env, lr)
        self.target_network_V = DVN(self.env, lr)
        self.target_network_Q.model.eval()
        self.target_network_V.model.eval()

        self.report_freq = 500
        self.epsilon_start = 1
        self.epsilon_end = 0.1
        self.epsilon_decay = 10000
        self.gamma = 0.99
        self.batch_size = 32
        self.replay_memory = Replay_Memory()
        self.render = render
        self.c = 0
        self.burn_in_memory()
        self.optimizer_Q = torch.optim.Adam(self.current_network_Q.model.parameters(), lr=self.current_network_Q.lr)
        self.optimizer_V = torch.optim.Adam(self.current_network_V.model.parameters(), lr=self.current_network_V.lr)
        self.tau = 0.005

    def soft_update(self, target, source):
        for tp, sp in zip(target.parameters(), source.parameters()):
            tp.data.copy_((1 - self.tau) * tp.data + self.tau * sp.data)

    def epsilon_greedy_policy_Q(self, q_values, over_ride=None):
        if over_ride is not None:
            epsilon = over_ride
        else:
            epsilon = max(self.epsilon_end, self.epsilon_start - (self.epsilon_start - self.epsilon_end) * self.c / self.epsilon_decay)
        if np.random.rand() >= epsilon:
            return self.greedy_policy_Q(q_values)
        else:
            return self.env.action_space.sample()

    def greedy_policy_Q(self, q_values):
        action = torch.argmax(q_values)
        return action.item()

    def epsilon_greedy_policy_V(self, v_values, over_ride=None):
        if over_ride is not None:
            epsilon = over_ride
        else:
            epsilon = max(self.epsilon_end, self.epsilon_start - (self.epsilon_start - self.epsilon_end) * self.c / self.epsilon_decay)
        if np.random.rand() >= epsilon:
            return self.greedy_policy_V(v_values)
        else:
            return np.random.choice(self.env.cube_pool)

    def greedy_policy_V(self, v_values):
        sorted, indices = torch.sort(v_values, descending=True)
        for index in indices.tolist()[0]:
            if index in self.env.cube_pool:
                return index

    def train(self, num_step):
        reward_mean, reward_std, max_height_mean, max_height_std, bumpiness_mean, bumpiness_std = test(self.environment_name, self.current_network_V.model, self.current_network_Q.model)
        reward_means = [reward_mean]
        reward_stds = [reward_std]
        max_height_means = [max_height_mean]
        max_height_stds = [max_height_std]
        bumpiness_means = [bumpiness_mean]
        bumpiness_stds = [bumpiness_std]
        total_loss_V = 0
        total_loss_Q = 0

        state = torch.tensor(self.env.reset()).view(1, -1).float().to(device)

        for i in range(num_step):
            self.optimizer_Q.zero_grad()
            self.optimizer_V.zero_grad()

            v_values = self.current_network_V.model(state)
            index = self.epsilon_greedy_policy_V(v_values)

            q_values = self.current_network_Q.model(torch.cat((state, heights[index].view(1, -1)), dim=1))
            action = self.epsilon_greedy_policy_Q(q_values)

            next_state, reward, done, _ = self.env.step([index, action])
            next_state = torch.tensor(next_state).view(1, -1).float().to(device)

            transition = [state, index, action, reward, next_state, done]
            self.replay_memory.add(transition)
            mini_batch = self.replay_memory.sample_batch(self.batch_size)

            sample_state = mini_batch[0]
            sample_action_I = mini_batch[1]
            sample_action_II = mini_batch[2]
            sample_reward = mini_batch[3]
            sample_next_state = mini_batch[4]
            sample_done = mini_batch[-1]

            actions_next_Q = torch.argmax(self.current_network_V.model(sample_next_state).detach(), dim=1)
            q_next_Q = torch.gather(self.target_network_V.model(sample_next_state).detach(), 1, actions_next_Q.view(-1, 1)).squeeze(1)
            target_Q = sample_reward + self.gamma * q_next_Q * (1 - sample_done)

            actions_next_V = torch.argmax(self.current_network_Q.model(torch.cat((sample_next_state, heights[sample_action_I].squeeze(1)), dim=1)).detach(), dim=1)
            q_next_V = torch.gather(self.target_network_Q.model(torch.cat((sample_next_state, heights[sample_action_I].squeeze(1)), dim=1)).detach(), 1, actions_next_V.view(-1, 1)).squeeze(1)
            target_V = sample_reward + self.gamma * q_next_V * (1 - sample_done)

            eval_Q = torch.gather(self.current_network_Q.model(torch.cat((sample_state, heights[sample_action_I].squeeze(1)), dim=1)), 1, sample_action_II).squeeze(1)
            eval_V = torch.gather(self.current_network_V.model(sample_state), 1, sample_action_I).squeeze(1)

            loss_Q = torch.mean((target_Q - eval_Q) ** 2)
            loss_Q.backward()
            self.optimizer_Q.step()

            loss_V = torch.mean((target_V - eval_V) ** 2)
            loss_V.backward()
            self.optimizer_V.step()

            with torch.no_grad():
                self.soft_update(self.target_network_Q.model, self.current_network_Q.model)
                self.soft_update(self.target_network_V.model, self.current_network_V.model)

            self.c += 1
            total_loss_V += loss_V.item()
            total_loss_Q += loss_Q.item()

            if self.c % self.report_freq == 0:
                reward_mean, reward_std, max_height_mean, max_height_std, bumpiness_mean, bumpiness_std = test(self.environment_name, self.current_network_V.model, self.current_network_Q.model)
                reward_means.append(reward_mean)
                reward_stds.append(reward_std)
                max_height_means.append(max_height_mean)
                max_height_stds.append(max_height_std)
                bumpiness_means.append(bumpiness_mean)
                bumpiness_stds.append(bumpiness_std)
                eps = max(self.epsilon_end, self.epsilon_start - (self.epsilon_start - self.epsilon_end) * self.c / self.epsilon_decay)
                print(f"Step: {self.c}, Reward mean: {reward_mean:.2f}, Reward std: {reward_std:.2f}, Max_height mean: {max_height_mean:.2f}, Max_height std: {max_height_std:.2f}, Bumpiness mean: {bumpiness_mean:.2f}, Bumpiness std: {bumpiness_std:.2f}, Epsilon: {eps:.2f}")
                if reward_means[-1] == max(reward_means):
                    self.current_network_Q.save_best_model()
                    self.current_network_V.save_best_model()
            if not done:
                state = next_state
            else:
                state = torch.tensor(self.env.reset()).view(1, -1).float().to(device)

        return reward_means, max_height_means, bumpiness_means

    def burn_in_memory(self):
        num = 0
        current_state = self.env.reset()
        while num < self.replay_memory.burn_in:
            index = np.random.choice(self.env.cube_pool)
            action = int(self.env.action_space.sample())
            next_state, reward, done, _ = self.env.step([index, action])
            next_state = torch.tensor(next_state).view(1, -1).float().to(device)
            transition = [current_state, index, action, reward, next_state, done]
            self.replay_memory.add(transition)
            num += 1
            if done:
                current_state = self.env.reset()
            if not done:
                current_state = next_state

def test(env_name, model_I, model_II, model_file=None):
    env = gym.make(env_name)
    with torch.no_grad():
        test_trial = 10
        epsilon = -1
        total_rewards = []
        total_max_heights = []
        total_bumpiness = []

        for _ in range(test_trial):
            state = torch.tensor(env.reset()).view(1, -1).float().to(device)
            rewards = 0
            while True:
                v_values = model_I(state)
                if np.random.rand() >= epsilon:
                    sorted, indices = torch.sort(v_values, descending=True)
                    for index in indices.tolist()[0]:
                        if index in env.cube_pool:
                            action_I = index
                            break
                else:
                    action_I = np.random.choice(env.cube_pool)

                q_values = model_II(torch.cat((state, heights[action_I].view(1, -1)), dim=1))
                if np.random.rand() >= epsilon:
                    action_II = torch.argmax(q_values).item()
                else:
                    action_II = env.action_space.sample()
                next_state, reward, done, _ = env.step([action_I, action_II])
                next_state = torch.tensor(next_state).view(1, -1).float().to(device)
                rewards += reward
                state = next_state
                if done:
                    total_max_heights.append(torch.max(next_state.cpu()).item())
                    virance = 0
                    for i in range(next_state.size(1) - 1):
                        virance += torch.abs(next_state[0, i + 1] - next_state[0, i])
                    total_bumpiness.append(virance.cpu().item())
                    break
            total_rewards.append(rewards)

        reward_mean = np.mean(total_rewards)
        reward_std = np.std(total_rewards)
        max_height_mean = np.mean(total_max_heights)
        max_height_std = np.std(total_max_heights)
        bumpiness_mean = np.mean(total_bumpiness)
        bumpiness_std = np.std(total_bumpiness)

    return reward_mean, reward_std, max_height_mean, max_height_std, bumpiness_mean, bumpiness_std

def parse_arguments():
    parser = argparse.ArgumentParser(description='Deep Q Network Argument Parser')
    parser.add_argument('--env', dest='env', type=str, default="Stack-v0")
    parser.add_argument('--render', dest='render', type=int, default=0)
    parser.add_argument('--train', dest='train', type=int, default=1)
    parser.add_argument('--model', dest='model_file', type=str)
    parser.add_argument('--lr', dest='lr', type=float, default=3e-4)
    parser.add_argument('--alpha', dest='alpha', type=float, default=0.7)
    parser.add_argument('--beta', dest='beta', type=float, default=0.4)
    parser.add_argument('--alpha_step', dest='alpha_step', type=float, default=0.0000)
    parser.add_argument('--beta_step', dest='beta_step', type=float, default=0.00003)
    return parser.parse_args()

def main(args):
    args = parse_arguments()
    environment_name = args.env
    lr = args.lr
    num_trails = 10
    num_step = 20000
    render = args.render
    alpha = args.alpha
    beta = args.beta
    alpha_step = args.alpha_step
    beta_step = args.beta_step

    reward_means_total = []
    max_height_means_total = []
    bumpiness_means_total = []

    for trail in tqdm.tqdm(range(num_trails)):
        reward_means = []
        reward_stds = []
        agent = Agent(environment_name, alpha, beta, alpha_step, beta_step, lr, render=render)
        reward_means, max_height_means, bumpiness_means = agent.train(num_step)
        reward_means_total.append(reward_means)
        max_height_means_total.append(max_height_means)
        bumpiness_means_total.append(bumpiness_means)

    reward_means_total = np.array(reward_means_total)
    max_height_means_total = np.array(max_height_means_total)
    bumpiness_means_total = np.array(bumpiness_means_total)

    reward_mean = reward_means_total.mean(axis=0)
    reward_std = reward_means_total.std(axis=0)
    max_height_mean = max_height_means_total.mean(axis=0)
    max_height_std = max_height_means_total.std(axis=0)
    bumpiness_mean = bumpiness_means_total.mean(axis=0)
    bumpiness_std = bumpiness_means_total.std(axis=0)

    pd.DataFrame(reward_mean).to_csv('./data/' + SAVE_STR + '_reward_mean.csv', header=None, index=None)
    pd.DataFrame(reward_std).to_csv('./data/' + SAVE_STR + '_reward_std.csv', header=None, index=None)
    pd.DataFrame(max_height_mean).to_csv('./data/' + SAVE_STR + '_max_height_mean.csv', header=None, index=None)
    pd.DataFrame(max_height_std).to_csv('./data/' + SAVE_STR + '_max_height_std.csv', header=None, index=None)
    pd.DataFrame(bumpiness_mean).to_csv('./data/' + SAVE_STR + '_bumpiness_mean.csv', header=None, index=None)
    pd.DataFrame(bumpiness_std).to_csv('./data/' + SAVE_STR + '_bumpiness_std.csv', header=None, index=None)

if __name__ == '__main__':
    main(sys.argv)