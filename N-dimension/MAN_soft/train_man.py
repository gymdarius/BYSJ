import numpy as np
import torch
from reacher.reacher_env import ReacherEnv_v1
import matplotlib.pyplot as plt
import sys
import tqdm
import argparse
from man import Replay_Memory
from man import MAN
import math
import pandas as pd
import torch.nn.functional as F

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

class MAN_Agent():
    def __init__(self, num_dimension, lr, render=False, tau = 0.005):
        self.n = num_dimension
        self.env = ReacherEnv_v1(self.n)
        self.replay_memory = Replay_Memory()
        self.agent = MAN(self.env, lr)

        # Move networks to device
        for net in self.agent.evaluate_nets:
            net.to(device)
        for net in self.agent.target_nets:
            net.to(device)

        self.update = 1000
        self.epsilon_start = 1
        self.epsilon_end = 0.1
        self.epsilon_decay = 10000
        self.gamma = 0.99
        self.batch_size = 32
        self.render = render
        self.c = 0
        self.direction = torch.tensor([-1, 0, 1]).to(device)
        self.tau = tau
        self.burn_in_memory()

    def epsilon_greedy_policy(self, ob, train=True):
        if train:
            #epsilon = self.epsilon_end + (self.epsilon_start - self.epsilon_end) * math.exp(-1. * self.c / self.epsilon_decay)
            steps = min(self.c, self.epsilon_decay)
            epsilon = self.epsilon_start - (self.epsilon_start - self.epsilon_end) * (steps / self.epsilon_decay)
        else:
            epsilon = 0.05
        if np.random.rand() >= epsilon:
            return self.greedy_policy_Q(ob)
        else:
            return np.random.choice(self.env.action_space[0], self.env.n)

    def greedy_policy_Q(self, ob):
        ob = torch.tensor(ob, dtype=torch.float32).to(device)
        actions = np.array([torch.argmax(self.agent.evaluate_nets[0](ob).detach()).view(-1).item()])
        for i in range(1, self.env.n):
            ob = torch.cat((ob, torch.tensor(actions[-1], dtype=torch.float32).view(1).to(device)))
            actions = np.hstack((actions, torch.argmax(self.agent.evaluate_nets[i](ob).detach()).view(-1).item()))
        return actions

    def train(self):
        state = self.env.reset()

        while True:
            action = self.greedy_policy_Q(state)
            next_state, reward, done, _ = self.env.step(action)

            transition = [
                torch.tensor(state).view(1, -1).float().to(device),
                torch.tensor(action).view(1, -1).long().to(device),
                torch.tensor(reward).view(1).float().to(device),
                torch.tensor(next_state).view(1, -1).float().to(device),
                done
            ]

            self.replay_memory.append(transition)
            mini_batch = self.replay_memory.sample_batch(self.batch_size)

            losses = []
            for i in range(self.env.n):
                j = (i + 1) % self.env.n

                y = torch.tensor([]).float().to(device)
                v_actions = torch.tensor([]).long().to(device)
                actions = torch.tensor([]).long().to(device)
                u_states = torch.tensor([]).float().to(device)
                u_states_next = torch.tensor([]).float().to(device)

                for transition in mini_batch:
                    sample_state = transition[0].to(device)
                    sample_action_index = transition[1].to(device)
                    sample_action = self.direction[sample_action_index].float()
                    sample_reward = transition[2].to(device)
                    sample_next_state = transition[3].to(device)
                    sample_done = transition[-1]

                    u_state_next = torch.cat([sample_next_state, sample_action[:, :j].float()], dim=1).to(device)
                    u_state = torch.cat([sample_state, sample_action[:, :i].float()], dim=1).to(device)
                    u_states = torch.cat([u_states, u_state])
                    v_action = sample_action[:, i].long().to(device)
                    v_actions = torch.cat([v_actions, v_action])
                    action = sample_action_index[:, i].to(device)
                    actions = torch.cat([actions, action])

                    if sample_done:
                        y = torch.cat((y, sample_reward))
                    else:
                        y = torch.cat([y, sample_reward + self.gamma * torch.max(self.agent.target_nets[j](u_state_next).detach())])

                q = torch.gather(self.agent.evaluate_nets[i](u_states), 1, actions.view(-1, 1))
                losses.append(F.mse_loss(y.view(-1, 1), q))

            for i in range(self.env.n):
                self.agent.optimizers[i].zero_grad()
                losses[i].backward()
                self.agent.optimizers[i].step()

            self.c += 1
            #if self.c % self.update == 0:
            #    path = self.agent.save_model_weights()
            #    self.agent.load_model(path)
            # 替换为
            #self.c += 1
            # 每次训练后都进行软更新
            self.agent.soft_update(self.tau)
    
            # 可以选择性地保留周期性保存模型
            if self.c % self.update == 0:
                self.agent.save_model_weights()
            if not done:
                state = next_state
            else:
                break

    def test(self, model_file=None):
        with torch.no_grad():
            test_trial = 20
            total_rewards = []
            for _ in range(test_trial):
                state = self.env.reset()
                rewards = 0
                while True:
                    if self.render:
                        self.env.render()
                    action = self.epsilon_greedy_policy(state, train=False)
                    next_state, reward, done, _ = self.env.step(action)
                    rewards += reward
                    state = next_state
                    if done:
                        break
                total_rewards.append(rewards)

            total_rewards = np.array(total_rewards)
            reward_mean = np.mean(total_rewards)
            reward_std = np.sqrt(np.mean(np.sum((total_rewards - reward_mean) ** 2) / test_trial))

        return reward_mean, reward_std

    def burn_in_memory(self):
        num = 0
        current_state = self.env.reset()
        while num < self.replay_memory.burn_in:
            action = np.random.choice(self.env.action_space[0], self.env.n)
            next_state, reward, done, _ = self.env.step(action)

            transition = [
                torch.tensor(current_state).view(1, -1).float().to(device),
                torch.tensor(action).view(1, -1).long().to(device),
                torch.tensor(reward).view(1).float().to(device),
                torch.tensor(next_state).view(1, -1).float().to(device),
                done
            ]

            self.replay_memory.append(transition)
            num += 1
            if done:
                current_state = self.env.reset()
            else:
                current_state = next_state

def parse_arguments():
    parser = argparse.ArgumentParser(description='Multi Action Network Argument Parser')
    parser.add_argument('--env', dest='env', type=int, default="4")
    parser.add_argument('--render', dest='render', type=int, default=0)
    parser.add_argument('--train', dest='train', type=int, default=1)
    parser.add_argument('--model', dest='model_file', type=str)
    parser.add_argument('--lr', dest='lr', type=float, default=1e-3)
    parser.add_argument('--tau', dest='tau', type=float, default=0.005)  # 添加tau参数
    return parser.parse_args()

def main(args):
    args = parse_arguments()
    num_dimension = args.env
    lr = args.lr
    render = args.render
    tau = args.tau
    num_trails = 10
    num_episodes = 1000
    SAVE_STR = 'MAN-' + str(num_dimension) + 'd'

    reward_means_total = []
    reward_stds_total = []

    for trail in tqdm.tqdm(range(num_trails)):
        reward_means = []
        reward_stds = []
        agent = MAN_Agent(num_dimension, lr, render=render, tau=tau)
        for epi in range(num_episodes):
            if epi % 10 == 0:
                reward_mean, reward_std = agent.test()
                reward_means.append(reward_mean)
                reward_stds.append(reward_std)
                print("The test reward for episode %d is %.1f." % (epi, reward_means[-1]))
                print('The epsilon is:', agent.epsilon_end + (agent.epsilon_start - agent.epsilon_end) * math.exp(-1. * agent.c / agent.epsilon_decay))

                if reward_means[-1] == max(reward_means) or epi % 100 == 0:
                    path = agent.agent.save_best_model()
                    print("The best model is saved with reward %.1f." % (reward_means[-1]))
            agent.train()

        reward_means_total.append(reward_means)
        reward_stds_total.append(reward_stds)

    pd.DataFrame(np.array(reward_means_total).mean(axis=0)).to_csv('./data/' + SAVE_STR + '_mean.csv', header=None, index=None)
    pd.DataFrame(np.array(reward_stds_total).mean(axis=0)).to_csv('./data/' + SAVE_STR + '_std.csv', header=None, index=None)

    plt.plot(np.array(reward_means_total).mean(axis=0))
    plt.savefig("./plots/reward_" + SAVE_STR)

if __name__ == '__main__':
    main(sys.argv)
