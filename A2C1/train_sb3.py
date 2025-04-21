import gymnasium as gym # 或者 import gym
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import torch # SB3 默认使用 PyTorch
import os
import tqdm
import argparse
from stable_baselines3 import PPO, A2C # 按需导入你想用的算法
from stable_baselines3.common.env_util import make_vec_env
from stable_baselines3.common.callbacks import BaseCallback, EvalCallback
from stable_baselines3.common.monitor import Monitor # 用于包装环境以记录 episode 统计信息
from stable_baselines3.common.vec_env import DummyVecEnv, SubprocVecEnv

# 导入修改后的环境
from reacher_env import ReacherEnv_v1

# --- 辅助函数：用于评估并返回均值和标准差 ---
def evaluate_policy_with_std(model, env, n_eval_episodes=20, deterministic=True):
    """
    评估策略并返回平均奖励和标准差。
    :param model: (BasePolicy) The policy to evaluate.
    :param env: (Env) The evaluation environment.
    :param n_eval_episodes: (int) Number of episodes to evaluate the agent.
    :param deterministic: (bool) Whether to use deterministic or stochastic actions.
    :return: (float, float) Mean reward per episode, Standard Deviation of reward per episode.
    """
    episode_rewards = []
    for _ in range(n_eval_episodes):
        obs, _ = env.reset()
        done = False
        episode_reward = 0.0
        while not done:
            action, _ = model.predict(obs, deterministic=deterministic)
            obs, reward, terminated, truncated, _ = env.step(action)
            done = terminated or truncated
            episode_reward += reward
        episode_rewards.append(episode_reward)

    mean_reward = np.mean(episode_rewards)
    std_reward = np.std(episode_rewards)
    return mean_reward, std_reward

# --- 自定义回调：用于记录均值/标准差，打印日志，并触发评估 ---
class CustomEvalCallback(BaseCallback):
    """
    自定义回调，用于周期性评估、记录均值/标准差、打印和保存最佳模型。
    """
    def __init__(self, eval_env, n_eval_episodes=20, eval_freq=10000, log_path=None,
                 best_model_save_path=None, deterministic=True, verbose=1,
                 reward_means_list=None, reward_stds_list=None):
        super(CustomEvalCallback, self).__init__(verbose)
        self.eval_env = eval_env
        # SB3 EvalCallback 通常在 VecEnv 上评估，如果 eval_env 不是 VecEnv，需要包装
        if not isinstance(self.eval_env, (DummyVecEnv, SubprocVecEnv)):
             self.eval_env = DummyVecEnv([lambda: self.eval_env])

        self.n_eval_episodes = n_eval_episodes
        self.eval_freq = eval_freq
        self.best_mean_reward = -np.inf
        self.last_mean_reward = -np.inf
        self.log_path = log_path
        self.best_model_save_path = best_model_save_path
        self.deterministic = deterministic
        self.reward_means_list = reward_means_list if reward_means_list is not None else []
        self.reward_stds_list = reward_stds_list if reward_stds_list is not None else []
        self.best_model_saved_at_step = 0

        # 用于保存最佳模型的文件名（不含扩展名）
        if self.best_model_save_path is not None:
            self.save_path_base = os.path.join(self.best_model_save_path, "best_model")
            os.makedirs(self.best_model_save_path, exist_ok=True)
        else:
            self.save_path_base = None


    def _on_step(self) -> bool:
        # self.num_timesteps 是当前总步数
        if self.eval_freq > 0 and self.num_timesteps % self.eval_freq == 0:
            # --- 执行评估 ---
            # 注意：SB3 的 model.predict 在 VecEnv 上会自动处理多个环境
            # 但我们的 evaluate_policy_with_std 需要单个环境接口
            # 因此，如果 eval_env 是 VecEnv，我们需要获取其内部的单个环境或修改评估函数
            # 这里假设 evaluate_policy_with_std 能处理传入的 eval_env (可能是 VecEnv)
            # 或者，更简单的方式是用 SB3 内置的 evaluate_policy，但它只返回均值
            # from stable_baselines3.common.evaluation import evaluate_policy
            # mean_reward, std_reward = evaluate_policy(self.model, self.eval_env, n_eval_episodes=self.n_eval_episodes, deterministic=self.deterministic, return_episode_rewards=False) # std_reward 需要 return_episode_rewards=True 再计算

            # 使用我们的辅助函数
            # 需要确保 eval_env 是单个环境实例传递给它
            sync_envs_possible = isinstance(self.eval_env, DummyVecEnv)
            if sync_envs_possible:
                mean_reward, std_reward = evaluate_policy_with_std(self.model, self.eval_env.envs[0], self.n_eval_episodes, self.deterministic)
            else:
                # 对于 SubprocVecEnv，评估可能更复杂，或者只用一个 DummyVecEnv 作评估环境
                print("Warning: Evaluation with SubprocVecEnv might require different handling. Using first env.")
                # 尝试在第一个子进程环境上评估（可能不准确或慢）
                # 或者最好在创建回调时就传入一个非向量化的评估环境
                mean_reward, std_reward = evaluate_policy_with_std(self.model, self.eval_env.envs[0], self.n_eval_episodes, self.deterministic)


            if self.log_path is not None:
                # 记录到 TensorBoard (如果配置了 logger)
                self.logger.record("eval/mean_reward", mean_reward)
                self.logger.record("eval/std_reward", std_reward)
                self.logger.dump(self.num_timesteps)

            # --- 打印日志 (模拟原始输出) ---
            # 估算当前是第几个 "episode block" (每10个原始episode)
            # 注意：这里的 eval_freq 是步数，原始是 episode 数，需要转换或调整逻辑
            # 为了简单起见，我们每次评估都打印
            print(f"Step: {self.num_timesteps}")
            print(f"  Eval: Mean Reward: {mean_reward:.1f} +/- {std_reward:.1f}")
            # 打印 Epsilon (对于 PPO/A2C 等算法，没有直接对应的 epsilon，可以省略或打印学习率)
            # print(f"  Epsilon: ...") # 对于 PPO/A2C 通常不适用

            # --- 记录结果 ---
            self.reward_means_list.append(mean_reward)
            self.reward_stds_list.append(std_reward)
            self.last_mean_reward = mean_reward

            # --- 保存最佳模型 ---
            if mean_reward > self.best_mean_reward:
                if self.verbose > 0:
                    print(f"New best mean reward! Previous: {self.best_mean_reward:.1f}, Current: {mean_reward:.1f}. Saving best model to {self.save_path_base}.zip")
                self.best_mean_reward = mean_reward
                if self.save_path_base is not None:
                    self.model.save(self.save_path_base)
                    self.best_model_saved_at_step = self.num_timesteps

        return True

# --- 主函数 ---
def parse_arguments():
    parser = argparse.ArgumentParser(description='Stable Baselines3 Reacher Training')
    parser.add_argument('--env', dest='env_dim', type=int, default=4, help="Dimension of the Reacher environment")
    # parser.add_argument('--render', dest='render', type=int, default=0) # 渲染可以在创建环境时指定
    parser.add_argument('--algo', dest='algo', type=str, default='PPO', choices=['PPO', 'A2C'], help="RL Algorithm")
    parser.add_argument('--lr', dest='lr', type=float, default=3e-4, help="Learning rate") # SB3 PPO/A2C 默认 3e-4
    parser.add_argument('--trials', dest='num_trials', type=int, default=10, help="Number of training trials")
    parser.add_argument('--total_steps', dest='total_timesteps', type=int, default=1_000_000, help="Total training steps per trial") # SB3 按步数训练
    parser.add_argument('--eval_freq_steps', dest='eval_freq', type=int, default=1000, help="Evaluate the agent every N steps") # 评估频率（步数）
    parser.add_argument('--n_eval_episodes', dest='n_eval_episodes', type=int, default=20, help="Number of episodes for evaluation")
    parser.add_argument('--seed', dest='seed', type=int, default=42, help="Random seed")
    parser.add_argument('--log_dir', dest='log_dir', type=str, default='./logs_sb3/', help="Directory for logs and models")
    parser.add_argument('--data_dir', dest='data_dir', type=str, default='./data_sb3/', help="Directory for CSV results")
    parser.add_argument('--plot_dir', dest='plot_dir', type=str, default='./plots_sb3/', help="Directory for reward plots")
    return parser.parse_args()

def main():
    args = parse_arguments()

    # --- 确保目录存在 ---
    os.makedirs(args.log_dir, exist_ok=True)
    os.makedirs(args.data_dir, exist_ok=True)
    os.makedirs(args.plot_dir, exist_ok=True)

    # --- 设置随机种子 ---
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    # if torch.cuda.is_available():
    #     torch.cuda.manual_seed_all(args.seed) # 如果使用 GPU

    reward_means_total = []
    reward_stds_total = []
    trial_eval_steps = [] # 记录每次评估对应的训练步数 (用于绘图x轴)

    # --- 试验循环 ---
    for trial in tqdm.tqdm(range(args.num_trials), desc="Trials"):
        trial_seed = args.seed + trial # 每个 trial 用不同种子
        trial_log_path = os.path.join(args.log_dir, f"{args.algo}-{args.env_dim}d_trial{trial}")
        trial_best_model_path = os.path.join(trial_log_path, "best_model") # SB3 EvalCallback 习惯保存在 log 路径下

        # --- 创建环境 ---
        # 使用 Monitor 包装器来自动记录 episode 奖励、长度等信息，SB3 会自动使用
        # 对于评估，我们通常用一个单独的、未包装或只用 Monitor 包装的环境
        # 注意：ReacherEnv_v1 现在需要 n 作为参数
        def make_env():
            env = ReacherEnv_v1(n=args.env_dim)
            env.reset(seed=trial_seed) # 设置环境种子
            env = Monitor(env) # Monitor 用于训练过程日志
            return env

        # 创建训练环境 (可以是向量化的)
        # train_env = make_vec_env(make_env, n_envs=4, seed=trial_seed, vec_env_cls=SubprocVecEnv) # 示例：使用4个并行环境
        train_env = make_env() # 示例：使用单个环境

        # 创建评估环境 (单个，非向量化，用于 CustomEvalCallback)
        eval_env = ReacherEnv_v1(n=args.env_dim)
        eval_env.reset(seed=trial_seed + 1000) # 使用不同种子进行评估


        # --- 设置回调 ---
        reward_means_trial = []
        reward_stds_trial = []
        eval_callback = CustomEvalCallback(eval_env=eval_env,
                                           n_eval_episodes=args.n_eval_episodes,
                                           eval_freq=args.eval_freq,
                                           log_path=trial_log_path, # Tensorboard 日志路径
                                           best_model_save_path=trial_best_model_path, # 最佳模型保存路径
                                           deterministic=True,
                                           verbose=1,
                                           reward_means_list=reward_means_trial,
                                           reward_stds_list=reward_stds_trial)

        # --- 选择并创建模型 ---
        if args.algo == 'PPO':
            # PPO 适用于 MultiDiscrete
            model = PPO("MlpPolicy", train_env, learning_rate=args.lr, verbose=0, seed=trial_seed,
                        tensorboard_log=args.log_dir) # 指定 Tensorboard 日志目录
        elif args.algo == 'A2C':
            # A2C 也适用于 MultiDiscrete
            model = A2C("MlpPolicy", train_env, learning_rate=args.lr, verbose=0, seed=trial_seed,
                        tensorboard_log=args.log_dir)
        else:
            raise ValueError(f"Algorithm {args.algo} not supported yet.")

        # --- 训练模型 ---
        print(f"\n--- Starting Trial {trial+1}/{args.num_trials} ---")
        model.learn(total_timesteps=args.total_timesteps, callback=eval_callback,
                    tb_log_name=f"{args.algo}-{args.env_dim}d_trial{trial}") # Tensorboard 运行名称

        # --- 收集结果 ---
        reward_means_total.append(reward_means_trial)
        reward_stds_total.append(reward_stds_trial)
        # 记录这次 trial 评估发生的步数 (假设所有 trial 的 eval_freq 相同)
        if not trial_eval_steps:
             trial_eval_steps = list(range(args.eval_freq, args.total_timesteps + 1, args.eval_freq))


        # 清理环境 (如果需要)
        train_env.close()
        eval_env.close()
        print(f"--- Finished Trial {trial+1}/{args.num_trials} ---")


    # --- 处理结果 (填充/对齐不同 trial 可能不同的评估次数) ---
    # 找到最长的评估记录长度
    max_len = 0
    if reward_means_total:
         max_len = max(len(lst) for lst in reward_means_total)

    # 用最后一个有效值填充较短的列表
    aligned_means = []
    aligned_stds = []
    for means, stds in zip(reward_means_total, reward_stds_total):
        last_mean = means[-1] if means else np.nan
        last_std = stds[-1] if stds else np.nan
        aligned_means.append(means + [last_mean] * (max_len - len(means)))
        aligned_stds.append(stds + [last_std] * (max_len - len(stds)))

    # --- 计算平均值 ---
    avg_means = np.nanmean(aligned_means, axis=0)
    avg_stds = np.nanmean(aligned_stds, axis=0) # 注意：直接平均标准差可能不太准确，但符合原始逻辑

    # 调整 x 轴步数以匹配数据长度
    eval_steps_x_axis = trial_eval_steps[:len(avg_means)]


    # --- 保存 CSV ---
    save_str = f'{args.algo}-{args.env_dim}d'
    mean_csv_path = os.path.join(args.data_dir, f'{save_str}_mean.csv')
    std_csv_path = os.path.join(args.data_dir, f'{save_str}_std.csv')
    pd.DataFrame(avg_means).to_csv(mean_csv_path, header=None, index=None)
    pd.DataFrame(avg_stds).to_csv(std_csv_path, header=None, index=None)
    print(f"\nSaved average results to {mean_csv_path} and {std_csv_path}")

    # --- 绘制并保存绘图 ---
    plt.figure(figsize=(8, 6))
    plt.plot(eval_steps_x_axis, avg_means, label=f'{args.algo} {args.env_dim}D Mean Reward')
    plt.fill_between(eval_steps_x_axis, avg_means - avg_stds, avg_means + avg_stds, alpha=0.2, label=f'{args.algo} {args.env_dim}D Std Dev')
    plt.xlabel("Training Steps")
    plt.ylabel("Average Reward")
    plt.title(f'{args.algo} Learning Curve ({args.env_dim}D Reacher)')
    plt.legend()
    plt.grid(True)
    plot_path = os.path.join(args.plot_dir, f'reward_{save_str}.png')
    plt.savefig(plot_path)
    print(f"Saved reward plot to {plot_path}")
    # plt.show() # 取消注释以显示绘图

if __name__ == '__main__':
    main()