import gymnasium as gym
import numpy as np
from stable_baselines3 import A2C  # 或者 PPO, SAC, DDPG 等，取决于你保存模型时使用的算法
from stable_baselines3.common.env_checker import check_env

# 导入你修改后的环境类
# 确保 reacher_env_gym.py 文件在你的 Python 路径下，或者与 test.py 在同一目录
try:
    from reacher_env_gym import ReacherEnvGym
except ImportError:
    print("错误：无法导入 ReacherEnvGym。")
    print("请确保 reacher_env_gym.py 文件存在于当前目录或 Python 路径中。")
    exit()

# --- 配置参数 ---
# !! 修改为你实际保存的模型路径 !!
MODEL_PATH = "./saved_models/a2c_reacher_2d.zip"
# !! 修改为你模型训练时使用的环境维度 !!
ENV_DIMENSIONS = 2
# 设置测试运行的回合数
N_TEST_EPISODES = 20
# 是否尝试渲染环境 (如果环境的 render 方法已实现)
# 设置为 "human" 来可视化, 设置为 None 则不进行可视化
RENDER_MODE = None # 或者 "human"

# --- 加载模型 ---
print(f"正在加载模型: {MODEL_PATH}")
try:
    # !! 确保这里的 A2C 与你保存模型时使用的算法一致 !!
    model = A2C.load(MODEL_PATH)
    print("模型加载成功。")
except Exception as e:
    print(f"加载模型时出错: {e}")
    print("请确认模型文件路径正确，并且使用了正确的算法类 (A2C, PPO 等)。")
    exit()

# --- 创建测试环境 ---
print(f"创建 {ENV_DIMENSIONS} 维 Reacher 测试环境...")
try:
    # 注意：测试时通常只创建一个环境实例
    env = ReacherEnvGym(n=ENV_DIMENSIONS, render_mode=RENDER_MODE)
    # (可选) 检查环境是否符合 Stable Baselines 规范
    # check_env(env)
except Exception as e:
    print(f"创建环境时出错: {e}")
    exit()

# --- 运行测试 ---
print(f"开始运行 {N_TEST_EPISODES} 个测试回合...")
episode_rewards = []
episode_lengths = []

for i in range(N_TEST_EPISODES):
    obs, info = env.reset()
    terminated = False
    truncated = False
    current_episode_reward = 0
    current_episode_length = 0

    while not terminated and not truncated:
        # 使用确定性策略进行预测 (deterministic=True)
        action, _states = model.predict(obs, deterministic=True)

        obs, reward, terminated, truncated, info = env.step(action)
        current_episode_reward += reward
        current_episode_length += 1

        if RENDER_MODE == "human":
            try:
                env.render()
            except NotImplementedError:
                if i == 0 and current_episode_length == 1: # 只在第一个回合的第一步提示一次
                    print("提示: 环境的 render() 方法未实现或为 pass，无法进行可视化。")
                RENDER_MODE = None # 避免后续回合重复尝试渲染和打印提示
            except Exception as e:
                 if i == 0 and current_episode_length == 1:
                    print(f"渲染时出错: {e}")
                 RENDER_MODE = None

    print(f"回合 {i+1}/{N_TEST_EPISODES} 完成 - 奖励: {current_episode_reward:.2f}, 步数: {current_episode_length}")
    episode_rewards.append(current_episode_reward)
    episode_lengths.append(current_episode_length)

# --- 输出结果 ---
mean_reward = np.mean(episode_rewards)
std_reward = np.std(episode_rewards)
mean_length = np.mean(episode_lengths)
std_length = np.std(episode_lengths)

print("\n--- 测试结果 ---")
print(f"总回合数: {N_TEST_EPISODES}")
print(f"平均奖励: {mean_reward:.2f} +/- {std_reward:.2f}")
print(f"平均步数: {mean_length:.2f} +/- {std_length:.2f}")

# --- 清理 ---
env.close()
print("测试完成。")
