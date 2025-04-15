import time
import numpy as np
import os # 需要导入 os
from stable_baselines3 import A2C
from stable_baselines3.common.env_util import make_vec_env
from stable_baselines3.common.vec_env import DummyVecEnv
from stable_baselines3.common.callbacks import CallbackList # 如果需要多个回调
from callbacks import RewardLoggerCallback # 导入我们刚定义的回调类

# 导入环境类
try:
    from reacher_env_gym import ReacherEnvGym
except ImportError:
    print("错误：无法导入 ReacherEnvGym。")
    exit()

# 导入我们刚定义的回调类
# (如果 RewardLoggerCallback 定义在同一个文件中，则无需这行)
# from callbacks import RewardLoggerCallback # 假设保存在 callbacks.py
# 如果定义在当前文件，则直接使用即可

# --- 训练配置 ---
N_DIMENSIONS = 2
N_ENVS = 4 # 可以使用并行环境
TOTAL_TIMESTEPS = 50000
MODEL_SAVE_PATH = f"./saved_models/a2c_reacher_{N_DIMENSIONS}d_logged" # 保存路径
TENSORBOARD_LOG_PATH = "./a2c_reacher_tensorboard/"
REWARD_LOG_DIR = "./training_logs/" # <--- 新增：奖励日志保存目录
REWARD_LOG_FILENAME = f"a2c_{N_DIMENSIONS}d_rewards.csv" # <--- 新增：奖励日志文件名
LOG_REWARD_FREQ = 1000 # <--- 新增：每隔多少 timestep 记录一次奖励

# --- 创建环境 ---
# 使用并行环境可以加速训练
vec_env = make_vec_env(lambda: ReacherEnvGym(n=N_DIMENSIONS), n_envs=N_ENVS)

# --- 定义并实例化回调 ---
# 实例化奖励记录回调
reward_logger_callback = RewardLoggerCallback(
    check_freq=LOG_REWARD_FREQ,
    log_dir=REWARD_LOG_DIR,
    filename=REWARD_LOG_FILENAME,
    verbose=1
)

# 如果你还想用其他回调（比如之前的渲染回调），可以创建一个列表
# render_callback = RenderCallback(...) # 如果需要渲染
# callback_list = CallbackList([reward_logger_callback, render_callback]) # 组合回调
# 否则，只使用奖励记录回调
callback_to_use = reward_logger_callback
# callback_to_use = callback_list # 如果使用了 CallbackList

# --- 定义 A2C 模型 ---
learning_rate_to_try = 7e-4
model = A2C(
    "MlpPolicy",
    vec_env,
    verbose=1,
    gamma=0.99,
    n_steps=8, # 根据你的设置调整
    ent_coef=0.01,
    learning_rate=learning_rate_to_try,
    vf_coef=0.5,
    max_grad_norm=0.5,
    tensorboard_log=TENSORBOARD_LOG_PATH
)

# --- 开始训练 (传入回调) ---
print(f"开始训练 A2C 模型，维度: {N_DIMENSIONS}, 总步数: {TOTAL_TIMESTEPS}")
print(f"奖励日志将每隔 {LOG_REWARD_FREQ} timesteps 记录到 {os.path.join(REWARD_LOG_DIR, REWARD_LOG_FILENAME)}")
try:
    model.learn(
        total_timesteps=TOTAL_TIMESTEPS,
        callback=callback_to_use, # <--- 在这里传入回调
        progress_bar=True
    )
except KeyboardInterrupt:
    print("\n训练被手动中断。")
    # 手动中断时，也尝试保存一下日志 (虽然 _on_training_end 应该也会被调用)
    # reward_logger_callback._on_training_end() # 可以显式调用，但通常不需要

# --- 保存模型 ---
print(f"训练结束（或中断），正在保存模型到 {MODEL_SAVE_PATH}.zip ...")
os.makedirs(os.path.dirname(MODEL_SAVE_PATH), exist_ok=True)
model.save(MODEL_SAVE_PATH)
print("模型已保存。")

# --- 清理 ---
vec_env.close()
print("环境已关闭。")
