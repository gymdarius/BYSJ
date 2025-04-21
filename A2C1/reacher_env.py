import numpy as np
import random
import matplotlib.pyplot as plt
import copy
import gymnasium as gym 
from gymnasium import spaces

# ... (ReacherEnv_v0 不变) ...

class ReacherEnv_v1(gym.Env): # 继承 gym.Env
    metadata = {'render_modes': ['human'], "render_fps": 30} # render_modes 和 render_fps 是新版 Gymnasium 的标准

    def __init__(self, n, render_mode=None): # 添加 render_mode
        super().__init__() # 调用父类构造函数
        self.n = n # n is the action number
        # self.action_space = [3]*n # 原始定义
        # SB3 需要明确的 Gym/Gymnasium 空间
        # 每个维度动作是 0, 1, 2 (对应 -1, 0, 1)
        self.action_space = spaces.MultiDiscrete([3] * self.n)

        # self.observation_space = 2*n # 原始定义
        # 观测是 [当前位置, 目标位置]，都是 n 维坐标
        # 假设坐标范围是 0 到 size-1
        self.size = 20
        low_obs = np.zeros(2 * self.n, dtype=np.float32)
        high_obs = np.full(2 * self.n, self.size - 1, dtype=np.float32)
        self.observation_space = spaces.Box(low_obs, high_obs, dtype=np.float32)

        self.direction = np.array([-1, 0, 1])
        # self.size = 20 # 已移到 observation_space 定义之前
        self.target = None
        self.origin = None
        self.position_previous = None
        self.position_current = None # 添加一个变量存储当前位置
        self.done = None
        self.c = None

        assert render_mode is None or render_mode in self.metadata["render_modes"]
        self.render_mode = render_mode # 用于渲染 (如果需要)

        # SB3 通常需要 reset() 返回 observation 和 info
        # self.reset() # 不在 init 中调用 reset

    def step(self, action):
        # action 现在直接是 MultiDiscrete space 的输出 (一个包含 n 个 0/1/2 的 numpy 数组)
        # get the direction
        action_decode = self.direction[action] # action已经是numpy数组，可以直接索引

        # move
        self.position_current = action_decode + self.position_previous # 使用 position_previous

        # use the mahhatton distance for convenience
        distance_previous = np.sum(abs(self.target - self.position_previous))
        distance_current = np.sum(abs(self.target - self.position_current))

        self.c += 1

        # get reward
        reward = (distance_previous - distance_current)

        terminated = False # 使用 Gymnasium 的术语
        truncated = False # 使用 Gymnasium 的术语

        # reach the goal
        if distance_current == 0:
            reward += 100
            terminated = True

        elif self.c >= 40: # 使用 >= 更安全
            # self.done = True # 不再使用 self.done
            truncated = True # 时间限制导致的结束是 truncation

        # move out of the maze:
        elif np.any(self.position_current > self.size - 1) or np.any(self.position_current < 0):
            reward = -10
            # self.done = True # 不再使用 self.done
            terminated = True # 撞墙是 termination

        self.position_previous = self.position_current # 更新上一步位置
        self.ob = np.hstack([self.position_current, self.target]).astype(np.float32) # 确保类型

        info = {} # Gym API 要求返回 info 字典

        # Gym API 返回 observation, reward, terminated, truncated, info
        return self.ob, reward, terminated, truncated, info

    def reset(self, seed=None, options=None):
        super().reset(seed=seed) # 处理随机种子

        while True:
            # 使用 self.np_random 保证可复现性
            self.origin = self.np_random.integers(0, self.size, self.n)
            self.target = self.np_random.integers(0, self.size, self.n)
            distance = np.sum(abs(self.target - self.origin))
            if distance != 0:
                break

        # init the state
        # self.done = False # 不再需要
        self.position_current = self.origin.copy() # 初始化当前位置
        self.position_previous = self.origin.copy() # 初始化上一步位置
        self.ob = np.hstack([self.position_current, self.target]).astype(np.float32) # 确保类型

        # init the count
        self.c = 0

        info = {} # Gym API 要求返回 info 字典

        # Gym API 要求 reset 返回 observation 和 info
        return self.ob, info

    def render(self):
        if self.render_mode == 'human':
            # 在这里添加可视化逻辑，例如使用 matplotlib 或 pygame
            # 显示 self.position_current 和 self.target
            print(f"Step: {self.c}, Pos: {self.position_current}, Target: {self.target}")
            pass # 暂不实现具体渲染

    def close(self):
        # 清理渲染资源等
        pass

    # seed 方法由 gym.Env 基类处理，通常不需要重写，除非有特殊需求
    # def seed(self, seed=None):
    #     pass