import numpy as np
import random
# import matplotlib.pyplot as plt # 如果要实现 render，可能需要
# import copy # 如果环境内部需要深拷贝
import gymnasium as gym
from gymnasium import spaces

class ReacherEnvGym(gym.Env): # 继承 gym.Env
    """修改后的 ReacherEnv_v1，遵循 Gymnasium 接口"""
    metadata = {'render_modes': ['human'], "render_fps": 30} # 添加 render_modes

    def __init__(self, n, render_mode=None): # 添加 render_mode 参数
        super().__init__() # 初始化父类

        self.n = n # n is the number of dimension
        self.size = 20 # the size of the world
        self.max_steps = 40 # 最大步数限制

        # 动作空间: n 个维度，每个维度有 3 个离散动作 (0, 1, 2) -> (-1, 0, 1)
        # 例如 n=2, action_space = MultiDiscrete([3, 3])
        self.action_space = spaces.MultiDiscrete([3] * n)

        # 观察空间: 2*n 维，包括当前位置和目标位置
        # 每个维度的值都在 [0, size-1] 范围内
        # shape 是 (2*n,)
        low = np.zeros(2 * n, dtype=np.float32)
        high = np.full(2 * n, self.size - 1, dtype=np.float32)
        self.observation_space = spaces.Box(low, high, shape=(2 * n,), dtype=np.float32)

        # 内部使用的方向映射
        self._direction_map = np.array([-1, 0, 1])

        # 渲染相关 (如果需要实现 render)
        self.render_mode = render_mode
        self.window = None
        self.clock = None

        # 确保 reset() 会初始化这些状态变量
        self.target = None
        self.position_current = None # 重命名 position_previous 为 position_current
        self._step_count = None

    def _get_obs(self):
        """获取当前观察"""
        return np.hstack([self.position_current, self.target]).astype(np.float32)

    def _get_info(self):
        """获取附加信息 (可选)"""
        distance = np.sum(np.abs(self.target - self.position_current))
        return {"distance_to_target": distance}

    def reset(self, seed=None, options=None):
        """重置环境，符合 Gymnasium 接口"""
        super().reset(seed=seed) # 处理随机种子

        # 随机初始化起点和终点，确保不重合
        while True:
            self.position_current = self.np_random.integers(0, self.size, size=self.n)
            self.target = self.np_random.integers(0, self.size, size=self.n)
            distance = np.sum(np.abs(self.target - self.position_current))
            if distance != 0:
                break

        self._step_count = 0 # 重置步数计数器

        observation = self._get_obs()
        info = self._get_info()

        # Gymnasium 的 reset 返回 (observation, info)
        return observation, info

    def step(self, action):
        """执行一步动作，符合 Gymnasium 接口"""
        # action 是一个包含 n 个整数 (0, 1, 或 2) 的 numpy 数组
        action_decode = self._direction_map[action]

        # 记录移动前的位置
        position_previous = self.position_current
        # 移动
        self.position_current = action_decode + position_previous

        # 计算奖励 (基于曼哈顿距离的缩减)
        distance_previous = np.sum(np.abs(self.target - position_previous))
        distance_current = np.sum(np.abs(self.target - self.position_current))
        reward = float(distance_previous - distance_current) # 确保是 float

        self._step_count += 1

        # 判断是否结束 (terminated or truncated)
        terminated = (distance_current == 0) # 到达目标
        truncated = (self._step_count >= self.max_steps) # 超过最大步数

        # 处理边界条件 (可以视为 truncated，或者给负奖励并 terminated)
        # 这里选择给大负奖励并结束 (terminated)
        is_out_of_bounds = np.any(self.position_current > self.size - 1) or np.any(self.position_current < 0)
        if is_out_of_bounds:
            reward = -10.0 # 移出边界的惩罚
            terminated = True # 也可以设为 truncated = True

        # 到达目标给予额外奖励
        if terminated and distance_current == 0:
            reward += 100.0

        observation = self._get_obs()
        info = self._get_info()

        # Gymnasium 的 step 返回 (observation, reward, terminated, truncated, info)
        return observation, reward, terminated, truncated, info

    def render(self):
        """渲染环境状态 (可选实现)"""
        """if self.render_mode == "human":
            # 在这里添加你的渲染逻辑，例如使用 Pygame 或 Matplotlib
            # 例如，绘制一个 n 维网格，标出当前位置和目标位置
            print(f"Step: {self._step_count}, Position: {self.position_current}, Target: {self.target}, Distance: {info['distance_to_target']}")
            pass # Placeholder
        else:
             # logger.warn("render_mode is not human, cannot render")
             pass
        """
        pass

    def close(self):
        """清理环境资源 (可选实现)"""
        """
        if self.window is not None:
            # 如果使用了 pygame 窗口等，在这里关闭
            # import pygame
            # pygame.display.quit()
            # pygame.quit()
            self.window = None
            self.clock = None
        """
        pass

# (可选) 注册环境，这样就可以通过 ID 字符串创建
# from gymnasium.envs.registration import register
# register(
#     id='ReacherGym-v0',
#     entry_point='__main__:ReacherEnvGym', # 或者 'your_module_name:ReacherEnvGym'
#     kwargs={'n': 2}, # 可以预设参数
#     max_episode_steps=40, # 可选，但建议设置
# )
