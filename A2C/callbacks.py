import os
import csv # 或者使用 pandas
import numpy as np
from stable_baselines3.common.callbacks import BaseCallback
# import pandas as pd # 如果想用 pandas 保存 CSV

class RewardLoggerCallback(BaseCallback):
    """
    一个自定义回调，用于记录训练过程中的平均奖励并保存到文件。
    """
    def __init__(self, check_freq: int, log_dir: str, filename: str = "rewards_log.csv", verbose: int = 1):
        """
        :param check_freq: 每隔多少个 timestep 记录一次奖励。
        :param log_dir: 保存日志文件的目录。
        :param filename: 保存日志的文件名。
        :param verbose: 详细级别。
        """
        super().__init__(verbose)
        self.check_freq = check_freq
        self.log_dir = log_dir
        self.save_path = os.path.join(log_dir, filename)
        self.rewards_history = [] # 用于存储 (timestep, reward)

    def _init_callback(self) -> None:
        """
        在训练开始时调用。
        """
        # 确保日志目录存在
        if self.log_dir is not None:
            os.makedirs(self.log_dir, exist_ok=True)
            # 可选：如果文件已存在，可以先清空或备份
            # with open(self.save_path, 'w', newline='') as f:
            #     writer = csv.writer(f)
            #     writer.writerow(['timestep', 'ep_rew_mean']) # 写入表头
            if self.verbose > 0:
                print(f"奖励日志将保存在: {self.save_path}")

    def _on_step(self) -> bool:
        """
        在模型执行完一步（或多步，取决于 check_freq）后调用。
        """
        # self.num_timesteps 是 BaseCallback 维护的总步数计数器
        if self.num_timesteps % self.check_freq == 0:
            # 从 logger 获取最新的 rollout/ep_rew_mean 值
            # 这个值是 stable-baselines3 在其内部日志记录周期计算的
            # 可能不是在精确的 self.num_timesteps 这一步计算的，但我们取最近的那个
            if 'rollout/ep_rew_mean' in self.logger.name_to_value:
                reward = self.logger.name_to_value['rollout/ep_rew_mean'].item() # .item() 从 tensor 获取 python number
                self.rewards_history.append((self.num_timesteps, reward))
                if self.verbose > 1:
                    print(f"记录奖励 - Timestep: {self.num_timesteps}, Reward: {reward:.2f}")
            # else:
                # 可能在训练初期还没有计算出第一个 ep_rew_mean
                # if self.verbose > 1:
                #     print(f"Timestep: {self.num_timesteps}, ep_rew_mean 尚未可用")

        # 返回 True 表示继续训练
        return True

    def _on_training_end(self) -> None:
        """
        在训练结束时调用。
        """
        if self.verbose > 0:
            print("训练结束，正在保存奖励日志...")
        try:
            # 使用 csv 模块保存
            with open(self.save_path, 'w', newline='') as f:
                writer = csv.writer(f)
                writer.writerow(['timestep', 'ep_rew_mean']) # 写入表头
                writer.writerows(self.rewards_history) # 写入数据

            # 或者使用 pandas 保存 (需要导入 pandas)
            # import pandas as pd
            # df = pd.DataFrame(self.rewards_history, columns=['timestep', 'ep_rew_mean'])
            # df.to_csv(self.save_path, index=False)

            if self.verbose > 0:
                print(f"奖励日志已成功保存到: {self.save_path}")
        except Exception as e:
            if self.verbose > 0:
                print(f"保存奖励日志时出错: {e}")
