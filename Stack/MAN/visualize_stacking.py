import numpy as np
import torch
import gym
import stack
import matplotlib.pyplot as plt
import os
from PIL import Image
import argparse
from MAN import DQN, DVN

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# 高度信息，与训练代码保持一致
heights = torch.tensor([[0, 0.025, 0], [0.025, 0.025, 0.025], [0.05, 0.05, 0.05], [0.075, 0.075, 0.075]]).to(device)

def parse_args():
    parser = argparse.ArgumentParser(description='Stack Task Visualization')
    parser.add_argument('--env', type=str, default='Stack-v0', help='Environment name')
    parser.add_argument('--model_path', type=str, default='./model', help='Path to saved models')
    parser.add_argument('--episodes', type=int, default=1, help='Number of episodes to visualize')
    parser.add_argument('--blocks_per_episode', type=int, default=0, 
                        help='Number of blocks to place per episode (0 means continue until environment terminates)')
    parser.add_argument('--output_dir', type=str, default='./visualization', help='Directory to save visualizations')
    return parser.parse_args()

def visualize_stacking(env_name, model_path, num_episodes=5, blocks_per_episode=0, output_dir='./visualization'):
    """
    使用保存的模型可视化堆叠任务
    
    参数:
        env_name: 环境名称
        model_path: 模型路径
        num_episodes: 要可视化的轮数
        blocks_per_episode: 每轮放置的块数量 (0表示一直运行到环境结束)
        output_dir: 输出目录
    """
    # 创建环境
    env = gym.make(env_name)
    
    # 创建并加载模型
    value_net = DVN(env, lr=3e-4)
    q_net = DQN(env, lr=3e-4)
    
    # 加载模型权重
    value_net.load_model(os.path.join(model_path, "best_model_V_v2"))
    q_net.load_model(os.path.join(model_path, "best_model_Q_v2"))
    
    # 设置模型为评估模式
    value_net.model.eval()
    q_net.model.eval()
    
    # 创建输出目录
    os.makedirs(output_dir, exist_ok=True)
    
    for episode in range(num_episodes):
        print(f"Episode {episode+1}/{num_episodes}")
        episode_dir = os.path.join(output_dir, f"episode_{episode+1}")
        os.makedirs(episode_dir, exist_ok=True)
        
        # 重置环境
        state = torch.tensor(env.reset()).view(1, -1).float().to(device)
        step = 0
        done = False
        
        # 保存初始状态渲染
        img = env.render(mode='rgb_array')
        plt.figure(figsize=(10, 10))
        plt.imshow(img)
        plt.title(f"初始状态")
        plt.axis('off')
        plt.savefig(os.path.join(episode_dir, f"step_0.png"))
        plt.close()
        
        # 设置块数限制条件
        block_limit_reached = False
        
        while not done and not block_limit_reached:
            with torch.no_grad():
                # 选择方块类型
                v_values = value_net.model(state)
                sorted_v, indices = torch.sort(v_values, descending=True)
                for index in indices.tolist()[0]:
                    if index in env.cube_pool:
                        action_I = index
                        break
                
                # 选择放置位置
                q_values = q_net.model(torch.cat((state, heights[action_I].view(1, -1)), dim=1))
                action_II = torch.argmax(q_values).item()
            
            # 执行动作
            next_state, reward, done, _ = env.step([action_I, action_II])
            next_state = torch.tensor(next_state).view(1, -1).float().to(device)
            
            # 渲染并保存图像
            img = env.render(mode='rgb_array')
            step += 1
            
            plt.figure(figsize=(10, 10))
            plt.imshow(img)
            plt.title(f"步骤 {step} - 方块类型: {action_I}, 位置: {action_II}, 奖励: {reward:.2f}")
            plt.axis('off')
            plt.savefig(os.path.join(episode_dir, f"step_{step}.png"))
            plt.close()
            
            # 打印当前步骤信息
            print(f"  Step {step}: 选择方块类型 {action_I}, 位置 {action_II}, 奖励 {reward:.2f}")
            
            # 更新状态
            state = next_state
            
            # 检查是否达到指定的块数限制
            if blocks_per_episode > 0 and step >= blocks_per_episode:
                block_limit_reached = True
                print(f"  达到指定块数限制: {blocks_per_episode}")
            
            # 如果完成或达到块数限制，保存最终状态
            if done or block_limit_reached:
                print(f"  Episode 完成! 总步数: {step}, 原因: {'环境终止' if done else '达到块数限制'}")
                
                # 计算一些指标
                max_height = torch.max(next_state.cpu()).item()
                bumpiness = 0
                for i in range(next_state.size(1) - 1):
                    bumpiness += torch.abs(next_state[0, i + 1] - next_state[0, i])
                bumpiness = bumpiness.cpu().item()
                
                # 保存最终结果信息
                plt.figure(figsize=(10, 10))
                plt.imshow(img)
                plt.title(f"最终状态 - 最大高度: {max_height:.2f}, 凹凸度: {bumpiness:.2f}")
                plt.axis('off')
                plt.savefig(os.path.join(episode_dir, f"final_state.png"))
                plt.close()
                
                # 创建GIF动画
                frames = []
                for i in range(step + 1):
                    image = Image.open(os.path.join(episode_dir, f"step_{i}.png"))
                    frames.append(image)
                
                # 保存为GIF，每帧持续0.5秒
                frames[0].save(
                    os.path.join(episode_dir, "stacking_animation.gif"),
                    save_all=True,
                    append_images=frames[1:],
                    optimize=False,
                    duration=500,
                    loop=0
                )

def main():
    args = parse_args()
    visualize_stacking(
        env_name=args.env,
        model_path=args.model_path,
        num_episodes=args.episodes,
        blocks_per_episode=args.blocks_per_episode,
        output_dir=args.output_dir
    )
    print(f"可视化完成! 结果保存在 {args.output_dir} 目录")

if __name__ == "__main__":
    main()