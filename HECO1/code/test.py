import os
import numpy as np
import scipy.sparse as sp

# IMDB数据集路径
data_path = "../data/imdb/"
print("检查IMDB数据集中各类型节点数量：")

# 加载节点类型数量
if os.path.exists(os.path.join(data_path, "type_num.npy")):
    type_num = np.load(os.path.join(data_path, "type_num.npy"), allow_pickle=True)
    print(f"节点类型数量: {type_num}")
    if len(type_num) >= 4:
        print(f"电影节点数: {type_num[0]}, 导演节点数: {type_num[1]}, 演员节点数: {type_num[2]}, 关键词节点数: {type_num[3]}")

# 检查标签文件
if os.path.exists(os.path.join(data_path, "labels.npy")):
    labels = np.load(os.path.join(data_path, "labels.npy"), allow_pickle=True)
    print(f"标签文件形状: {labels.shape}")
    unique_labels = np.unique(labels)
    print(f"标签中唯一类别数: {len(unique_labels)}")
    print(f"标签值范围: {min(unique_labels)} 到 {max(unique_labels)}")

# 检查边关系文件
edge_types = ["ma.txt", "md.txt", "mw.txt"]
edge_names = ["电影-演员", "电影-导演", "电影-关键词"]

for i, edge_file in enumerate(edge_types):
    file_path = os.path.join(data_path, edge_file)
    if os.path.exists(file_path):
        with open(file_path, 'r') as f:
            lines = f.readlines()
        print(f"{edge_names[i]}关系总数: {len(lines)}")
        
        # 提取唯一的电影ID和另一类型的节点ID
        movie_ids = set()
        other_ids = set()
        for line in lines:
            parts = line.strip().split()
            if len(parts) >= 2:
                movie_ids.add(int(parts[0]))
                other_ids.add(int(parts[1]))
        
        print(f"  唯一电影节点数: {len(movie_ids)}")
        print(f"  唯一{edge_names[i].split('-')[1]}节点数: {len(other_ids)}")

# 检查数据集划分
split_files = ["train_20.npy", "train_40.npy", "train_60.npy", 
               "val_20.npy", "val_40.npy", "val_60.npy",
               "test_20.npy", "test_40.npy", "test_60.npy"]

for split_file in split_files:
    file_path = os.path.join(data_path, split_file)
    if os.path.exists(file_path):
        split_indices = np.load(file_path, allow_pickle=True)
        print(f"{split_file} 包含 {len(split_indices)} 个索引")
        print(f"  索引范围: {min(split_indices)} 到 {max(split_indices)}")