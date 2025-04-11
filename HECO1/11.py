import numpy as np

# 加载邻居数据
nei_d = np.load("data/imdb/nei_d_fixed.npy", allow_pickle=True)
nei_a = np.load("data/imdb/nei_a_fixed.npy", allow_pickle=True)
nei_g = np.load("data/imdb/nei_g_fixed.npy", allow_pickle=True)

# 检查空邻居
empty_d = [i for i, x in enumerate(nei_d) if len(x) == 0]
empty_a = [i for i, x in enumerate(nei_a) if len(x) == 0]
empty_g = [i for i, x in enumerate(nei_g) if len(x) == 0]

print(f"没有导演的电影数: {len(empty_d)}")
print(f"没有演员的电影数: {len(empty_a)}")
print(f"没有类型的电影数: {len(empty_g)}")