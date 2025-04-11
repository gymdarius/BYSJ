import numpy as np
import scipy.sparse as sp

# 加载邻居数据
nei_d = np.load("data/imdb/nei_d.npy", allow_pickle=True)
nei_a = np.load("data/imdb/nei_a.npy", allow_pickle=True)
nei_g = np.load("data/imdb/nei_g.npy", allow_pickle=True)

# 加载邻接矩阵
mdm = sp.load_npz("data/imdb/mdm.npz")
mam = sp.load_npz("data/imdb/mam.npz")
mgm = sp.load_npz("data/imdb/mgm.npz")

# 修复空邻居 - 为没有邻居的节点添加自环
for i in range(len(nei_d)):
    if len(nei_d[i]) == 0:
        # 添加自环到mdm矩阵
        mdm[i, i] = 1
        nei_d[i] = np.array([i])  # 使用自身作为邻居
        
for i in range(len(nei_a)):
    if len(nei_a[i]) == 0:
        # 添加自环到mam矩阵
        mam[i, i] = 1
        nei_a[i] = np.array([i])
        
for i in range(len(nei_g)):
    if len(nei_g[i]) == 0:
        # 添加自环到mgm矩阵
        mgm[i, i] = 1
        nei_g[i] = np.array([i])

# 保存修复后的数据
np.save("data/imdb/nei_d_fixed.npy", nei_d)
np.save("data/imdb/nei_a_fixed.npy", nei_a)
np.save("data/imdb/nei_g_fixed.npy", nei_g)
sp.save_npz("data/imdb/mdm_fixed.npz", mdm)
sp.save_npz("data/imdb/mam_fixed.npz", mam)
sp.save_npz("data/imdb/mgm_fixed.npz", mgm)

print("数据修复完成！")