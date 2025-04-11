import numpy as np
import scipy.sparse as sp
import torch as th
import os
from sklearn.preprocessing import OneHotEncoder


def encode_onehot(labels, n_classes=None):
    """将标签转换为独热编码，可选指定类别数"""
    labels = labels.reshape(-1, 1)
    
    if n_classes is None:
        enc = OneHotEncoder()
        enc.fit(labels)
    else:
        enc = OneHotEncoder(categories=[range(n_classes)])
    
    labels_onehot = enc.transform(labels).toarray()
    return labels_onehot


def preprocess_features(features):
    """Row-normalize feature matrix and convert to tuple representation"""
    rowsum = np.array(features.sum(1))
    r_inv = np.power(rowsum, -1).flatten()
    r_inv[np.isinf(r_inv)] = 0.
    r_mat_inv = sp.diags(r_inv)
    features = r_mat_inv.dot(features)
    return features.todense()


def normalize_adj(adj):
    """Symmetrically normalize adjacency matrix."""
    adj = sp.coo_matrix(adj)
    rowsum = np.array(adj.sum(1))
    d_inv_sqrt = np.power(rowsum, -0.5).flatten()
    d_inv_sqrt[np.isinf(d_inv_sqrt)] = 0.
    d_mat_inv_sqrt = sp.diags(d_inv_sqrt)
    return adj.dot(d_mat_inv_sqrt).transpose().dot(d_mat_inv_sqrt).tocoo()


def sparse_mx_to_torch_sparse_tensor(sparse_mx):
    """Convert a scipy sparse matrix to a torch sparse tensor."""
    sparse_mx = sparse_mx.tocoo().astype(np.float32)
    indices = th.from_numpy(
        np.vstack((sparse_mx.row, sparse_mx.col)).astype(np.int64))
    values = th.from_numpy(sparse_mx.data)
    shape = th.Size(sparse_mx.shape)
    return th.sparse.FloatTensor(indices, values, shape)


def load_acm(ratio, type_num):
    # The order of node types: 0 p 1 a 2 s
    path = "../data/acm/"
    label = np.load(path + "labels.npy").astype('int32')
    label = encode_onehot(label)
    nei_a = np.load(path + "nei_a.npy", allow_pickle=True)
    nei_s = np.load(path + "nei_s.npy", allow_pickle=True)
    feat_p = sp.load_npz(path + "p_feat.npz")
    feat_a = sp.eye(type_num[1])
    feat_s = sp.eye(type_num[2])
    pap = sp.load_npz(path + "pap.npz")
    psp = sp.load_npz(path + "psp.npz")
    pos = sp.load_npz(path + "pos.npz")
    train = [np.load(path + "train_" + str(i) + ".npy") for i in ratio]
    test = [np.load(path + "test_" + str(i) + ".npy") for i in ratio]
    val = [np.load(path + "val_" + str(i) + ".npy") for i in ratio]

    label = th.FloatTensor(label)
    nei_a = [th.LongTensor(i) for i in nei_a]
    nei_s = [th.LongTensor(i) for i in nei_s]
    feat_p = th.FloatTensor(preprocess_features(feat_p))
    feat_a = th.FloatTensor(preprocess_features(feat_a))
    feat_s = th.FloatTensor(preprocess_features(feat_s))
    pap = sparse_mx_to_torch_sparse_tensor(normalize_adj(pap))
    psp = sparse_mx_to_torch_sparse_tensor(normalize_adj(psp))
    pos = sparse_mx_to_torch_sparse_tensor(pos)
    train = [th.LongTensor(i) for i in train]
    val = [th.LongTensor(i) for i in val]
    test = [th.LongTensor(i) for i in test]
    return [nei_a, nei_s], [feat_p, feat_a, feat_s], [pap, psp], pos, label, train, val, test


def load_dblp(ratio, type_num):
    # The order of node types: 0 a 1 p 2 c 3 t
    path = "../data/dblp/"
    label = np.load(path + "labels.npy").astype('int32')
    label = encode_onehot(label)
    nei_p = np.load(path + "nei_p.npy", allow_pickle=True)
    feat_a = sp.load_npz(path + "a_feat.npz").astype("float32")
    feat_p = sp.eye(type_num[1])
    apa = sp.load_npz(path + "apa.npz")
    apcpa = sp.load_npz(path + "apcpa.npz")
    aptpa = sp.load_npz(path + "aptpa.npz")
    pos = sp.load_npz(path + "pos.npz")
    train = [np.load(path + "train_" + str(i) + ".npy") for i in ratio]
    test = [np.load(path + "test_" + str(i) + ".npy") for i in ratio]
    val = [np.load(path + "val_" + str(i) + ".npy") for i in ratio]
    
    label = th.FloatTensor(label)
    nei_p = [th.LongTensor(i) for i in nei_p]
    feat_p = th.FloatTensor(preprocess_features(feat_p))
    feat_a = th.FloatTensor(preprocess_features(feat_a))
    apa = sparse_mx_to_torch_sparse_tensor(normalize_adj(apa))
    apcpa = sparse_mx_to_torch_sparse_tensor(normalize_adj(apcpa))
    aptpa = sparse_mx_to_torch_sparse_tensor(normalize_adj(aptpa))
    pos = sparse_mx_to_torch_sparse_tensor(pos)
    train = [th.LongTensor(i) for i in train]
    val = [th.LongTensor(i) for i in val]
    test = [th.LongTensor(i) for i in test]
    return [nei_p], [feat_a, feat_p], [apa, apcpa, aptpa], pos, label, train, val, test


def load_aminer(ratio, type_num):
    # The order of node types: 0 p 1 a 2 r
    path = "../data/aminer/"
    label = np.load(path + "labels.npy").astype('int32')
    label = encode_onehot(label)
    nei_a = np.load(path + "nei_a.npy", allow_pickle=True)
    nei_r = np.load(path + "nei_r.npy", allow_pickle=True)
    # Because none of P, A or R has features, we assign one-hot encodings to all of them.
    feat_p = sp.eye(type_num[0])
    feat_a = sp.eye(type_num[1])
    feat_r = sp.eye(type_num[2])
    pap = sp.load_npz(path + "pap.npz")
    prp = sp.load_npz(path + "prp.npz")
    pos = sp.load_npz(path + "pos.npz")
    train = [np.load(path + "train_" + str(i) + ".npy") for i in ratio]
    test = [np.load(path + "test_" + str(i) + ".npy") for i in ratio]
    val = [np.load(path + "val_" + str(i) + ".npy") for i in ratio]

    label = th.FloatTensor(label)
    nei_a = [th.LongTensor(i) for i in nei_a]
    nei_r = [th.LongTensor(i) for i in nei_r]
    feat_p = th.FloatTensor(preprocess_features(feat_p))
    feat_a = th.FloatTensor(preprocess_features(feat_a))
    feat_r = th.FloatTensor(preprocess_features(feat_r))
    pap = sparse_mx_to_torch_sparse_tensor(normalize_adj(pap))
    prp = sparse_mx_to_torch_sparse_tensor(normalize_adj(prp))
    pos = sparse_mx_to_torch_sparse_tensor(pos)
    train = [th.LongTensor(i) for i in train]
    val = [th.LongTensor(i) for i in val]
    test = [th.LongTensor(i) for i in test]
    return [nei_a, nei_r], [feat_p, feat_a, feat_r], [pap, prp], pos, label, train, val, test


def load_freebase(ratio, type_num):
    # The order of node types: 0 m 1 d 2 a 3 w
    path = "../data/freebase/"
    label = np.load(path + "labels.npy").astype('int32')
    label = encode_onehot(label)
    nei_d = np.load(path + "nei_d.npy", allow_pickle=True)
    nei_a = np.load(path + "nei_a.npy", allow_pickle=True)
    nei_w = np.load(path + "nei_w.npy", allow_pickle=True)
    feat_m = sp.eye(type_num[0])
    feat_d = sp.eye(type_num[1])
    feat_a = sp.eye(type_num[2])
    feat_w = sp.eye(type_num[3])
    # Because none of M, D, A or W has features, we assign one-hot encodings to all of them.
    mam = sp.load_npz(path + "mam.npz")
    mdm = sp.load_npz(path + "mdm.npz")
    mwm = sp.load_npz(path + "mwm.npz")
    pos = sp.load_npz(path + "pos.npz")
    train = [np.load(path + "train_" + str(i) + ".npy") for i in ratio]
    test = [np.load(path + "test_" + str(i) + ".npy") for i in ratio]
    val = [np.load(path + "val_" + str(i) + ".npy") for i in ratio]

    label = th.FloatTensor(label)
    nei_d = [th.LongTensor(i) for i in nei_d]
    nei_a = [th.LongTensor(i) for i in nei_a]
    nei_w = [th.LongTensor(i) for i in nei_w]
    feat_m = th.FloatTensor(preprocess_features(feat_m))
    feat_d = th.FloatTensor(preprocess_features(feat_d))
    feat_a = th.FloatTensor(preprocess_features(feat_a))
    feat_w = th.FloatTensor(preprocess_features(feat_w))
    mam = sparse_mx_to_torch_sparse_tensor(normalize_adj(mam))
    mdm = sparse_mx_to_torch_sparse_tensor(normalize_adj(mdm))
    mwm = sparse_mx_to_torch_sparse_tensor(normalize_adj(mwm))
    pos = sparse_mx_to_torch_sparse_tensor(pos)
    train = [th.LongTensor(i) for i in train]
    val = [th.LongTensor(i) for i in val]
    test = [th.LongTensor(i) for i in test]
    return [nei_d, nei_a, nei_w], [feat_m, feat_d, feat_a, feat_w], [mdm, mam, mwm], pos, label, train, val, test

def load_imdb(ratio, type_num):
    # 节点类型顺序：0 m(电影) 1 d(导演) 2 a(演员) 3 w(关键词)
    path = "../data/imdb/"
    
    # 加载标签并检查类别数
    print("加载IMDB数据集...")
    label = np.load(path + "labels.npy").astype('int32')
    unique_labels = np.unique(label)
    num_classes = len(unique_labels)
    print(f"IMDB数据集标签类别数: {num_classes}")
    print(f"标签值范围: {unique_labels.min()} 到 {unique_labels.max()}")
    
    # 检查标签是否需要重新映射（确保从0开始的连续整数）
    if not np.array_equal(np.sort(unique_labels), np.arange(num_classes)):
        print("标签不是从0开始的连续整数，进行重新映射...")
        label_map = {old_label: new_label for new_label, old_label in enumerate(unique_labels)}
        label = np.array([label_map[lbl] for lbl in label])
        print(f"标签重新映射后的类别数: {len(np.unique(label))}")
    
    # 使用独热编码
    label = encode_onehot(label)
    print(f"标签形状: {label.shape}")
    
    # 加载邻居列表
    nei_d = np.load(path + "nei_d.npy", allow_pickle=True)
    nei_a = np.load(path + "nei_a.npy", allow_pickle=True)
    nei_w = np.load(path + "nei_w.npy", allow_pickle=True)
    
    # 为所有节点类型创建独热编码特征
    feat_m = sp.eye(type_num[0])
    feat_d = sp.eye(type_num[1])
    feat_a = sp.eye(type_num[2])
    feat_w = sp.eye(type_num[3])
    
    # 构建元路径矩阵
    print("构建元路径矩阵...")
    
    # 从关系文件加载边
    md_edges = []
    ma_edges = []
    mw_edges = []
    
    print("加载边数据...")
    with open(path + "md.txt", 'r') as f:
        for line in f:
            parts = line.strip().split('\t')
            if len(parts) == 2:
                md_edges.append([int(parts[0]), int(parts[1])])
                
    with open(path + "ma.txt", 'r') as f:
        for line in f:
            parts = line.strip().split('\t')
            if len(parts) == 2:
                ma_edges.append([int(parts[0]), int(parts[1])])
                
    with open(path + "mw.txt", 'r') as f:
        for line in f:
            parts = line.strip().split('\t')
            if len(parts) == 2:
                mw_edges.append([int(parts[0]), int(parts[1])])
    
    print(f"边数统计: MD={len(md_edges)}, MA={len(ma_edges)}, MW={len(mw_edges)}")
    
    # 构建二部图邻接矩阵
    md_rows, md_cols = zip(*md_edges) if md_edges else ([], [])
    ma_rows, ma_cols = zip(*ma_edges) if ma_edges else ([], [])
    mw_rows, mw_cols = zip(*mw_edges) if mw_edges else ([], [])
    
    md_adj = sp.coo_matrix(
        (np.ones(len(md_rows)), (md_rows, md_cols)), 
        shape=(type_num[0], type_num[1])
    )
    ma_adj = sp.coo_matrix(
        (np.ones(len(ma_rows)), (ma_rows, ma_cols)), 
        shape=(type_num[0], type_num[2])
    )
    mw_adj = sp.coo_matrix(
        (np.ones(len(mw_rows)), (mw_rows, mw_cols)), 
        shape=(type_num[0], type_num[3])
    )
    
    # 构建元路径矩阵
    mdm = md_adj.dot(md_adj.transpose())  # 电影-导演-电影
    mam = ma_adj.dot(ma_adj.transpose())  # 电影-演员-电影
    mwm = mw_adj.dot(mw_adj.transpose())  # 电影-关键词-电影
    
    # 构建正例矩阵
    pos = mdm + mam + mwm
    pos.setdiag(0)  # 移除自环
    pos.data = np.ones_like(pos.data)  # 将所有非零元素设为1
    
    # 可选：保存构建的矩阵供未来使用
    sp.save_npz(path + "mdm.npz", mdm)
    sp.save_npz(path + "mam.npz", mam)
    sp.save_npz(path + "mwm.npz", mwm)
    sp.save_npz(path + "pos.npz", pos)
    
    # 加载数据集分割
    train = [np.load(path + "train_" + str(i) + ".npy") for i in ratio]
    test = [np.load(path + "test_" + str(i) + ".npy") for i in ratio]
    val = [np.load(path + "val_" + str(i) + ".npy") for i in ratio]
    
    # 转换为PyTorch张量
    label = th.FloatTensor(label)
    nei_d = [th.LongTensor(i) for i in nei_d]
    nei_a = [th.LongTensor(i) for i in nei_a]
    nei_w = [th.LongTensor(i) for i in nei_w]
    feat_m = th.FloatTensor(preprocess_features(feat_m))
    feat_d = th.FloatTensor(preprocess_features(feat_d))
    feat_a = th.FloatTensor(preprocess_features(feat_a))
    feat_w = th.FloatTensor(preprocess_features(feat_w))
    
    # 归一化邻接矩阵并转换为PyTorch稀疏张量
    mdm = sparse_mx_to_torch_sparse_tensor(normalize_adj(mdm))
    mam = sparse_mx_to_torch_sparse_tensor(normalize_adj(mam))
    mwm = sparse_mx_to_torch_sparse_tensor(normalize_adj(mwm))
    pos = sparse_mx_to_torch_sparse_tensor(pos)
    
    train = [th.LongTensor(i) for i in train]
    val = [th.LongTensor(i) for i in val]
    test = [th.LongTensor(i) for i in test]
    
    print("IMDB数据集加载完成")
    return [nei_d, nei_a, nei_w], [feat_m, feat_d, feat_a, feat_w], [mdm, mam, mwm], pos, label, train, val, test

def load_data(dataset, ratio, type_num):
    if dataset == "acm":
        data = load_acm(ratio, type_num)
    elif dataset == "dblp":
        data = load_dblp(ratio, type_num)
    elif dataset == "aminer":
        data = load_aminer(ratio, type_num)
    elif dataset == "freebase":
        data = load_freebase(ratio, type_num)
    elif dataset == "imdb":  # 添加IMDB数据集
        data = load_imdb(ratio, type_num)
    return data
