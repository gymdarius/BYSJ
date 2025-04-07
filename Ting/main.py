import numpy
import torch
from utils import load_data, set_params, evaluate
from utils import perform_pca_analysis
from module import HeCo
import warnings
import datetime
import pickle as pkl
import os
import random
import time
import json
import psutil
import GPUtil
from sklearn.decomposition import PCA
import matplotlib.pyplot as plt
import pandas as pd

warnings.filterwarnings('ignore')
args = set_params()
if torch.cuda.is_available():
    device = torch.device("cuda:" + str(args.gpu))
    torch.cuda.set_device(args.gpu)
else:
    device = torch.device("cpu")

## name of intermediate document ##
own_str = args.dataset

## random seed ##
seed = args.seed
numpy.random.seed(seed)
random.seed(seed)
torch.manual_seed(seed)
torch.cuda.manual_seed(seed)


def get_memory_usage():
    process = psutil.Process(os.getpid())
    mem_info = process.memory_info()
    return mem_info.rss / 1024 ** 2  # in MB

def get_gpu_usage(gpu_id=0):
    if torch.cuda.is_available():
        gpu_memory = torch.cuda.memory_allocated(0) / 1024 / 1024  # 获取显存（MB）
        gpu_memory_max = torch.cuda.memory_reserved(0) / 1024 / 1024  # 最大显存
        return gpu_memory, gpu_memory_max
    return 0, 0

def visualize_pca(feat_full, labels):
    pca = PCA(n_components=2)  # 使用PCA将高维数据降到二维
    reduced_data = pca.fit_transform(feat_full)  # 适配PCA并降维
    labels = np.argmax(labels, axis=1)  # 如果labels是one-hot编码，使用np.argmax转换为一维标签
    plt.scatter(reduced_data[:, 0], reduced_data[:, 1], c=labels, cmap='jet', alpha=0.5)
    plt.colorbar()  # 显示颜色条
    plt.title('PCA of Features')
    plt.show()


def save_results(results, filename):
    with open(filename, "w") as f:
        json.dump(results, f, indent=4)

def train():
    results = {
        "train_times": [],
        "losses": [],
        "mp": [],
        "sc": [],
        "classification": [],
        "clustering": [],
        "parameter_sensitivity": [],
        "pca_visualization": [],
        "memory_usage": [],
        "gpu_usage": []
    }
    nei_index, feats, mps, pos, label, idx_train, idx_val, idx_test = \
        load_data(args.dataset, args.ratio, args.type_num)
    nb_classes = label.shape[-1]
    feats_dim_list = [i.shape[1] for i in feats]
    P = int(len(mps))
    print("seed ", args.seed)
    print("Dataset: ", args.dataset)
    print("The number of meta-paths: ", P)
    
    model = HeCo(args.hidden_dim, feats_dim_list, args.feat_drop, args.attn_drop,
                    P, args.sample_rate, args.nei_num, args.tau, args.lam)
    optimiser = torch.optim.Adam(model.parameters(), lr=args.lr, weight_decay=args.l2_coef)

    if torch.cuda.is_available():
        print('Using CUDA')
        model.cuda()
        feats = [feat.cuda() for feat in feats]
        mps = [mp.cuda() for mp in mps]
        pos = pos.cuda()
        label = label.cuda()
        idx_train = [i.cuda() for i in idx_train]
        idx_val = [i.cuda() for i in idx_val]
        idx_test = [i.cuda() for i in idx_test]

    cnt_wait = 0
    best = 1e9
    best_t = 0

    starttime = datetime.datetime.now()
    for epoch in range(args.nb_epochs):
        epoch_start = time.time()
        model.train()
        optimiser.zero_grad()
        loss = model(feats, pos, mps, nei_index)
        results["losses"].append(loss.item())
        print("loss ", loss.data.cpu())
        if loss < best:
            best = loss
            best_t = epoch
            cnt_wait = 0
            torch.save(model.state_dict(), 'HeCo_' + own_str + '.pkl')
        else:
            cnt_wait += 1

        if cnt_wait == args.patience:
            print('Early stopping!')
            break
        loss.backward()
        optimiser.step()
        epoch_end = time.time()
        epoch_time = epoch_end - epoch_start
        results["train_times"].append(epoch_time)
        print(f"Epoch {epoch} time: {epoch_time} seconds")
        
        # 记录内存和显存使用情况
        memory_usage = get_memory_usage()
        gpu_usage, gpu_total = get_gpu_usage(args.gpu)
        results["memory_usage"].append(memory_usage)
        results["gpu_usage"].append({
            "used": gpu_usage,
            "total": gpu_total
        })
        print(f"Memory usage: {memory_usage} MB, GPU usage: {gpu_usage}/{gpu_total} MB")
    
    print('Loading {}th epoch'.format(best_t))
    model.load_state_dict(torch.load('HeCo_' + own_str + '.pkl'))
    model.eval()
    os.remove('HeCo_' + own_str + '.pkl')
    embeds = model.get_embeds(feats, mps)
    for i in range(len(idx_train)):
        evaluate(embeds, args.ratio[i], idx_train[i], idx_val[i], idx_test[i], label, nb_classes, device, args.dataset,
                 args.eva_lr, args.eva_wd)
    endtime = datetime.datetime.now()
    total_time = (endtime - starttime).seconds
    results["total_time"] = total_time
    print("Total time: ", total_time, "s")
    
    if args.save_emb:
        with open(f"./embeds/{args.dataset}/{str(args.turn)}.pkl", "wb") as f:
            pkl.dump(embeds.cpu().data.numpy(), f)
    
    # Perform PCA analysis
    if args.pca_analysis:
        #from utils import perform_pca_analysis
        print("Performing PCA analysis...")
        # Use all nodes for PCA analysis
        perform_pca_analysis(embeds.cpu().data.numpy(), label.cpu().numpy(), args.dataset, n_components=args.pca_components)
    
    # 保存结果
    save_results(results, f"results_{args.dataset}.json")

if __name__ == '__main__':
    train()