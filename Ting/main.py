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
from sklearn.manifold import TSNE  # Add t-SNE import
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np  # Add numpy import for np.argmax

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

# Replace PCA visualization with t-SNE visualization
def visualize_tsne(feat_full, labels):
    print("Performing t-SNE dimensionality reduction...")
    start_time = time.time()
    
    # t-SNE for dimensionality reduction to 2D
    tsne = TSNE(n_components=2, random_state=seed, perplexity=30)
    reduced_data = tsne.fit_transform(feat_full)
    
    # Convert one-hot labels to indices if needed
    if labels.ndim > 1:
        labels = np.argmax(labels, axis=1)
    
    plt.figure(figsize=(10, 8))
    scatter = plt.scatter(reduced_data[:, 0], reduced_data[:, 1], c=labels, cmap='jet', alpha=0.5)
    plt.colorbar(scatter)  # 显示颜色条
    plt.title('t-SNE Visualization of Features')
    
    # Create directory for saving results if it doesn't exist
    os.makedirs(f"./tsne_results/{args.dataset}/", exist_ok=True)
    plt.savefig(f"./tsne_results/{args.dataset}/tsne_visualization.png")
    plt.show()
    
    end_time = time.time()
    print(f"t-SNE completed in {end_time - start_time:.2f} seconds")


def perform_tsne_analysis(features, labels, dataset_name, n_components=2):
    """
    Perform t-SNE dimensionality reduction and visualization
    
    Args:
        features: Feature matrix to reduce
        labels: Node labels for coloring
        dataset_name: Name of the dataset for saving files
        n_components: Number of components for visualization (default: 2)
    """
    print(f"Performing t-SNE analysis with {n_components} components...")
    start_time = time.time()
    
    # Apply t-SNE
    # Note: t-SNE is computationally expensive, might need to subsample for large datasets
    if len(features) > 10000:
        print(f"Dataset is large ({len(features)} samples). Subsampling to 10000 samples for t-SNE.")
        indices = np.random.choice(len(features), 10000, replace=False)
        features_sample = features[indices]
        labels_sample = labels[indices] if labels.ndim == 1 else labels[indices, :]
    else:
        features_sample = features
        labels_sample = labels
    
    tsne = TSNE(n_components=n_components, perplexity=30, random_state=seed)
    reduced_features = tsne.fit_transform(features_sample)
    
    # Save reduced features
    os.makedirs(f"./tsne_results/{dataset_name}/", exist_ok=True)
    np.save(f"./tsne_results/{dataset_name}/tsne_features.npy", reduced_features)
    
    # Visualization for 2D
    if n_components == 2:
        plt.figure(figsize=(10, 8))
        labels_cat = np.argmax(labels_sample, axis=1) if labels_sample.ndim > 1 else labels_sample
        scatter = plt.scatter(reduced_features[:, 0], reduced_features[:, 1], 
                             c=labels_cat, cmap='jet', alpha=0.5)
        plt.colorbar(scatter)
        plt.title(f't-SNE Visualization of {dataset_name}')
        plt.savefig(f"./tsne_results/{dataset_name}/tsne_visualization.png")
        plt.close()
    
    end_time = time.time()
    print(f"t-SNE analysis completed in {end_time - start_time:.2f} seconds")
    return reduced_features


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
        "tsne_visualization": [],  # Changed from pca_visualization
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
    
    # Perform t-SNE analysis instead of PCA
    if args.pca_analysis:  # Keep the same parameter name for backward compatibility
        print("Performing t-SNE analysis instead of PCA...")
        # Use all nodes for t-SNE analysis
        perform_tsne_analysis(embeds.cpu().data.numpy(), label.cpu().numpy(), args.dataset, n_components=2)
    
    # 保存结果
    save_results(results, f"results_{args.dataset}.json")

if __name__ == '__main__':
    train()