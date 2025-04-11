import numpy as np
import matplotlib.pyplot as plt
import os
import argparse

def main():
    # 解析命令行参数
    parser = argparse.ArgumentParser(description='t-SNE可视化脚本')
    parser.add_argument('--dataset', type=str, default='imdb', help='数据集名称')
    parser.add_argument('--labels_path', type=str, default='../data/imdb/labels.npy', help='标签文件路径')
    args = parser.parse_args()
    
    # 设置目录和文件路径
    dataset_name = args.dataset
    features_dir = f"./tsne_results/{dataset_name}"
    features_path = os.path.join(features_dir, "tsne_features.npy")
    
    # 检查文件是否存在
    if not os.path.exists(features_path):
        print(f"错误：找不到t-SNE特征文件: {features_path}")
        return
    
    # 加载t-SNE特征
    print(f"加载t-SNE特征: {features_path}")
    reduced_features = np.load(features_path)
    
    # 加载标签
    if os.path.exists(args.labels_path):
        print(f"加载标签: {args.labels_path}")
        labels = np.load(args.labels_path)
        # 如果标签是one-hot编码，转换为类别索引
        if len(labels.shape) > 1 and labels.shape[1] > 1:
            labels = np.argmax(labels, axis=1)
    else:
        print(f"警告：找不到标签文件，将使用聚类ID作为标签")
        from sklearn.cluster import KMeans
        n_clusters = 20  # 根据IMDB数据集的情况设置
        kmeans = KMeans(n_clusters=n_clusters, random_state=42)
        labels = kmeans.fit_predict(reduced_features)
    
    # 创建带有颜色分类的散点图
    plt.figure(figsize=(10, 8))
    scatter = plt.scatter(reduced_features[:, 0], reduced_features[:, 1], 
                         c=labels, cmap='jet', s=10, alpha=0.7)
    plt.colorbar(scatter, label="类别")
    plt.title(f"{dataset_name}数据集的t-SNE聚类可视化")
    plt.tight_layout()
    
    # 保存图像
    os.makedirs(features_dir, exist_ok=True)
    viz_path = os.path.join(features_dir, "tsne_visualization.png")
    plt.savefig(viz_path, dpi=300)
    print(f"t-SNE可视化已保存至: {viz_path}")
    
    # 创建带有图例的可视化
    plt.figure(figsize=(12, 10))
    unique_labels = np.unique(labels)
    for i in unique_labels:
        idx = np.where(labels == i)[0]
        plt.scatter(reduced_features[idx, 0], reduced_features[idx, 1], 
                   label=f'类别 {i}', s=15, alpha=0.7)
    
    plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
    plt.title(f"{dataset_name}数据集的t-SNE类别分布")
    plt.tight_layout()
    
    # 保存带图例的图像
    legend_viz_path = os.path.join(features_dir, "tsne_visualization_with_legend.png")
    plt.savefig(legend_viz_path, dpi=300)
    print(f"带图例的t-SNE可视化已保存至: {legend_viz_path}")
    
    print("可视化完成！")

if __name__ == "__main__":
    main()