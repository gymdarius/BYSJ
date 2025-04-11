import numpy as np
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA
import os
import torch

def perform_pca_analysis(embeddings, labels, dataset_name, n_components=2):
    """
    Perform PCA analysis on the node embeddings.
    
    Parameters:
    -----------
    embeddings : numpy.ndarray
        The node embeddings matrix
    labels : numpy.ndarray
        The node labels
    dataset_name : str
        Name of the dataset
    n_components : int, optional
        Number of PCA components to use (default is 2)
    """
    # Create directory for PCA results if it doesn't exist
    os.makedirs(f"./pca_results/{dataset_name}", exist_ok=True)
    
    # Convert embeddings to numpy if they are torch tensors
    if isinstance(embeddings, torch.Tensor):
        embeddings = embeddings.cpu().numpy()
    
    # Convert labels to one-hot format to class indices
    if len(labels.shape) > 1 and labels.shape[1] > 1:  # If labels are one-hot encoded
        labels = np.argmax(labels, axis=1)
    
    # Apply PCA
    pca = PCA(n_components=n_components)
    reduced_embeddings = pca.fit_transform(embeddings)
    
    # Calculate explained variance
    explained_variance = pca.explained_variance_ratio_
    cumulative_variance = np.cumsum(explained_variance)
    
    # Plot explained variance
    plt.figure(figsize=(10, 6))
    plt.bar(range(1, len(explained_variance) + 1), explained_variance)
    plt.plot(range(1, len(cumulative_variance) + 1), cumulative_variance, 'r-')
    plt.xlabel('Principal Components')
    plt.ylabel('Explained Variance Ratio')
    plt.title(f'Explained Variance by Principal Components - {dataset_name}')
    plt.savefig(f"./pca_results/{dataset_name}/explained_variance.png")
    
    # Plot the first two principal components
    if n_components >= 2:
        plt.figure(figsize=(10, 8))
        scatter = plt.scatter(reduced_embeddings[:, 0], reduced_embeddings[:, 1], c=labels, cmap='viridis', alpha=0.7)
        plt.colorbar(scatter, label='Class')
        plt.xlabel('Principal Component 1')
        plt.ylabel('Principal Component 2')
        plt.title(f'PCA Visualization of Node Embeddings - {dataset_name}')
        plt.savefig(f"./pca_results/{dataset_name}/pca_visualization.png")
    
    # Save PCA results
    np.save(f"./pca_results/{dataset_name}/pca_embeddings.npy", reduced_embeddings)
    np.save(f"./pca_results/{dataset_name}/explained_variance.npy", explained_variance)
    
    print(f"PCA analysis completed. Results saved in ./pca_results/{dataset_name}/")
    
    return reduced_embeddings, explained_variance