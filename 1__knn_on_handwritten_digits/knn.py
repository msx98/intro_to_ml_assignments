from typing import *
import numpy as np
from scipy.spatial import KDTree


class KNeighborsClassifier:
    def __init__(self, n_neighbors: int):
        self.k = n_neighbors
        self.train_data, self.train_labels = None, None
        self.tree = None
    
    def fit(self, data: np.ndarray, labels: np.ndarray):
        self.labels = np.array(labels, dtype=np.int)
        self.tree = KDTree(
            data=data,
            leafsize=1000,
            compact_nodes=True,
            balanced_tree=True)
    
    def predict(self, data: np.ndarray):
        data_samples_count = len(data)
        _, closest_k_indices_per_entry = self.tree.query(
            x=data,
            k=self.k,
            p=2) # p=2 <=> L2 norm
        assert len(closest_k_indices_per_entry) == data_samples_count
        if len(closest_k_indices_per_entry.shape) == 1:
            closest_k_indices_per_entry = closest_k_indices_per_entry.reshape(-1,1)
        labels_of_k_closest_indices_per_entry = self.labels[closest_k_indices_per_entry]
        most_frequent_class_per_entry = [
            np.argmax(np.bincount(labels_of_k_closest_indices_per_entry[entry_idx]))
            for entry_idx in range(data_samples_count)]
        result = np.array([str(label) for label in most_frequent_class_per_entry])
        return result
        
        
def norm(v: np.ndarray) -> float:
    # calculates L2 norm
    assert len(v.shape) == 1
    result = np.linalg.norm(v)
    return result