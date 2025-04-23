import torch
from abc import ABC
from visualizer import Visualizer

class Dataset(ABC):
    """
    Abstract dataset class that concatenates multiple clusters of data and tracks labels.

    The Dataset stores:
      - data: a single Tensor of all samples stacked along the first dimension.
      - labels: a list of integer labels indicating cluster membership for each sample.
      - clusters: the original clusters as numpy arrays for external use.
      - cluster_sizes: list of sizes for each cluster, useful for segmentation.
    """
    def __init__(self, clusters):
        """
        Initialize the dataset from a list of tensor clusters.

        Parameters:
            clusters (List[Tensor]): List of Tensors, each of shape (Ni, D),
                                     where Ni is number of samples in cluster i.
        """
        labels = []
        # Build label list: assign integer label for each cluster's samples
        for label, cluster in enumerate(clusters):
            labels.extend([label] * cluster.size(0))
        self.labels = labels
        # Concatenate all clusters into one data tensor
        self.data = torch.cat(clusters, dim=0)
        # Store original clusters as numpy arrays for visualization or analysis
        self.clusters = [c.numpy() for c in clusters]
        # Keep track of how many samples per cluster
        self.cluster_sizes = [c.size(0) for c in clusters]


''' -------------------------------------- 3D Data -------------------------------------- '''
class Dataset3Dbut3D(Dataset):
    def __init__(self):
        super().__init__([
            torch.randn( 5, 3) * 0.75 + torch.tensor([ 0.00,  0.00,  0.00]),
            torch.randn(50, 3) * 0.5  + torch.tensor([ 2.00,  2.00,  2.00]),
            torch.randn(50, 3) * 0.5  + torch.tensor([ 2.00, -2.00, -2.00]),
            torch.randn(50, 3) * 0.5  + torch.tensor([-2.00,  2.00, -2.00]),
            torch.randn(50, 3) * 0.5  + torch.tensor([-2.00, -2.00,  2.00]),
        ])

class Dataset3Dbut2D(Dataset):
    def __init__(self):
        super().__init__([
            torch.randn( 5, 3) * 0.75 + torch.tensor([ 0.00,  0.00,  0.00]),
            torch.randn(50, 3) * 0.5  + torch.tensor([ 0.00,  3.50,  0.00]),
            torch.randn(50, 3) * 0.5  + torch.tensor([ 3.03, -1.75,  0.00]),
            torch.randn(50, 3) * 0.5  + torch.tensor([-3.03, -1.75,  0.00]),
        ])

class Dataset3Dbut1D(Dataset):
    def __init__(self):
        super().__init__([
            torch.randn( 5, 3) * 0.75 + torch.tensor([ 0.00,  0.00,  0.00]),
            torch.randn(50, 3) * 0.5  + torch.tensor([ 2.00,  2.00,  2.00]),
            torch.randn(50, 3) * 0.5  + torch.tensor([-2.00, -2.00, -2.00]),
        ])


''' -------------------------------------- 2D Data -------------------------------------- '''
class Dataset2Dbut2D(Dataset):
    def __init__(self):
        super().__init__([
            torch.randn( 5, 2) * 0.75 + torch.tensor([ 0.00,  0.00]),
            torch.randn(50, 2) * 0.5  + torch.tensor([ 0.00,  3.50]),
            torch.randn(50, 2) * 0.5  + torch.tensor([ 3.03, -1.75]),
            torch.randn(50, 2) * 0.5  + torch.tensor([-3.03, -1.75]),
        ])

class Dataset2Dbut1D(Dataset):
    def __init__(self):
        super().__init__([
            torch.randn( 5, 2) * 0.75 + torch.tensor([ 0.00,  0.00]),
            torch.randn(50, 2) * 0.5  + torch.tensor([ 2.00,  2.00]),
            torch.randn(50, 2) * 0.5  + torch.tensor([-2.00, -2.00]),
        ])

if __name__ == "__main__":
    # Dataset Showcase
    Visualizer(
    layout=(2, 3), 
    plot_specs=[
        {"type": "2d", "clusters": Dataset2Dbut1D().clusters, "kwargs": {"title": "Dataset2Dbut1D"}},
        {"type": "2d", "clusters": Dataset2Dbut2D().clusters, "kwargs": {"title": "Dataset2Dbut2D"}},
        {"type": "3d", "clusters": Dataset3Dbut1D().clusters, "kwargs": {"title": "Dataset3Dbut1D"}},
        {"type": "3d", "clusters": Dataset3Dbut2D().clusters, "kwargs": {"title": "Dataset3Dbut2D"}},
        {"type": "3d", "clusters": Dataset3Dbut3D().clusters, "kwargs": {"title": "Dataset3Dbut3D"}},
        ]
    ).show()