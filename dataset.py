import torch
import numpy as np
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

    def cloud(self, n_samples, center, sigma):
        """
        Generate a point cloud of n_samples points around a center in R^d.

        Args:
            n_samples: Number of points to sample.
            center: Center of the cloud (array-like of shape (d,)).
            sigma: Standard deviation of the Gaussian noise.

        Returns:
            Tensor of shape (n_samples, d): The generated point cloud.
        """
        return torch.randn(n_samples, len(center)) * sigma + torch.tensor(center)

''' -------------------------------------- 3D Data -------------------------------------- '''
class Ds3Dbut3Dmulti(Dataset):
    def __init__(self):
        super().__init__([
            self.cloud( 3, ( 2.00,  0.00,  0.00), 0.50),
            self.cloud( 3, ( 0.00,  2.00,  0.00), 0.50),
            self.cloud( 3, ( 0.00, -2.00,  0.00), 0.50),
            #self.cloud( 3, (-2.00,  0.00,  0.00), 0.50),
            self.cloud(50, ( 2.00,  2.00,  2.00), 0.50),
            self.cloud(50, ( 2.00, -2.00, -2.00), 0.50),
            self.cloud(50, (-2.00,  2.00, -2.00), 0.50),
            self.cloud(50, (-2.00, -2.00,  2.00), 0.50),
        ])
class Ds3Dbut3Dsingle(Dataset):
    def __init__(self):
        super().__init__([
            self.cloud( 5, ( 0.00,  0.00,  0.00), 0.75),
            self.cloud(50, ( 2.00,  2.00,  2.00), 0.50),
            self.cloud(50, ( 2.00, -2.00, -2.00), 0.50),
            self.cloud(50, (-2.00,  2.00, -2.00), 0.50),
            self.cloud(50, (-2.00, -2.00,  2.00), 0.50),
        ])
class Ds3Dbut2Dmulti(Dataset):
    def __init__(self):
        super().__init__([
            self.cloud( 3, ( 1.52,  0.88,  0.00), 0.50),
            self.cloud( 3, (-1.52,  0.88,  0.00), 0.50),
            #self.cloud( 3, ( 0.00, -1.75,  0.00), 0.50),
            self.cloud(50, ( 0.00,  3.50,  0.00), 0.50),
            self.cloud(50, ( 3.03, -1.75,  0.00), 0.50),
            self.cloud(50, (-3.03, -1.75,  0.00), 0.50),
        ])
class Ds3Dbut2Dsingle(Dataset):
    def __init__(self):
        super().__init__([
            self.cloud( 5, ( 0.00,  0.00,  0.00), 0.75),
            self.cloud(50, ( 0.00,  3.50,  0.00), 0.50),
            self.cloud(50, ( 3.03, -1.75,  0.00), 0.50),
            self.cloud(50, (-3.03, -1.75,  0.00), 0.50),
        ])
class Ds3Dbut1Dsingle(Dataset):
    def __init__(self):
        super().__init__([
            self.cloud( 5, ( 0.00,  0.00,  0.00), 0.75),
            self.cloud(50, ( 2.00,  2.00,  2.00), 0.50),
            self.cloud(50, (-2.00, -2.00, -2.00), 0.50),
        ])


''' -------------------------------------- 2D Data -------------------------------------- '''
class Ds2Dbut2Dmulti(Dataset):
    def __init__(self):
        super().__init__([
            self.cloud( 3, ( 1.52,  0.88), 0.50),
            self.cloud( 3, (-1.52,  0.88), 0.50),
            #self.cloud( 3, ( 0.00, -1.75), 0.50),
            self.cloud(50, ( 0.00,  3.50), 0.50),
            self.cloud(50, ( 3.03, -1.75), 0.50),
            self.cloud(50, (-3.03, -1.75), 0.50),
        ])
class Ds2Dbut2Dsingle(Dataset):
    def __init__(self):
        super().__init__([
            self.cloud( 5, ( 0.00,  0.00), 0.75),
            self.cloud(50, ( 0.00,  3.50), 0.50),
            self.cloud(50, ( 3.03, -1.75), 0.50),
            self.cloud(50, (-3.03, -1.75), 0.50),
        ])
class Ds2Dbut1Dsingle(Dataset):
    def __init__(self):
        super().__init__([
            self.cloud( 5, ( 0.00,  0.00), 0.75),
            self.cloud(50, ( 2.00,  2.00), 0.50),
            self.cloud(50, (-2.00, -2.00), 0.50),
        ])


''' -------------------------------------- Moon Data -------------------------------------- '''
class DatasetMoon(Dataset, ABC):
    """
    Abstract Moon dataset class providing a method to generate arc-shaped clusters.
    """
    def __init__(self, clusters):
        super().__init__(clusters)

    def moon(self, dim, n_samples, noise, center, radius, arc_start, arc_length):
        """
        Generate a half-moon cluster embedded in R^dim.

        Parameters:
            dim: ambient dimensionality (>=2)
            n_samples: number of points to sample
            noise: standard deviation of Gaussian noise added to all dimensions
            center: sequence of length dim specifying shift along each axis
            radius: radius of the circular arc in the first two dimensions
            arc_start: starting angle of the arc (radians)
            arc_length: angular span of the arc (radians)

        Returns:
            torch.Tensor of shape (n_samples, dim)
        """
        # Sample arc angles
        theta = np.random.rand(n_samples) * arc_length + arc_start
        # 2D half-moon in first two coords
        pts2d = np.column_stack([np.cos(theta), np.sin(theta)]) * radius
        # Embed into higher dims and add noise
        X = np.zeros((n_samples, dim))
        X[:, :2] = pts2d
        X += noise * np.random.randn(n_samples, dim)
        # Apply shift
        X += np.array(center).reshape(1, dim)
        # Convert to torch tensor
        return torch.tensor(X, dtype=torch.float32)

class Ds2DMoons(DatasetMoon):
    def __init__(self):
        super().__init__([
            self.cloud( 1, (-1.00,  0.00), 0.01),
            self.cloud( 1, ( 0.00,  0.00), 0.01),
            self.cloud( 1, ( 1.00,  0.00), 0.01),
            self.moon(2, 50, 0.05, ( 0.5, -0.2), 1.0,     0, np.pi),
            self.moon(2, 50, 0.05, (-0.5,  0.2), 1.0, np.pi, np.pi),
        ])
class Ds3DMoons(DatasetMoon):
    def __init__(self):
        super().__init__([
            self.cloud( 3, (-1.00,  0.00,  0.00), 0.10),
            self.cloud( 3, ( 0.00,  0.00,  0.00), 0.10),
            self.cloud( 3, ( 1.00,  0.00,  0.00), 0.10),
            self.moon(3, 75, 0.05, ( 0.5, -0.2, 0.0), 1.0,     0, np.pi),
            self.moon(3, 75, 0.05, (-0.5,  0.2, 0.0), 1.0, np.pi, np.pi),
        ])

if __name__ == "__main__":
    # Dataset Showcase
    # Visualizer(
    # layout=(2, 4), 
    # plot_specs=[
    #     {"clusters": Ds2Dbut1Dsingle().clusters, "kwargs": {"title": "Ds2Dbut1Dsingle"}},
    #     {"clusters": Ds2Dbut2Dsingle().clusters, "kwargs": {"title": "Ds2Dbut2Dsingle"}},
    #     {"clusters": Ds2Dbut2Dmulti().clusters,  "kwargs": {"title": "Ds2Dbut2Dmulti"}},
    #     {"clusters": Ds3Dbut1Dsingle().clusters, "kwargs": {"title": "Ds3Dbut1Dsingle"}},
    #     {"clusters": Ds3Dbut2Dsingle().clusters, "kwargs": {"title": "Ds3Dbut2Dsingle"}},
    #     {"clusters": Ds3Dbut2Dmulti().clusters,  "kwargs": {"title": "Ds3Dbut2Dmulti"}},
    #     {"clusters": Ds3Dbut3Dsingle().clusters, "kwargs": {"title": "Ds3Dbut3Dsingle"}},
    #     {"clusters": Ds3Dbut3Dmulti().clusters,  "kwargs": {"title": "Ds3Dbut3Dmulti"}},
    #     ]
    # ).show()
    Visualizer(
    layout=(1, 2),
    plot_specs=[
        {"clusters": Ds2DMoons().clusters, "kwargs": {"title": "Ds2DMoons"}},
        {"clusters": Ds3DMoons().clusters, "kwargs": {"title": "Ds3DMoons"}},
        ]
    ).show()