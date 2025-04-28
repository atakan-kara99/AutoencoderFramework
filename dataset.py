import torch
import numpy as np
from numpy import pi
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

    def sphere3D(self, n_samples, noise, center, radius, 
            theta_start=0.0, theta_length=2*np.pi, phi_start=0.0, phi_length=np.pi):
        """
        Generate a parametric spherical segment in R^3.

        Parameters:
            n_samples    : number of points to sample
            noise        : std. dev. of Gaussian noise in all dimensions
            center       : sequence of length 3 specifying shift along each axis
            radius       : radius of the sphere
            theta_start  : starting azimuth angle around z-axis (radians)
            theta_length : angular span of azimuth (radians)
            phi_start    : starting polar (inclination) angle from z-axis (radians)
            phi_length   : angular span of polar angle (radians)

        Returns:
            torch.Tensor of shape (n_samples, 3)
        """
        # Sample spherical angles within given ranges
        theta = np.random.rand(n_samples) * theta_length + theta_start
        phi = np.random.rand(n_samples) * phi_length + phi_start
        # Convert spherical coordinates to Cartesian
        x = radius * np.sin(phi) * np.cos(theta)
        y = radius * np.sin(phi) * np.sin(theta)
        z = radius * np.cos(phi)
        X = np.column_stack([x, y, z])
        # Add Gaussian noise in each dimension
        X += noise * np.random.randn(n_samples, 3)
        # Shift cluster to the specified center
        X += np.array(center).reshape(1, 3)
        return torch.tensor(X, dtype=torch.float32)

    def swiss_roll3D(self,
                n_samples,
                noise,
                center,
                t_start=0.0,
                t_length=4*np.pi,
                height_start=0.0,
                height_length=1.0,
                radius_factor=1.0):
        """
        Generate a Swiss roll surface in R^3.

        Parameters:
            n_samples    : number of points to sample
            noise        : std. dev. of Gaussian noise in all dimensions
            center       : sequence of length 3 specifying shift along each axis
            t_start      : starting parameter around the roll (radians)
            t_length     : span of the roll parameter (radians)
            height_start : starting height along the roll axis
            height_length: range of heights along the roll axis
            radius_factor: scale factor for the roll radius

        Returns:
            torch.Tensor of shape (n_samples, 3)
        """
        t = np.random.rand(n_samples) * t_length + t_start
        h = np.random.rand(n_samples) * height_length + height_start
        x = radius_factor * t * np.cos(t)
        y = h
        z = radius_factor * t * np.sin(t)
        X = np.column_stack([x, y, z])
        X += noise * np.random.randn(n_samples, 3)
        X += np.array(center).reshape(1, 3)
        return torch.tensor(X, dtype=torch.float32)

    def swiss_roll_clusters(self,
                            n_samples_per,
                            noise,
                            center,
                            n_segments,
                            t_start=0.0,
                            t_length=4*np.pi,
                            height_start=0.0,
                            height_length=1.0,
                            radius_factor=1.0):
        """
        Partition the 3D Swiss roll surface into clusters for each segment of the parameter t.

        Parameters:
            n_samples_per: points per segment
            noise        : std. dev. of Gaussian noise
            center       : shift (length 3)
            n_segments   : number of segments along the roll
            t_start      : starting t for the full roll
            t_length     : total span of t
            height_start : starting height
            height_length: total height span
            radius_factor: scale of roll radius

        Returns:
            List[torch.Tensor] of length n_segments, each of shape (n_samples_per, 3)
        """
        segments = []
        seg_len = t_length / n_segments
        for i in range(n_segments):
            ts = t_start + i * seg_len
            cluster = self.swiss_roll3D(
                n_samples_per,
                noise,
                center,
                t_start=ts,
                t_length=seg_len,
                height_start=height_start,
                height_length=height_length,
                radius_factor=radius_factor)
            segments.append(cluster)
        return segments

''' -------------------------------------- 3D Data -------------------------------------- '''
class Ds3Dbut3Dmulti(Dataset):
    def __init__(self):
        super().__init__([
            self.cloud(50, ( 2.00,  2.00,  2.00), 0.50),
            self.cloud(50, ( 2.00, -2.00, -2.00), 0.50),
            self.cloud(50, (-2.00,  2.00, -2.00), 0.50),
            self.cloud(50, (-2.00, -2.00,  2.00), 0.50),
            self.cloud( 3, ( 2.00,  0.00,  0.00), 0.50),
            self.cloud( 3, ( 0.00,  2.00,  0.00), 0.50),
            self.cloud( 3, ( 0.00, -2.00,  0.00), 0.50),
            #self.cloud( 3, (-2.00,  0.00,  0.00), 0.50),
        ])
class Ds3Dbut3Dsingle(Dataset):
    def __init__(self):
        super().__init__([
            self.cloud(50, ( 2.00,  2.00,  2.00), 0.50),
            self.cloud(50, ( 2.00, -2.00, -2.00), 0.50),
            self.cloud(50, (-2.00,  2.00, -2.00), 0.50),
            self.cloud(50, (-2.00, -2.00,  2.00), 0.50),
            self.cloud( 5, ( 0.00,  0.00,  0.00), 0.75),
        ])
class Ds3Dbut2Dmulti(Dataset):
    def __init__(self):
        super().__init__([
            self.cloud(50, ( 0.00,  3.50,  0.00), 0.50),
            self.cloud(50, ( 3.03, -1.75,  0.00), 0.50),
            self.cloud(50, (-3.03, -1.75,  0.00), 0.50),
            self.cloud( 3, ( 1.52,  0.88,  0.00), 0.50),
            self.cloud( 3, (-1.52,  0.88,  0.00), 0.50),
            #self.cloud( 3, ( 0.00, -1.75,  0.00), 0.50),
        ])
class Ds3Dbut2Dsingle(Dataset):
    def __init__(self):
        super().__init__([
            self.cloud(50, ( 0.00,  3.50,  0.00), 0.50),
            self.cloud(50, ( 3.03, -1.75,  0.00), 0.50),
            self.cloud(50, (-3.03, -1.75,  0.00), 0.50),
            self.cloud( 5, ( 0.00,  0.00,  0.00), 0.75),
        ])
class Ds3Dbut1Dsingle(Dataset):
    def __init__(self):
        super().__init__([
            self.cloud(50, ( 2.00,  2.00,  2.00), 0.50),
            self.cloud(50, (-2.00, -2.00, -2.00), 0.50),
            self.cloud( 5, ( 0.00,  0.00,  0.00), 0.75),
        ])


''' -------------------------------------- 2D Data -------------------------------------- '''
class Ds2Dbut2Dmulti(Dataset):
    def __init__(self):
        super().__init__([
            self.cloud(50, ( 0.00,  3.50), 0.50),
            self.cloud(50, ( 3.03, -1.75), 0.50),
            self.cloud(50, (-3.03, -1.75), 0.50),
            self.cloud( 3, ( 1.52,  0.88), 0.50),
            self.cloud( 3, (-1.52,  0.88), 0.50),
            #self.cloud( 3, ( 0.00, -1.75), 0.50),
        ])
class Ds2Dbut2Dsingle(Dataset):
    def __init__(self):
        super().__init__([
            self.cloud(50, ( 0.00,  3.50), 0.50),
            self.cloud(50, ( 3.03, -1.75), 0.50),
            self.cloud(50, (-3.03, -1.75), 0.50),
            self.cloud( 5, ( 0.00,  0.00), 0.75),
        ])
class Ds2Dbut1Dsingle(Dataset):
    def __init__(self):
        super().__init__([
            self.cloud(50, ( 2.00,  2.00), 0.50),
            self.cloud(50, (-2.00, -2.00), 0.50),
            self.cloud( 5, ( 0.00,  0.00), 0.75),
        ])


''' -------------------------------------- Moon Data -------------------------------------- '''
class Ds2DMoons(Dataset):
    def __init__(self):
        super().__init__([
            self.moon(2, 50, 0.05, ( 0.50,  0.00), 1.0,  0, pi),
            self.moon(2, 50, 0.05, (-0.50,  0.00), 1.0, pi, pi),
            self.cloud(1, (-0.50, -0.50), 0.01),
            self.cloud(1, ( 0.50,  0.50), 0.01),
        ])
class Ds3DMoons(Dataset):
    def __init__(self):
        super().__init__([
            self.moon(3, 100, 0.05, ( 0.50,  0.00, 0.00), 1.0,  0, pi),
            self.moon(3, 100, 0.05, (-0.50,  0.00, 0.00), 1.0, pi, pi),
            self.cloud(3, (-0.50, -0.50,  0.00), 0.05),
            self.cloud(3, ( 0.50,  0.50,  0.00), 0.05),
        ])
class DsTrue3DMoons(Dataset):
    def __init__(self):
        super().__init__([
            self.moon(3, 75, 0.1, ( 0.50,  0.00, 0.00), 1.0,  0,   pi),
            self.moon(3, 75, 0.1, (-0.50,  0.00, 0.00), 1.0, pi,   pi),
            self.sphere3D(100, 0.05, ( 0.00,  0.00, -0.50), 2.0, 0.0, 2*np.pi, 0.0, np.pi/2),
            self.cloud(2, (-0.50, -0.50,  0.00), 0.05),
            self.cloud(2, ( 0.50,  0.50,  0.00), 0.05),
            self.cloud(3, ( 0.00,  0.00,  0.75), 0.10),
        ])

''' -------------------------------------- Swiss Roll Data -------------------------------------- '''
class Ds3DSwissRoll(Dataset):
    def __init__(self):
        super().__init__(
            self.swiss_roll_clusters(50, 0.10, ( 0.00,  0.00,  0.00), 4, 0.0, 4*pi, -5.0, -5.0, 1.0)
            )

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
    layout=(2, 3),
    plot_specs=[
        {"clusters": Ds2DMoons().clusters,     "kwargs": {"title": "Ds2DMoons"}},
        {"clusters": Ds3DMoons().clusters,     "kwargs": {"title": "Ds3DMoons"}},
        {"clusters": DsTrue3DMoons().clusters, "kwargs": {"title": "DsTrue3DMoons"}},
        {"clusters": Ds3DSwissRoll().clusters, "kwargs": {"title": "Ds3DSwissRoll"}},
        ]
    ).show()