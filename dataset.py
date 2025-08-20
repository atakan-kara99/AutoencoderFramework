import os
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
        self.labels = torch.tensor(labels, dtype=torch.long)
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

    def sphere3D(self, n_samples, noise, center, radius, theta_start=0.0, 
                 theta_length=2*np.pi, phi_start=0.0, phi_length=np.pi):
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

    def swiss_roll(self, n_samples, noise, center, dim = 3, n_segments = 1, t_start = 0.0, 
                   t_length = 4*np.pi, height_start = 0.0, height_length = 1.0, radius_factor = 1.0,):
        """
        Generate a Swiss-roll (spiral) in 2D or 3D, with optional segmentation into clusters.

        Parameters:
            n_samples    : if n_segments==1, total points; if >1, points per segment
            noise        : std. dev. of Gaussian noise
            center       : length-2 or length-3 shift along each axis
            dim          : 2 or 3
            n_segments   : how many clusters to split into (1 => no splitting)
            t_start      : start angle (radians)
            t_length     : total angle span (radians)
            height_start : 3D only: start height
            height_length: 3D only: total height span
            radius_factor: scale factor for the spiral radius

        Returns:
            If n_segments == 1:
                Tensor of shape (n_samples, dim)
            Else:
                List of n_segments tensors, each of shape (n_samples, dim)
        """
        center = np.array(center, dtype=float).reshape(1, dim)
        def _sample_segment(ts, seg_len):
            t = np.random.rand(n_samples) * seg_len + ts
            x = radius_factor * t * np.cos(t)
            if dim == 3:
                h = np.random.rand(n_samples) * height_length + height_start
                y, z = h, radius_factor * t * np.sin(t)
                X = np.column_stack([x, y, z])
            else:
                y = radius_factor * t * np.sin(t)
                X = np.column_stack([x, y])
            X += noise * np.random.randn(n_samples, dim)
            X += center
            return torch.tensor(X, dtype=torch.float32)
        # if only one segment, just sample and return a Tensor
        if n_segments == 1:
            return _sample_segment(t_start, t_length)
        # otherwise, partition into clusters
        seg_len = t_length / n_segments
        return [ _sample_segment(t_start + i * seg_len, seg_len) for i in range(n_segments) ]

    def torus(self, num_samples, noise, center, R=2.0, r=1.0, euler_angles=None):
        """
        Sample points on a torus, optionally rotate and translate them.

        Args:
            num_samples (int): Number of points to sample.
            R (float): Major radius (distance from center of tube to center of torus).
            r (float): Minor radius (tube radius).
            noise (float): Stddev of Gaussian noise to add.
            center (tuple of 3 floats): (cx, cy, cz) translation to apply.
            euler_angles (tuple of 3 floats): (alpha, beta, gamma) in radians;
                builds R = Rz(gamma) Ry(beta) Rx(alpha).

        Returns:
            torch.FloatTensor of shape (num_samples, 3)
        """
        # --- sample raw torus ---
        theta = 2 * np.pi * np.random.rand(num_samples)
        phi   = 2 * np.pi * np.random.rand(num_samples)
        x = (R + r * np.cos(phi)) * np.cos(theta)
        y = (R + r * np.cos(phi)) * np.sin(theta)
        z = r * np.sin(phi)
        points = np.stack((x, y, z), axis=1)  # (N,3)
        # --- optional rotation from Euler angles ---
        if euler_angles is not None:
            alpha, beta, gamma = euler_angles
            # Rotation about X, Y, Z
            Rx = np.array([[1, 0, 0],
                        [0, np.cos(alpha), -np.sin(alpha)],
                        [0, np.sin(alpha),  np.cos(alpha)]])
            Ry = np.array([[ np.cos(beta), 0, np.sin(beta)],
                        [           0, 1,           0],
                        [-np.sin(beta), 0, np.cos(beta)]])
            Rz = np.array([[np.cos(gamma), -np.sin(gamma), 0],
                        [np.sin(gamma),  np.cos(gamma), 0],
                        [            0,              0, 1]])
            rotation_matrix = Rz @ Ry @ Rx
            rm = np.asarray(rotation_matrix, dtype=float)
            points = points @ rm.T
        # --- translate to center ---
        center = np.asarray(center, dtype=float).reshape(1, 3)
        points = points + center
        # --- optional noise ---
        if noise > 0:
            noise = np.random.normal(scale=noise, size=points.shape)
            points += noise
        return torch.tensor(points, dtype=torch.float32)


''' -------------------------------------- 3D Data -------------------------------------- '''
class Ds3Dbut3Dmulti(Dataset):
    def __init__(self):
        super().__init__([
            self.cloud(50, ( 2.00,  2.00,  2.00), 0.75),
            self.cloud(50, ( 2.00, -2.00, -2.00), 0.75),
            self.cloud(50, (-2.00,  2.00, -2.00), 0.75),
            self.cloud(50, (-2.00, -2.00,  2.00), 0.75),
            self.cloud( 2, ( 2.00,  0.00,  0.00), 0.40),
            self.cloud( 2, ( 0.00,  2.00,  0.00), 0.40),
            self.cloud( 2, ( 0.00, -2.00,  0.00), 0.40),
            #self.cloud( 2, (-2.00,  0.00,  0.00), 0.40),
        ])
class Ds3Dbut3Dsingle(Dataset):
    def __init__(self):
        super().__init__([
            self.cloud(50, ( 2.00,  2.00,  2.00), 0.75),
            self.cloud(50, ( 2.00, -2.00, -2.00), 0.75),
            self.cloud(50, (-2.00,  2.00, -2.00), 0.75),
            self.cloud(50, (-2.00, -2.00,  2.00), 0.75),
            self.cloud( 4, ( 0.00,  0.00,  0.00), 0.80),
        ])
class Ds3Dbut2Dmulti(Dataset):
    def __init__(self):
        super().__init__([
            self.cloud(75, ( 0.00,  3.50,  0.00), 0.75),
            self.cloud(75, ( 3.03, -1.75,  0.00), 0.75),
            self.cloud(75, (-3.03, -1.75,  0.00), 0.75),
            self.cloud( 2, ( 1.52,  0.88,  0.00), 0.40),
            self.cloud( 2, (-1.52,  0.88,  0.00), 0.40),
            #self.cloud( 2, ( 0.00, -1.75,  0.00), 0.40),
        ])
class Ds3Dbut2Dsingle(Dataset):
    def __init__(self):
        super().__init__([
            self.cloud(75, ( 0.00,  3.50,  0.00), 0.75),
            self.cloud(75, ( 3.03, -1.75,  0.00), 0.75),
            self.cloud(75, (-3.03, -1.75,  0.00), 0.75),
            self.cloud( 3, ( 0.00,  0.00,  0.00), 0.60),
        ])
class Ds3Dbut1Dsingle(Dataset):
    def __init__(self):
        super().__init__([
            self.cloud(50, ( 1.83,  1.83,  1.83), 0.75),
            self.cloud(50, (-1.83, -1.83, -1.83), 0.75),
            self.cloud( 2, ( 0.00,  0.00,  0.00), 0.40),
        ])


''' -------------------------------------- 2D Data -------------------------------------- '''
class Ds2Dbut2Dmulti(Dataset):
    def __init__(self):
        super().__init__([
            self.cloud(50, ( 0.00,  3.50), 0.66),
            self.cloud(50, ( 3.03, -1.75), 0.66),
            self.cloud(50, (-3.03, -1.75), 0.66),
            self.cloud( 2, ( 1.52,  0.88), 0.40),
            self.cloud( 2, (-1.52,  0.88), 0.40),
            #self.cloud( 2, ( 0.00, -1.75), 0.40),
        ])
class Ds2Dbut2Dsingle(Dataset):
    def __init__(self):
        super().__init__([
            self.cloud(50, ( 0.00,  3.50), 0.66),
            self.cloud(50, ( 3.03, -1.75), 0.66),
            self.cloud(50, (-3.03, -1.75), 0.66),
            self.cloud( 3, ( 0.00,  0.00), 0.60),
        ])
class Ds2Dbut1Dsingle(Dataset):
    def __init__(self):
        super().__init__([
            self.cloud(50, ( 2.475,  2.475), 0.66),
            self.cloud(50, (-2.475, -2.475), 0.66),
            self.cloud( 2, ( 0.000,  0.000), 0.40),
        ])


''' -------------------------------------- Moon Data -------------------------------------- '''
class Ds2DMoons(Dataset):
    def __init__(self):
        super().__init__([
            self.moon(2, 25, 0.05, (-0.50,  0.00), 1.0,     pi, pi/2),
            self.moon(2, 25, 0.05, (-0.50,  0.00), 1.0, 1.5*pi, pi/2),
            self.moon(2, 25, 0.05, ( 0.50,  0.00), 1.0,      0, pi/2),
            self.moon(2, 25, 0.05, ( 0.50,  0.00), 1.0,   pi/2, pi/2),
            self.cloud(1, ( 0.00,  0.00), 0.01),
        ])
class Ds3DMoons(Dataset):
    def __init__(self):
        super().__init__([
            self.moon(3, 150, 0.05, ( 0.50,  0.00, 0.00), 1.0,  0, pi),
            self.moon(3, 150, 0.05, (-0.50,  0.00, 0.00), 1.0, pi, pi),
            self.cloud(2, ( 0.00,  0.00,  0.00), 0.05),
        ])
class DsTrue3DMoons(Dataset):
    def __init__(self):
        super().__init__([
            self.moon(3, 75, 0.1, ( 0.50,  0.00, 0.00), 1.0,  0,   pi),
            self.moon(3, 75, 0.1, (-0.50,  0.00, 0.00), 1.0, pi,   pi),
            self.sphere3D(150, 0.05, ( 0.00,  0.00, -0.50), 2.0, 0.0, 2*np.pi, 0.0, np.pi/2),
            self.cloud(2, (-0.50, -0.50,  0.00), 0.05),
            self.cloud(2, ( 0.50,  0.50,  0.00), 0.05),
            self.cloud(3, ( 0.00,  0.00,  0.75), 0.10),
        ])

''' -------------------------------------- Swiss Roll Data -------------------------------------- '''
class Ds2DSwissRoll(Dataset):
    def __init__(self):
        super().__init__(
            self.swiss_roll(50, 0.20, ( 0.00,  0.00), 2, 4, 0.0, 4*pi, 1.0) +
            [self.cloud(1, ( 3.0, 0.00), 0.25),
             self.cloud(1, (-6.0, 0.00), 0.25),]
             )
class Ds3DSwissRoll(Dataset):
    def __init__(self):
        super().__init__(
            self.swiss_roll(100, 0.10, ( 0.00,  0.00,  0.00), 3, 4, 0.0, 4*pi, -5.0, 10.0, 1.0) +
            [self.cloud(2, ( 3.0, 0.00, 0.00), 0.25),
             self.cloud(2, (-6.5, 0.00, 0.00), 0.25),]
            )
        
''' -------------------------------------- Torus Data -------------------------------------- '''
class Ds3DTorus(Dataset):
    def __init__(self):
        super().__init__([
            self.torus(200, 0.05, (-2.00,  0.00,  0.00), 5.0, 1.0, (0   , 0, 0)),
            self.torus(200, 0.05, ( 2.00,  0.00,  0.00), 5.0, 1.0, (pi/2, 0, 0)),
            self.cloud(2, ( 0.00,  0.00,  0.00), 0.25),
        ])

''' -------------------------------------- Sphere Data -------------------------------------- '''
class Ds3DSphere(Dataset):
    def __init__(self):
        super().__init__([
            self.sphere3D(50, 0.00, ( 0.00,  0.00,  0.00), 2.0),
            self.sphere3D(125, 0.00, ( 0.00,  0.00,  0.00), 4.0),
            self.sphere3D(313, 0.00, ( 0.00,  0.00,  0.00), 6.0),
            self.cloud(2, ( 2.50,  0.00,  0.00), 0.25),
            self.cloud(2, ( 4.50,  0.00,  0.00), 0.25),
        ])


''' ----------------------------------------------------------------------------------------- '''
def save(dataset, filename):
    """
    Visualizes and saves a dataset to datasets/ directory.
    If the file exists, prompts the user before overwriting.

    Args:
        dataset (Dataset): The dataset instance to save.
        filename (str): Name of the file (without path).
    """
    # Ensure datasets directory exists
    os.makedirs("datasets", exist_ok=True)
    filepath = os.path.join("datasets", filename)

    # 1. Visualize the dataset before saving
    Visualizer(
        layout=(1, 1),
        plot_specs=[{"clusters": dataset.clusters, "kwargs": {"title": filename}}]
    ).show()

    # 2. If file exists, ask for overwrite
    if os.path.exists(filepath):
        choice = input(f"File '{filename}' already exists. Overwrite? (y/n): ").strip().lower()
        if choice != 'y':
            print("Save cancelled.")
            return

    # 3. Save dataset
    torch.save({
        "data": dataset.data,
        "labels": dataset.labels,
        "clusters": dataset.clusters,
        "cluster_sizes": dataset.cluster_sizes
    }, filepath)

    print(f"Dataset saved to {filepath}")

def load(filename):
    """
    Loads a dataset from datasets/ directory and returns it as a Dataset instance.
    """
    filepath = os.path.join("datasets", filename)
    if not os.path.exists(filepath):
        raise FileNotFoundError(f"Dataset file '{filename}' not found in 'datasets/'.")

    saved = torch.load(filepath, weights_only=False)

    # Create an empty Dataset-like object without running __init__
    dataset = object.__new__(Dataset)
    dataset.data = saved["data"]
    dataset.labels = saved["labels"]
    dataset.clusters = saved["clusters"]
    dataset.cluster_sizes = saved["cluster_sizes"]

    print(f"Dataset '{filename}' loaded from {filepath}")
    return dataset

if __name__ == "__main__":
    # Dataset Showcase
    # Visualizer(
    # layout=(2, 4), 
    #     plot_specs=[
    #         {"clusters": Ds2Dbut1Dsingle().clusters, "kwargs": {"title": "Ds2Dbut1Dsingle"}},
    #         {"clusters": Ds2Dbut2Dsingle().clusters, "kwargs": {"title": "Ds2Dbut2Dsingle"}},
    #         {"clusters": Ds2Dbut2Dmulti().clusters,  "kwargs": {"title": "Ds2Dbut2Dmulti"}},
    #         {"clusters": Ds3Dbut1Dsingle().clusters, "kwargs": {"title": "Ds3Dbut1Dsingle"}},
    #         {"clusters": Ds3Dbut2Dsingle().clusters, "kwargs": {"title": "Ds3Dbut2Dsingle"}},
    #         {"clusters": Ds3Dbut2Dmulti().clusters,  "kwargs": {"title": "Ds3Dbut2Dmulti"}},
    #         {"clusters": Ds3Dbut3Dsingle().clusters, "kwargs": {"title": "Ds3Dbut3Dsingle"}},
    #         {"clusters": Ds3Dbut3Dmulti().clusters,  "kwargs": {"title": "Ds3Dbut3Dmulti"}},
    #         ]
    # ).show()
    # Visualizer(
    # layout=(2, 3),
    #     plot_specs=[
    #         {"clusters": Ds2DMoons().clusters,     "kwargs": {"title": "Ds2DMoons"}},
    #         {"clusters": Ds3DMoons().clusters,     "kwargs": {"title": "Ds3DMoons"}},
    #         {"clusters": DsTrue3DMoons().clusters, "kwargs": {"title": "DsTrue3DMoons"}},
    #         {"clusters": Ds2DSwissRoll().clusters, "kwargs": {"title": "Ds2DSwissRoll"}},
    #         {"clusters": Ds3DSwissRoll().clusters, "kwargs": {"title": "Ds3DSwissRoll"}},
    #         ]
    # ).show()
    # Visualizer(
    # layout=(1, 2),
    #     plot_specs=[
    #         {"clusters": Ds3DTorus().clusters,  "kwargs": {"title": "Ds3DTorus"}},
    #         {"clusters": Ds3DSphere().clusters, "kwargs": {"title": "Ds3DSphere"}},
    #         ]
    # ).show()
    #save(Ds3DSphere(), "Ds3DSphere.pt")
    Visualizer(
    layout=(1, 1),
        plot_specs=[
            {"clusters": load("Ds3DSphere.pt").clusters,  "kwargs": {"title": "3DSphere"}},
            ]
    ).show()