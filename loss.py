import torch
import torch.nn as nn
import torch.nn.functional as F

class TrustworthinessLoss(nn.Module):
    """
    Differentiable approximation of the trustworthiness score as a PyTorch loss.

    This loss penalizes "intrusions" in the embedding space (Z) that were not true
    neighbors in the original data (X), using soft rankings and soft top-k masks.

    Args:
        k (int): Number of neighbors to consider for trustworthiness.
        tau_r (float): Temperature for soft rank approximation in the original space.
        tau_s (float): Temperature for soft top-k mask in the embedding space.
    """
    def __init__(self, k = 5, tau_r = 0.5, tau_s = 0.5):
        super().__init__()
        self.k = k                # neighborhood size
        self.tau_r = tau_r        # rank smoothness temperature
        self.tau_s = tau_s        # selection smoothness temperature

    def forward(self, X, Z):
        """
        Compute the soft trustworthiness loss.

        Args:
            X (Tensor): Original data matrix of shape (n_samples, input_dim).
            Z (Tensor): Embedded data matrix of shape (n_samples, latent_dim).

        Returns:
            Tensor: Scalar loss value.
        """
        n = X.size(0)
        # 1) Pairwise Euclidean distances in original and embedding spaces
        D_X = torch.cdist(X, X, p=2)  # shape: (n, n)
        D_Z = torch.cdist(Z, Z, p=2)  # shape: (n, n)

        # 2) Soft rank approximation in the original space
        #    diffs_X[i,j,l] = D_X[i,l] - D_X[i,j]
        diffs_X = D_X.unsqueeze(1) - D_X.unsqueeze(2)  # shape: (n, n, n)
        #    P ≈ 1 if D_X[i,l] ≤ D_X[i,j], else ≈0
        P = torch.sigmoid(diffs_X / self.tau_r)
        #    soft_rank_X[i,j] = 1 + Σ P over l
        soft_rank_X = 1 + P.sum(dim=2)                  # shape: (n, n)
        #    intrusion penalty: values >0 only if rank > k
        intrusion = F.relu(soft_rank_X - self.k)        # shape: (n, n)

        # 3) Soft top-k membership in the embedding space
        W_Z = self.soft_topk_weights(D_Z)
        #    S[i,j] is high for points not in the top-k of Z
        S = 1 - W_Z                                     # shape: (n, n)

        # 4) Combine intrusion with exclusion mask and normalize
        #    This mirrors the classic trustworthiness normalization constant
        loss_matrix = intrusion * S
        norm = 2.0 / (n * self.k * (2 * n - 3 * self.k - 1))
        return norm * loss_matrix.sum()

    def soft_topk_weights(self, distances):
        """
        Create a differentiable soft top-k mask from pairwise distances.

        Args:
            distances (Tensor): Pairwise distance matrix of shape (n, n).

        Returns:
            Tensor: Soft mask W of shape (n, n), where W[i,j] ≈ 1 if j is among
                    the top-k nearest neighbors of i, else ≈ 0.
        """
        k = self.k
        n = distances.size(0)
        W = torch.zeros_like(distances)
        # Compute soft rank for each anchor i
        for i in range(n):
            d = distances[i]  # distances from point i to all points (shape: n)
            # diffs[j,l] = d[j] - d[l]
            diffs = d.unsqueeze(1) - d.unsqueeze(0)    # shape: (n, n)
            # P[j,l] ≈ 1 if d[j] ≤ d[l]
            P = torch.sigmoid(-diffs / self.tau_s)
            # soft rank of each candidate j
            R = 1 + P.sum(dim=1)                       # shape: (n,)
            # weight ≈ 1 for ranks ≤ k, decays to 0 at rank = k+1
            W[i] = torch.clamp((k + 1 - R) / k, 0.0, 1.0)
        return W

class LLELoss(nn.Module):
    """
    Differentiable LLE-style loss for autoencoders.

    Args:
        k: number of neighbors for local reconstruction
        reg: regularization term for numerical stability
    """
    def __init__(self, k=10, reg=1e-3):
        super(LLELoss, self).__init__()
        self.k = k
        self.reg = reg

    def forward(self, X, Z):
        """
        X: input data tensor, shape (n, d_in)
        Z: latent codes tensor, shape (n, d_lat)
        """
        n = X.size(0)

        # pairwise distances in input space
        D = torch.cdist(X, X)

        # k+1 nearest neighbors (including self), then drop self
        knn = torch.topk(D, self.k + 1, largest=False).indices
        neighbors = knn[:, 1:]

        # solve reconstruction weights for each point
        W = torch.zeros(n, self.k, device=X.device, dtype=X.dtype)
        I = torch.eye(self.k, device=X.device, dtype=X.dtype)
        for i in range(n):
            Xi = X[i : i + 1]
            Xj = X[neighbors[i]]
            C = (Xj - Xi) @ (Xj - Xi).t()
            C += self.reg * I
            ones = torch.ones(self.k, device=X.device, dtype=X.dtype)
            w = torch.linalg.solve(C, ones)
            W[i] = w / w.sum()

        # reconstruct each Z[i] from its neighbors
        Z_neigh = Z[neighbors]              # shape: (n, k, d_lat)
        Z_recon = (W.unsqueeze(-1) * Z_neigh).sum(dim=1)

        # mean squared error between Z and its local reconstruction
        return F.mse_loss(Z_recon, Z)
    