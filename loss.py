import torch
import torch.nn as nn
import torch.nn.functional as F

class GraphLaplacianLoss(nn.Module):
    """
    Graph‐Laplacian regularization for autoencoders.

    Args:
        k: number of neighbors for graph construction
        sigma: bandwidth for Gaussian kernel
    """
    def __init__(self, k=8, sigma=1.0):
        super().__init__()
        self.k = k
        self.sigma = sigma

    def forward(self, X, Z):
        """
        X: input data tensor, shape (n, d_in)
        Z: latent codes tensor, shape (n, d_lat)
        """
        n = X.size(0)
        # pairwise distances in input space
        D = torch.cdist(X, X, p=2)  # (n, n)
        # find k+1 nearest (including self), then drop self
        knn = torch.topk(D, self.k+1, largest=False).indices
        neighbors = knn[:, 1:]      # (n, k)

        # build adjacency matrix W
        W = torch.zeros(n, n, device=X.device, dtype=X.dtype)
        for i in range(n):
            for j in neighbors[i]:
                w_ij = torch.exp(- (D[i, j] ** 2) / (self.sigma ** 2))
                W[i, j] = w_ij
                W[j, i] = w_ij  # make symmetric

        # compute pairwise squared distances in latent space
        diff = Z.unsqueeze(1) - Z.unsqueeze(0)    # (n, n, d_lat)
        sqdist = diff.pow(2).sum(dim=2)            # (n, n)

        # Laplacian loss = sum_{i,j} W[i,j] * ||z_i - z_j||^2
        return (W * sqdist).sum() / (n * self.k)

class TripletMarginLoss(nn.Module):
    """
    Computes triplet margin loss between an anchor, positive, and negative example.

    This module wraps `torch.nn.TripletMarginLoss`, but automatically constructs
    triplets by treating the first example in a batch as the anchor, and sampling
    positives and negatives from the remainder of the batch.

    Args:
        margin (float, optional): Margin value for the triplet loss.
        p (float, optional): The norm degree for pairwise distance.
    """
    def __init__(self, margin=1.0, p=2):
        super().__init__()
        self.tri_loss = nn.TripletMarginLoss(margin=margin, p=p)
    
    def forward(self, X, labels):
        """
        Constructs triplets from the batch and computes loss.

        The first element in X (and labels) is used as the anchor. Other elements
        in the batch whose labels match the anchor form the positive set; those
        with different labels form the negative set. Triplets are formed by pairing
        the anchor with up to M positive-negatives pairs, where
        M = min(num_pos, num_neg). If there are fewer positives or negatives than
        the other, only M triplets are used.

        Args:
            X (torch.Tensor): Tensor of shape (N, D) containing N embeddings of
                              dimensionality D. The 0-th element is the anchor.
            labels (torch.Tensor): Long tensor of shape (N,) containing integer
                                   class labels for each embedding.

        Returns:
            torch.Tensor: Scalar tensor containing the average triplet margin loss
                          over the formed triplets.
        """
        # indices of positive and negative samples
        pos_idx = (labels == labels[0]).nonzero(as_tuple=False).flatten()
        neg_idx = (labels != labels[0]).nonzero(as_tuple=False).flatten()
        # remove the anchor index from positives
        pos_idx = pos_idx[pos_idx != 0]
        # how many triplets we can form
        M = min(pos_idx.size(0), neg_idx.size(0))
        # anchor: repeat the first sample M times
        anc = X[0].unsqueeze(0).repeat(M, 1)
        # take the first M positives and negatives
        pos = X[pos_idx[:M]]
        neg = X[neg_idx[:M]]
        return self.tri_loss(anc, pos, neg)

class VAELoss(nn.Module):
    """Computes the Kullback–Leibler divergence term for a Variational Autoencoder.

    This loss module calculates the KL divergence between a diagonal Gaussian
    posterior parameterized by `mu` and `logvar` and the standard normal prior.
    It returns the mean KL divergence per batch.
    """
    def __init__(self):
        super().__init__()

    def forward(self, mu, logvar):
        """Compute the KL divergence component of the VAE loss.

        Given the mean (`mu`) and log-variance (`logvar`) of the approximate
        posterior q(z|x), computes

            KL(q(z|x) || p(z)) = -0.5 * sum(1 + logvar - mu^2 - exp(logvar))

        across the latent dimensions, then averages over the batch.

        Args:
            mu (torch.Tensor): Tensor of shape (batch_size, latent_dim)
                representing the mean of the approximate posterior.
            logvar (torch.Tensor): Tensor of the same shape as `mu`
                representing the logarithm of the variance of the approximate posterior.

        Returns:
            torch.Tensor: A scalar tensor containing the mean KL divergence
            over the batch.
        """
        # KL divergence per sample
        kl_per_sample = -0.5 * torch.sum(
            1 + logvar - mu.pow(2) - logvar.exp(),
            dim=1
        )
        # Mean over the batch
        kl = kl_per_sample.mean()
        return kl

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
    def __init__(self, k=10, reg=1e-6):
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
    