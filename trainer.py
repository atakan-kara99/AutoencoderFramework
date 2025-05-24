import loss
import torch
import numpy as np
import torch.nn as nn
import torch.optim as optim
from sklearn.neighbors import NearestNeighbors
from visualizer import Visualizer

class Trainer:
    """
    Trainer orchestrates the training loop for an autoencoder-style model on a given dataset.

    Parameters:
        model (nn.Module): PyTorch model with encoder and decoder methods.
        dataset (object): Dataset providing 'data' Tensor of shape (N, ...) and 'cluster_sizes' list.
        learning_rate (float): Learning rate for the optimizer.
        num_epochs (int): Number of training epochs. Use -1 to enable early stopping.
        patience (int): Number of epochs with no improvement after which training will be stopped when num_epochs=-1.
        min_delta (float): Minimum change in the monitored loss to qualify as an improvement.
        print_every (int): Interval (in epochs) at which to print training loss.
        batch_size (int): Number of samples per mini-batch.
        losses (Dict[str, float], optional): Mapping from loss-names to their weights.
            Keys must be a subset of ['mse', 'cos', 'trust', 'lle', 'kld'].
        sample_neighbors (bool): If True, samples neighbors for mini-batch training.
    """
    def __init__(self, model, dataset, learning_rate=1e-3,
                 num_epochs=-1, patience=40, min_delta=1e-6,
                 print_every=20, batch_size=16, losses={'mse': 1.0},
                 sample_neighbors=False):
        # Store dataset and model
        self.model = model
        self.dataset = dataset
        # Training hyperparameters
        self.learning_rate = learning_rate
        self.num_epochs = num_epochs
        self.patience = patience
        self.min_delta = min_delta
        self.print_every = print_every
        self.batch_size = batch_size
        self.sample_neighbors = sample_neighbors
        self.losses = losses
        # Define loss function and optimizer
        self.mse_loss = nn.MSELoss()
        self.cos_sim = nn.CosineSimilarity()
        self.trust_loss = loss.TrustworthinessLoss(k=4, tau_r=0.1, tau_s=0.1)
        self.lle_loss = loss.LLELoss(k=4)
        self.kld_loss = nn.KLDivLoss(reduction='batchmean')
        self.vae_loss = loss.VAELoss()
        self.tri_loss = loss.TripletMarginLoss()
        self.lap_loss = loss.GraphLaplacianLoss(k=4, sigma=1.0)
        # Initialize optimizer
        self.optimizer = optim.Adam(model.parameters(), lr=learning_rate)
        # precompute neighbors if requested
        if self.sample_neighbors:
            nbrs = NearestNeighbors(n_neighbors=batch_size,
                                    algorithm='auto').fit(dataset.data)
            # neighbor_indices[i] is an array of indices (including i itself) closest to i
            self.neighbor_indices = nbrs.kneighbors(dataset.data, return_distance=False)

    def train(self):
        """
        Run the training loop over epochs or until early stopping.

        Returns:
            self: Enables method chaining after training.
        """
        model = self.model
        # Total number of samples
        N = self.dataset.data.size(0)
        # Best loss for early stopping
        best_loss = float('inf')
        # Counter for epochs without improvement
        epochs_no_improve = 0
        epoch = 0

        # Determine if using fixed epochs or early stopping
        use_early_stopping = (self.num_epochs < 0)
        max_epochs = float('inf') if use_early_stopping else self.num_epochs

        # Training loop
        while epoch < max_epochs:
            epoch += 1
            # Shuffle data indices for this epoch
            perm = torch.randperm(N)
            num_batches = 0

            epoch_totals = {name: 0.0 for name in self.losses}
            epoch_totals['total'] = 0.0

            # Mini-batch training
            for start in range(0, N, self.batch_size):
                # Select batch indices and data
                if self.sample_neighbors:
                    # choose one random anchor, then its nearest‐neighbor block
                    anchor = np.random.randint(0, N)
                    idx = torch.from_numpy(self.neighbor_indices[anchor])
                else:
                    idx = perm[start:start + self.batch_size]
                batch = self.dataset.data[idx]
                labels = self.dataset.labels[idx]

                # # Optional: visualize the batch
                # Visualizer(
                #     layout=(1, 1),
                #     plot_specs=[{"clusters": [self.dataset.data, test], "kwargs": {"title": "Batch"}}]
                # ).show()

                # Zero gradient buffers
                self.optimizer.zero_grad()

                # Forward pass through model
                output, latent, mu, logvar = model(batch)

                # Compute only selected raw losses
                raw = {}   
                if "mse"   in self.losses: raw["mse"]   = self.mse_loss(output, batch)
                if "cos"   in self.losses: raw["cos"]   = (1.0 - self.cos_sim(output, batch)).mean()
                if "trust" in self.losses: raw["trust"] = self.trust_loss(output, batch)
                if "lle"   in self.losses: raw["lle"]   = self.lle_loss(output, batch)
                if "kld"   in self.losses: raw["kld"]   = self.kld_loss(output.log_softmax(dim=1), batch.softmax(dim=1))
                if "vae"   in self.losses: raw["vae"]   = self.vae_loss(mu, logvar)
                if "tri"   in self.losses: raw["tri"]   = self.tri_loss(output, labels)
                if "lap"   in self.losses: raw["lap"]   = self.lap_loss(output, batch)

                # Apply weights
                weighted = {
                    name: self.losses[name] * v
                    for name, v in raw.items()
                }
                loss = sum(weighted.values())
                
                # Backpropagation
                loss.backward()
                # Update model parameters
                self.optimizer.step()

                # Accumulate loss
                epoch_totals['total'] += loss.item()
                for name, val in weighted.items():
                    epoch_totals[name] += val.item()
                num_batches += 1

            # Compute average loss for this epoch
            avg = {name: epoch_totals[name] / num_batches
                   for name in epoch_totals}

            # Print average loss at specified intervals
            if epoch % self.print_every == 0:
                losses_str = "   ".join(f"{name.upper():>4}={avg[name]:.6f}"
                                       for name in self.losses)
                total_epochs = '∞' if use_early_stopping else self.num_epochs
                print(f"EPOCH [{epoch}/{total_epochs}]   "
                      f"TOTAL={avg['total']:.6f}   {losses_str}")

            # Early stopping check
            if use_early_stopping:
                if avg['total'] + self.min_delta < best_loss:
                    best_loss = avg['total']
                    epochs_no_improve = 0
                else:
                    epochs_no_improve += 1
                if epochs_no_improve >= self.patience:
                    print(f"Early stopping at epoch {epoch} "
                          f"(no improvement for {self.patience} epochs).")
                    break

        return self

    def get(self, type):
        """
        Extracts data segments from the model or input based on 'type' specifier.

        Parameters:
            type (str): One of:
                - 'input': returns raw input data segments.
                - 'latent': returns encoded latent representations.
                - 'output': returns reconstructed outputs from decoder.

        Returns:
            List[Tensor]: A list of Tensors segmented according to dataset.cluster_sizes.

        Raises:
            ValueError: If 'type' is not recognized.
        """
        model = self.model
        data = self.dataset.data
        # Disable gradient computation for inference
        with torch.no_grad():
            if type == "input":
                d = data
            else:
                # Encode data
                output, latent, _, _ = model(data)

                if type == "latent":
                    d = latent
                    # If latent has singleton feature dimension, squeeze it
                    if d.dim() >= 2 and d.size(1) == 1:
                        d = d.squeeze(1)
                elif type == "output":
                    # Decode latent back to output space
                    d = output
                else:
                    raise ValueError(f"Unknown type={type!r}, expected 'input', 'latent' or 'output'")

        # Segment output according to cluster sizes
        segments = []
        offset = 0
        for length in self.dataset.cluster_sizes:
            segments.append(d[offset:offset+length])
            offset += length
        # If there are leftover samples, include them as an extra segment
        if offset < d.size(0):
            segments.append(d[offset:])
        return segments
