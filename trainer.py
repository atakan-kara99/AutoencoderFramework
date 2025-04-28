import loss
import torch
import torch.nn as nn
import torch.optim as optim

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
    """
    def __init__(self, model, dataset, learning_rate=1e-3,
                 num_epochs=-1, patience=40, min_delta=1e-5,
                 print_every=20, batch_size=32):
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

        # Define loss function and optimizer
        self.mse_loss = nn.MSELoss()
        self.cosine_sim = nn.CosineSimilarity(dim=1, eps=1e-8)
        self.softTrust_loss = loss.SoftTrustworthinessLoss(k=10, tau_r=0.5, tau_s=0.5)
        self.optimizer = optim.Adam(self.model.parameters(), lr=self.learning_rate)

    def train(self):
        """
        Run the training loop over epochs or until early stopping.

        Returns:
            self: Enables method chaining after training.
        """
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
            epoch_loss = 0.0
            epoch_loss_mse = 0.0
            epoch_loss_cos = 0.0
            epoch_loss_st  = 0.0
            num_batches = 0

            # Mini-batch training
            for start in range(0, N, self.batch_size):
                # Select batch indices and data
                idx = perm[start:start + self.batch_size]
                batch = self.dataset.data[idx]

                # Zero gradient buffers
                self.optimizer.zero_grad()
                # Forward pass through model
                output = self.model(batch)
                # Compute losses
                mse_loss = self.mse_loss(output, batch)
                cos_loss = (1.0 - self.cosine_sim(output, batch)).mean() * 10
                st_loss  = self.softTrust_loss(output, batch) / 10
                loss = mse_loss
                # Backpropagation
                loss.backward()
                # Update model parameters
                self.optimizer.step()

                # Accumulate loss
                epoch_loss += loss.item()
                epoch_loss_mse += mse_loss.item()
                epoch_loss_cos += cos_loss.item()
                epoch_loss_st  += st_loss.item()
                num_batches += 1

            # Compute average loss for this epoch
            avg_loss = epoch_loss / num_batches
            mse_loss = epoch_loss_mse / num_batches
            cos_loss = epoch_loss_cos / num_batches
            st_loss  = epoch_loss_st / num_batches

            # Print average loss at specified intervals
            if epoch % self.print_every == 0:
                total_epochs = '∞' if use_early_stopping else self.num_epochs
                print(f"Epoch [{epoch}/{total_epochs}]   "
                      f"Total Loss: {avg_loss:.6f}   "
                      f"MSE Loss: {mse_loss:.6f}   "
                      f"Cosine Loss: {cos_loss:.6f}   "
                      f"sTrust Loss: {st_loss:.6f}   "
                      f"TotalTotal Loss: {mse_loss + cos_loss + st_loss:.6f}")

            # Early stopping check
            if use_early_stopping:
                # Check if loss has improved by at least min_delta
                if avg_loss + self.min_delta < best_loss:
                    best_loss = avg_loss
                    epochs_no_improve = 0
                else:
                    epochs_no_improve += 1

                # Trigger early stopping if no improvement for patience epochs
                if epochs_no_improve >= self.patience:
                    print(f"Early stopping triggered at epoch {epoch} (no improvement for {self.patience} epochs).")
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
        # Disable gradient computation for inference
        with torch.no_grad():
            if type == "input":
                d = self.dataset.data
            else:
                # Encode data to latent space
                d = self.model.encoder(self.dataset.data)
                if type == "latent":
                    # If latent has singleton feature dimension, squeeze it
                    if d.dim() >= 2 and d.size(1) == 1:
                        d = d.squeeze(1)
                elif type == "output":
                    # Decode latent back to output space
                    d = self.model.decoder(d)
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
