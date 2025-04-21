import torch
import torch.nn as nn
import torch.optim as optim

class Trainer:
    """
    Trainer orchestrates the training loop for an autoencoder-style model on a given dataset.

    Parameters:
        model (nn.Module): PyTorch model with encoder and decoder methods.
        dataset (object): Dataset providing 'data' Tensor of shape (N, ...) and 'cluster_sizes' list.
        learning_rate (float): Learning rate for the optimizer. Default: 1e-3.
        num_epochs (int): Number of training epochs. Default: 200.
        print_every (int): Interval (in epochs) at which to print training loss. Default: 20.
        batch_size (int): Number of samples per mini-batch. Default: 16.
    """
    def __init__(self, model, dataset, learning_rate=1e-3, num_epochs=200, print_every=20, batch_size=16):
        # Store dataset and model
        self.model = model
        self.dataset = dataset
        # Training hyperparameters
        self.learning_rate = learning_rate
        self.num_epochs = num_epochs
        self.print_every = print_every
        self.batch_size = batch_size

        # Define loss function and optimizer
        self.criterion = nn.MSELoss()
        self.optimizer = optim.Adam(self.model.parameters(), lr=self.learning_rate)
    
    def train(self):
        """
        Run the training loop over the specified number of epochs.

        For each epoch:
          - Shuffle data indices.
          - Loop over mini-batches.
          - Zero gradients, perform forward pass, compute loss, backpropagate, and update weights.
          - Accumulate batch losses to report average loss periodically.

        Returns:
            self: Enables method chaining after training.
        """
        # Total number of samples
        N = self.dataset.data.size(0)
        
        for epoch in range(self.num_epochs):
            # Generate a random permutation of indices for shuffling
            perm = torch.randperm(N)
            epoch_loss = 0.0
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
                # Compute reconstruction loss
                loss = self.criterion(output, batch)
                # Backpropagation
                loss.backward()
                # Update model parameters
                self.optimizer.step()

                # Accumulate loss
                epoch_loss += loss.item()
                num_batches += 1
            
            # Print average loss at specified intervals
            if (epoch + 1) % self.print_every == 0:
                avg_loss = epoch_loss / num_batches
                print(f"Epoch [{epoch+1}/{self.num_epochs}], Loss: {avg_loss:.6f}")

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
