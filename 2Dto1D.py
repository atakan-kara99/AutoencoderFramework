from trainer import Trainer
from visualizer import Visualizer
from model import Ae2Dto1D, Ae2Dto1Dlinear
from dataset import Dataset2Dbut2D, Dataset2Dbut1D


trainer = Trainer(
    model=Ae2Dto1D(),
    dataset=Dataset2Dbut2D(),
    learning_rate=1e-3,
    num_epochs=-1,
    patience=20,
    min_delta=1e-5,
    print_every=20,
    batch_size=16,
    ).train()

Visualizer(
    layout=(1, 3), 
    plot_specs=[
        {"type": "2d", "clusters": trainer.get("input"),  "kwargs": {"title": "Input Data"}},
        {"type": "1d", "clusters": trainer.get("latent"), "kwargs": {"title": "Latent Representation"}},
        {"type": "2d", "clusters": trainer.get("output"), "kwargs": {"title": "Reconstructed Data"}},
        ]
    ).show()
