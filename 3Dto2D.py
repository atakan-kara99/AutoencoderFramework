from trainer import Trainer
from visualizer import Visualizer
from model import Ae3Dto2D, Ae3Dto2Dlinear
from dataset import Dataset3Dbut3D, Dataset3Dbut2D, Dataset3Dbut1D


trainer = Trainer(
    model=Ae3Dto2D(),
    dataset=Dataset3Dbut3D(),
    learning_rate=1e-3,
    num_epochs=200,
    print_every=20,
    batch_size=16,
    ).train()

Visualizer(
    layout=(1, 3), 
    plot_specs=[
        {"type": "3d", "clusters": trainer.get("input"),  "kwargs": {"title": "Input Data"}},
        {"type": "2d", "clusters": trainer.get("latent"), "kwargs": {"title": "Latent Representation"}},
        {"type": "3d", "clusters": trainer.get("output"), "kwargs": {"title": "Reconstructed Data"}},
        ]
    ).show()
