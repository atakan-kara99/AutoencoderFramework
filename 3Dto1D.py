from trainer import Trainer
from visualizer import Visualizer
from model import Ae3Dto1D
from dataset import Ds3Dbut3Dsingle, Ds3Dbut2Dsingle, Ds3Dbut1Dsingle, Ds3Dbut2Dmulti


trainer = Trainer(
    model=Ae3Dto1D(),
    dataset=Ds3Dbut2Dmulti(),
    ).train()

Visualizer(
    layout=(1, 3), 
    plot_specs=[
        {"clusters": trainer.get("input"),  "kwargs": {"title": "Input Data"}},
        {"clusters": trainer.get("latent"), "kwargs": {"title": "Latent Representation"}},
        {"clusters": trainer.get("output"), "kwargs": {"title": "Reconstructed Data"}},
        ]
    ).show()
