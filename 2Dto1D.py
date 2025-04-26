from trainer import Trainer
from visualizer import Visualizer
from model import Ae2Dto1D, Ae2Dto1Dlinear
from dataset import Ds2Dbut2Dsingle, Ds2Dbut1Dsingle, Ds2Dbut2Dmulti


trainer = Trainer(
    model=Ae2Dto1D(),
    dataset=Ds2Dbut2Dmulti(),
    ).train()

Visualizer(
    layout=(1, 3), 
    plot_specs=[
        {"clusters": trainer.get("input"),  "kwargs": {"title": "Input Data"}},
        {"clusters": trainer.get("latent"), "kwargs": {"title": "Latent Representation"}},
        {"clusters": trainer.get("output"), "kwargs": {"title": "Reconstructed Data"}},
        ]
    ).show()
