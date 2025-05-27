from trainer import Trainer
from visualizer import Visualizer

from model import AE, VAE

from dataset import (
    Ds2Dbut2Dsingle, Ds2Dbut1Dsingle, Ds2Dbut2Dmulti,
    Ds3Dbut3Dsingle, Ds3Dbut2Dsingle, Ds3Dbut1Dsingle, Ds3Dbut2Dmulti, Ds3Dbut3Dmulti,
    Ds2DMoons, Ds3DMoons, DsTrue3DMoons,
    Ds2DSwissRoll, Ds3DSwissRoll,
    Ds3DTorus,
    Ds3DSphere,
)

trainer = Trainer(
    model=AE(3, 64, 2),
    dataset=Ds3DTorus(),
    losses={'tri': 1.0},
    sample_neighbors=False,
    batch_size=32,
    ).train()

Visualizer(
    layout=(1, 3), 
    plot_specs=[
        {"clusters": trainer.get("input"),  "kwargs": {"title": "Input Data"}},
        {"clusters": trainer.get("latent"), "kwargs": {"title": "Latent Representation"}},
        {"clusters": trainer.get("output"), "kwargs": {"title": "Reconstructed Data"}},
        ]
    ).show()
