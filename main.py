from dataset import load
from trainer import Trainer
from visualizer import Visualizer

from model import AE, VAE

from dataset import (
    Ds2Dbut2Dsingle, Ds2Dbut1Dsingle, Ds2Dbut2Dmulti,
    Ds3Dbut3Dsingle, Ds3Dbut2Dsingle, Ds3Dbut1Dsingle, Ds3Dbut2Dmulti, Ds3Dbut2Dsingle,
    Ds2DMoons, Ds3DMoons, DsTrue3DMoons,
    Ds2DSwissRoll, Ds3DSwissRoll,
    Ds3DTorus,
    Ds3DSphere,
)

# 2D
# (2, [32, 16, 8], 1)
# Ds2Dbut2Dsingle, Ds2Dbut2Dmulti, Ds2DMoons, Ds2DSwissRoll

# 3D
# (3, [64, 32, 16], 2)
# Ds3Dbut2Dsingle, Ds3Dbut2Dmulti, Ds3DMoons, Ds3DSwissRoll, Ds3DTorus, Ds3DSphere

trainer = Trainer(
    model=AE(3, [64, 32, 16], 2),
    dataset=load("Ds3DSwissRoll.pt"),
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
