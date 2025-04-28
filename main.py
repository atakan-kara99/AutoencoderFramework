from trainer import Trainer
from visualizer import Visualizer

# 2D to 1D
from model import Ae2Dto1D, Ae2Dto1Dlinear
from dataset import Ds2Dbut2Dsingle, Ds2Dbut1Dsingle, Ds2Dbut2Dmulti

# 3D to 1D
from model import Ae3Dto1D
from dataset import Ds3Dbut3Dsingle, Ds3Dbut2Dsingle, Ds3Dbut1Dsingle, Ds3Dbut2Dmulti

# 3D to 2D
from model import Ae3Dto2D, Ae3Dto2Dlinear
from dataset import Ds3Dbut3Dsingle, Ds3Dbut2Dsingle, Ds3Dbut1Dsingle, Ds3Dbut2Dmulti, Ds3Dbut3Dmulti

# Moons
from dataset import Ds2DMoons, Ds3DMoons, DsTrue3DMoons

# Swiss Roll
from dataset import Ds3DSwissRoll


trainer = Trainer(
    model=Ae3Dto2D(),
    dataset=Ds3DSwissRoll(),
    ).train()

Visualizer(
    layout=(1, 3), 
    plot_specs=[
        {"clusters": trainer.get("input"),  "kwargs": {"title": "Input Data"}},
        {"clusters": trainer.get("latent"), "kwargs": {"title": "Latent Representation"}},
        {"clusters": trainer.get("output"), "kwargs": {"title": "Reconstructed Data"}},
        ]
    ).show()
