from trainer import Trainer
from visualizer import Visualizer

from model import Ae2Dto1D, Ae2Dto1Dlinear, Ae2Dto1Dwide
from model import Ae3Dto1D
from model import Ae3Dto2D, Ae3Dto2Dlinear, Ae3Dto2Dwide
from model import StandardVAE, MixtureVAE

from dataset import Ds2Dbut2Dsingle, Ds2Dbut1Dsingle, Ds2Dbut2Dmulti
from dataset import Ds3Dbut3Dsingle, Ds3Dbut2Dsingle, Ds3Dbut1Dsingle, Ds3Dbut2Dmulti
from dataset import Ds3Dbut3Dsingle, Ds3Dbut2Dsingle, Ds3Dbut1Dsingle, Ds3Dbut2Dmulti, Ds3Dbut3Dmulti
from dataset import Ds2DMoons, Ds3DMoons, DsTrue3DMoons
from dataset import Ds2DSwissRoll, Ds3DSwissRoll
from dataset import Ds3DTorus
from dataset import Ds3DSphere


trainer = Trainer(
    model=StandardVAE(3, 64, 2),
    dataset=Ds3DTorus(),
    losses={'vae': 1.0},
    sample_neighbors=True,
    batch_size=8,
    ).train()

Visualizer(
    layout=(1, 3), 
    plot_specs=[
        {"clusters": trainer.get("input"),  "kwargs": {"title": "Input Data"}},
        {"clusters": trainer.get("latent"), "kwargs": {"title": "Latent Representation"}},
        {"clusters": trainer.get("output"), "kwargs": {"title": "Reconstructed Data"}},
        ]
    ).show()
