# Autoencoder Training Framework

**Framework for exploring the performance of autoencoders with different loss objectives preserving in-between instances on synthetic datasets.**

This repository provides a flexible experimental environment to investigate how different loss functions influence **intermediate (in-between) sample preservation**, **manifold structure**, and **continuity in latent space** for Autoencoders (AE) and Variational Autoencoders (VAE).
It includes synthetic datasets, a modular training engine, and visualization utilities designed to analyze how well models reconstruct transitions between data clusters and preserve geometric relationships.

**NOTE: THIS CODE IS RELATED TO MY [MASTER THESIS](https://github.com/atakan-kara99/MasterThesis).**

---

## ✨ Key Ideas

* Study how **loss functions shape latent manifolds**
* Evaluate **in-between samples**, not only endpoints
* Use **synthetic manifolds & clusters** for controlled experiments
* Compare geometry-aware & contrastive objectives
* Visualize **input → latent → decoded** space and cluster transitions

Losses include:

* MSE
* Cosine similarity & contrastive losses
* Triplet margin loss
* Graph Laplacian regularization
* Trustworthiness loss
* LLE-style neighborhood reconstruction loss
* KL divergence (VAE)

---

## 📁 Structure

```
model.py       # AE and VAE models
loss.py        # manifold, contrastive, and probabilistic losses
dataset.py     # synthetic datasets (Swiss roll, Moons, Sphere, Torus, etc.)
trainer.py     # multi-loss training pipeline + early stopping
visualizer.py  # 1D / 2D / 3D scatter visualization
main.py        # example experiment
```

---

## 🚀 Running an Experiment

```python
from model import AE
from dataset import load
from trainer import Trainer
from visualizer import Visualizer

trainer = Trainer(
    model=AE(3, [64,32,16], 2),
    dataset=load("Ds3DSwissRoll.pt"),
    losses={'tri': 1.0},   # choose your objective(s)
    batch_size=32,
).train()

Visualizer(
    layout=(1,3),
    plot_specs=[
        {"clusters": trainer.get("input"),  "kwargs": {"title": "Input"}},
        {"clusters": trainer.get("latent"), "kwargs": {"title": "Latent"}},
        {"clusters": trainer.get("output"), "kwargs": {"title": "Reconstructed"}},
    ]
).show()
```

---

## 🔬 Synthetic Datasets Included

* Gaussian clusters (2D / 3D)
* Moons (2D / 3D)
* Swiss Roll (2D / 3D)
* Torus
* Sphere shells

These datasets allow controlled tests of geometry & cluster interpolation behavior.

---

## 🎯 Research Questions You Can Explore

* Which loss functions preserve **inter-cluster relationships**?
* Which losses maintain **smooth paths** between clusters?
* Can AEs learn **latent topological structure** from toy data?

---

## 📦 Installation

```bash
pip install torch numpy matplotlib scikit-learn
```

---

## 🛠️ Configure Loss Objectives

Example mixing reconstruction + contrastive + geometric losses:

```python
losses = {
    'mse': 1.0,
    'tri': 1.0,
    'trust': 0.2,
}
```

---

## 📊 Visual Output

You can visualize:

* Original distribution
* Latent embedding
* Reconstructed distribution
* In-between samples and cluster transitions
