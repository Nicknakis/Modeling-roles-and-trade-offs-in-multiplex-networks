# Modeling roles and trade-offs in multiplex networks

### Official Python 3.11.11 and PyTorch 2.5.1 implementation of the Multiplex Latent Trade-off Model (MLT).

---

## Description

A multiplex social network captures multiple types of social relations among the same set of people, with each layer representing a distinct type of relationship. Understanding the structure of such systems allows us to identify how social exchanges may be driven by a person's own attributes and actions (independence), the status or resources of others (dependence), and mutual influence between entities (interdependence). Characterizing structure in multiplex networks is challenging, as the distinct layers can reflect different yet complementary roles, with interdependence emerging across multiple scales. Here, we introduce the **Multiplex Latent Trade-off Model (MLT)**, a framework for extracting roles in multiplex social networks that accounts for independence, dependence, and interdependence. MLT defines roles as trade-offs, requiring each node to distribute its source and target roles across layers while simultaneously distributing community memberships within hierarchical, multi-scale structures.

---

## Repository structure

```
MLT/
├── README.md
├── requirements.txt
├── .gitignore
├── main.py                  # Entry point: data generation + training loop
└── src/
    ├── __init__.py
    ├── model.py             # LSM (MLT) model
    ├── spectral.py          # Spectral clustering initialization
    ├── nmf_utils.py         # Per-layer NMF helpers
    └── data_generator.py    # Directed multiplex SBM generator
```

| File | Purpose |
|---|---|
| `main.py` | Generates a synthetic 3-layer directed multiplex SBM and runs the MLT training loop. |
| `src/model.py` | The `LSM` class implementing MLT (bias scaling phase + full hierarchical, multi-scale phase). |
| `src/spectral.py` | `Spectral_clustering_init` — adjacency / normalized / MDS spectral initializations. |
| `src/nmf_utils.py` | `calculate_nmf_for_layer` — per-layer NMF factorization. |
| `src/data_generator.py` | `multiplex_sbm_edges_np` — directed multiplex stochastic block model. |

---

## Data Availability

Applying the MLT approach to **176 real-world multiplex networks**—comprising social, health, and economic layers—from villages in western Honduras, we find that core social-exchange principles emerge, alongside local, layer-specific, and multiscale communities.

Regarding data access, and in accordance with confidentiality constraints for human participants, academic researchers from established institutions may request the data (with IRB approval) by contacting the authors directly. These data are non-transferable to other investigators and are not for commercial use. Any release is subject to the policies in effect at Yale University and the Yale Institute for Network Science at the time of release.

The provided code, by default, operates on randomly generated networks under a directed multiplex SBM, in order to promote open research and to facilitate application to additional networks.

---

## Installation

### 1. Create a Python 3.11.11 environment with conda
```bash
conda create -n mlt python=3.11.11
```

### 2. Activate the environment
```bash
conda activate mlt
```

### 3. Install the required packages
```bash
pip install -r requirements.txt
```

---

## Learning embeddings for multiplex directed networks using MLT

**RUN:**
```bash
python main.py
```

By default this generates a synthetic directed multiplex SBM (3 layers, 120 nodes split into 3 communities of sizes 30 / 40 / 50) and fits the MLT model. Two checkpoints are saved at the project root during training:

- `Bias_model.pth` — checkpoint at the end of the bias-only scaling phase (epoch 1999)
- `Full_model.pth` — checkpoint of the best full MLT model across inner runs

To apply the model to your own network, replace the synthetic data block in `main.py` with your edge lists for each layer (`final_i_*`, `final_j_*`).

---

## Reference

[Modeling roles and trade-offs in multiplex networks](https://www.nature.com/articles/s41467-026-68896-1). Nikolaos Nakis, Sune Lehmann, Nicholas A. Christakis, and Morten Mørup, *Nature Communications*, 2026.

```bibtex
@article{nakis2026mlt,
  title={Modeling roles and trade-offs in multiplex networks},
  author={Nakis, Nikolaos and Lehmann, Sune and Christakis, Nicholas A. and M{\o}rup, Morten},
  journal={Nature Communications},
  year={2026}
}
```
