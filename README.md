# MACE-PAN-PGS

**Physics-Aware Scalar Pathways Enhance the Accuracy and Stability of Equivariant Interatomic Potentials**

[![Paper](https://img.shields.io/badge/Paper-Submitted-blue)](https://github.com/JIABI/mace_librakan)
[![License](https://img.shields.io/badge/License-MIT%202.0-blue.svg)](https://opensource.org/licenses/mit)
[![Framework](https://img.shields.io/badge/Based%20on-MACE-green)](https://github.com/ACEsuit/mace)

## Table of Contents
- [About MACE-PAN-PGS](#about-mace-pan-pgs)
- [Key Innovations](#key-innovations)
  - [1. Physics-Aware Neighbourhood (PAN) Pooling](#1-physics-aware-neighbourhood-pan-pooling)
  - [2. Physics-Guided Spectral (PGS) Components](#2-physics-guided-spectral-pgs-components)
- [Installation](#installation)
- [Usage](#usage)
  - [Quick Start](#quick-start)
  - [PGS Configuration Mapping](#pgs-configuration-mapping)
- [Data Availability](#data-availability)
- [References](#references)
- [License](#license)

---

## About MACE-PAN-PGS

This repository implements **MACE-PAN-PGS**, an enhanced equivariant MPNN architecture designed to overcome the "scalar bottleneck" in standard force fields. By introducing physics-aware scalar pathways, we significantly improve stability in MD simulations and accuracy in complex chemical environments (defects, surfaces).

The architecture consists of two main contributions:
1.  **PAN Pooling:** A geometry-gated aggregation mechanism that restores local sensitivity.
2.  **PGS Layers:** Spectral decomposition modules that mitigate low-frequency bias in scalar features.

## Key Innovations

### 1. Physics-Aware Neighbourhood (PAN) Pooling
Standard MACE uses uniform summation (`scatter_sum`) for message aggregation. PAN introduces a learnable gating mechanism driven by local geometry:
$$h_i^{(0)} = \sum_{j \in \mathcal{N}(i)} \sigma(\text{MLP}(h_{ij}^{(0)}, r_{ij})) \cdot h_{ij}^{(0)}$$
This allows the model to dynamically weight neighbours based on bond distances and chemical identities, preserving $O(3)$ equivariance by operating solely on invariant scalars.

### 2. Physics-Guided Spectral (PGS) Components
We introduce PGS layers to enrich scalar representations using spectral transforms. In this implementation, we distinguish between two types of PGS:

* **Geometric PGS (Edge):** Applied to edge features. It decomposes radial signals using **Bessel Basis** (low-freq) and a **Physics Kernel** ($\exp(i\omega r)$) to capture high-frequency repulsive potentials.
* **Latent PGS (Readout/Node):** Applied to node/readout features. It treats feature vectors as latent signals and applies **Non-Uniform Fourier Transforms (NUFFT)** to capture complex energy fluctuations and scalar interactions.

## Installation

### Requirements
- Python >= 3.7
- PyTorch >= 1.12 (Recommended 1.13+ or 2.0+)
- [e3nn](https://e3nn.org)

### Installation from Source
Clone this repository and install in editable mode:

```sh
git clone [https://github.com/JIABI/mace_librakan.git](https://github.com/JIABI/mace_librakan.git)
cd mace_librakan
pip install -e .
```

## Usage
We provide a unified training script run_ag.sh to easily configure and run experiments with PAN and PGS components.

### Quick Start
To reproduce the full MACE-PAN-PGS model (activating Edge, Node, and Readout spectral components) on the example Ag dataset:

```sh
# Run training with the 'libra' mixer (Readout PGS) and enable Edge/Node PGS
bash run_ag.sh -m libra -e true -n true
```

## Data Availability

1. Example Data: A subset of the Silver (Ag) dataset is provided in the ag/ directory (Ag-train.xyz, Ag-valid.xyz, Ag-test.xyz) to demonstrate the training workflow.
2. Full Datasets: The complete training datasets for Si (Silicon) and LiF (Lithium Fluoride) used in the paper are available from the authors upon reasonable request due to storage limitations.

## References
If you use this code or the PAN/PGS architecture, please cite:
```sh
@article{Bi2024PANPGS,
  title={Physics-Aware Scalar Pathways Enhance the Accuracy and Stability of Equivariant Interatomic Potentials},
  author={Jia Bi and Samuel Pinilla and [Full Author List]},
  journal={Nature Communications (Submitted)},
  year={2024}
}
```

And the original MACE framework:
```sh
@inproceedings{Batatia2022mace,
  title={{MACE}: Higher Order Equivariant Message Passing Neural Networks for Fast and Accurate Force Fields},
  author={Batatia, Ilyes and et al.},
  booktitle={Advances in Neural Information Processing Systems},
  year={2022}
}
```

## Contact
For questions regarding the implementation, please open an issue or contact:
Jia Bi: Jia.Bi@stfc.ac.uk

## License
This project is distributed under the MIT License.




