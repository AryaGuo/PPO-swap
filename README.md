# PPO-swap
[![DOI](https://zenodo.org/badge/922004113.svg)](https://doi.org/10.5281/zenodo.14847752) [![License](https://img.shields.io/badge/License-Apache_2.0-blue.svg)](https://opensource.org/licenses/Apache-2.0)

This is the implementation of `PPO-swap` in the paper "Unified and Generalizable Reinforcement Learning for Facility Location Problems on Graphs" (WWW 2025). In this paper, we propose a deep reinforcement learning algorithm to solve the facility relocation problem (FRP) and the p-median problem (PMP) in a swap-based manner. 

## Dependencies

The required packages are specified in `environment.yml`. 
  - python=3.9.15
  - numpy=1.21.5
  - pytorch=1.13.0
  - pytorch-cuda=11.6
  - pytorch-lightning=1.7.7
  - pyg=2.2.0
  - networkx=2.8.4
  - gurobi=10.0.0
  - yaml=0.2.5

## Usage

### Preparing data

```
python gen_data.py
```

By default, it generates graphs with 100 nodes saved at `./data` directory.

### Training
```
python train.py
```
The default configuration file is `config/train.yaml`.

### Evaluation
For facility relocation problem (FRP), run
```
python eval_frp.py
```

For p-median problem (PMP), run
```
python eval_pmp.py
```
The configuration files are `config/eval_frp.yaml` and `config/eval_pmp.yaml` respectively.
