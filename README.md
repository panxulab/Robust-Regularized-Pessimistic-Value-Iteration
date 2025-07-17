# Robust-Regularized-Pessimistic-Value-Iteration

### <p align="center">[ICML 2025]</p>

<p align="center">
  <a href="https://tach1018.github.io/">Cheng Tang</a><sup>*</sup> ·
  <a href="">Zhishuai Liu</a><sup>†</sup> ·
  <a href="https://panxulab.github.io/">Pan Xu</a><sup>†</sup>
</p>
<p align="center">
<sup>*</sup> University of Illinois Urbana-Champaign .  
<sup>†</sup> Duke University
</p>

Code for the Paper "Robust Offline Reinforcement Learning with Linearly Structured f-Divergence Regularization", International Conference on Machine Learning (ICML) 2025.  All experiments are conducted on a machine with
an 11th Gen Intel(R) Core(TM) i5-11300H @ 3.10GHz
processor, featuring 8 logical CPUs, 4 physical cores, and
2 threads per core. 

## Experiments

### Simulated Linear MDP

In the Simulated Linear MDP task, the environment contains parameter and hyperparameter as follows:
- T1: Number of training episode
- H: Horizon
- beta: $\beta$ in the paper, controls the penalty
- gamma : $\gamma$ in the paper
- xi_norm, delta: Parameters in the environment
- T2: Number of test episodes


To run the experiments, use the following commands:
```bash
cd Simulated_Linear_MDP
python main.py --T1 100 --H 3 ...
```

### American Put Option
In the American Put Option task, the environment contains parameter and hyperparameter as follows:

- N: Sample size
- H: Horizon
- beta: $\beta$ in the paper, controls the penalty
- gamma : $\gamma$ in the paper
- p0 : Perturbation of the environment
- d: Dimention of the feature mapping


To run the experiments, use the following commands:
```bash
cd American_put_opt
python main.py --N 100 --H 20 ...
```

## Citation
```
@inproceedings{liu2024distributionally,
  title={Robust Offline Reinforcement Learning with Linearly Structured f-Divergence Regularization},
  author={Cheng Tang, Zhishuai Liu and Xu, Pan},
  booktitle={International Conference on Machine Learning},
  year={2025}
}
```