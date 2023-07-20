# Differential Analysis of Triggers and Benign Features for Black-Box DNN Backdoor Detection
This is the official code for the paper "Differential Analysis of Triggers and Benign Features for Black-Box DNN Backdoor Detection" accepted by the IEEE Transactions on Information Forensics & Security.

## Description of the files

- config.py
  > Hyperparameters, the pool of ratios and noise variances in Alg. 1 of the paper.

- cuda.py
  > CPU or GPU setting

- models.py
  > Neural network architectures

- trainmnist.py
  > Training demo for backdoored networks, you can write your own backdoor training scripts.

- rob_sens_check.py
  > Our defense

## Implementation of the files

### Setup device (GPU or CPU)
Modify
> cuda.py

### Train a backdoored model
Run

```
python trainmnist.py
```

### Implement the defense

```
python rob_sens_check.py
```
## Modification of the code

### Different regions and noise

Modify 
> config.py

### Load pre-trained models

Modify the *load()* function in
> rob_sense_check.py

### Cite the Work

```
@ARTICLE{10187163,
  author={Fu, Hao and Krishnamurthy, Prashanth and Garg, Siddharth and Khorrami, Farshad},
  journal={IEEE Transactions on Information Forensics and Security}, 
  title={Differential Analysis of Triggers and Benign Features for Black-Box DNN Backdoor Detection}, 
  year={2023},
  volume={},
  number={},
  pages={1-1},
  doi={10.1109/TIFS.2023.3297056}}
```
