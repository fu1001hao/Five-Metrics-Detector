# Differential-Analysis-of-Triggers-and-Benign-Features-for-Black-Box-DNN-Backdoor-Detection
This is the official implementation of the paper "Differential Analysis of Triggers and Benign Features for Black-Box DNN Backdoor Detection".

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

### Setup device
If you do not have cuda, go to 
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

Modify the *load()* in
> rob_sense_check.py
