# MiniACSIncome
A benchmark for fairness verification of neural networks.
The benchmark is based on the ACSIncome dataset from the `folktables` package.
MiniACSIncome is based on USA census data from 2018.

## Examples
Load a `MiniACSIncome` dataset.
```python
from miniacsincome import MiniACSIncome

dataset = MiniACSIncome(num_variables=5)
dataset[0]
# (tensor([ 0.,  1.,  1.,  0.,  0.,  0.,  0.,  0.,  0.,  0.,  0.,  0.,  0.,  0.,
#           0.,  0.,  0.,  0.,  0.,  0.,  0.,  0.,  0.,  0.,  0.,  0.,  0.,  0.,
#           0.,  1.,  0.,  0.,  0.,  0., 60.,  0.,  0.,  1.,  0.,  0.]),
#  tensor(0)
```
The `MiniACSIncome` class is a `torch.utils.data.Dataset`.
It can be used with the usual PyTorch dataloaders.

Load a trained MiniACSIncome dataset from the benchmark.
```python
from miniacsincome import get_network

get_network(num_variables=5)
# Sequential(
#   (0): Linear(in_features=40, out_features=10, bias=True)
#   (1): ReLU()
#   (2): Linear(in_features=10, out_features=2, bias=True)
# )
```

The last part of MiniACSIncome are the input distributions (population models) 
for each MiniACSIncome dataset.
```python
from miniacsincome import get_population_model, get_network
import torch

pop_model, input_space, input_transform = get_population_model(5)
input_space.input_shape
# torch.Size([40])
input_space.input_bounds
# (tensor([0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0.,
#          0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 1., 0.,
#          0., 0., 0., 0.]),
#  tensor([ 1.,  1.,  1.,  1.,  1.,  1.,  1.,  1.,  1.,  1.,  1.,  1.,  1.,  1.,
#           1.,  1.,  1.,  1.,  1.,  1.,  1.,  1.,  1.,  1.,  1.,  1.,  1.,  1.,
#           1.,  1.,  1.,  1.,  1.,  1., 99.,  1.,  1.,  1.,  1.,  1.]))

data = pop_model.sample(100, seed=0).to(torch.float32)
data.shape
# torch.Size([100, 40])

net = get_network(num_variables=5)
net(input_transform(data)).argmax(dim=-1).float().mean()
# tensor(0.2900)
```

## Population Model
The population models are trained in `population_model.ipynb`.
Each population model is derived from the population model for MiniACSIncome-8.
The marginal distributions of this population model are:

![marginal distributions](resources/marginal-distributions.png)

The covariance of the population model matches the covariance of the original dataset reasonably well.

![covariance matrix comparison](resources/covariance.png)

## Citation
If you use this dataset in your research, please cite:
```
@inproceedings{probspec,
  author       = {David Boetius and Stefan Leue and Tobias Sutter},
  title        = {Solving Probabilistic Verification Problems of Neural Networks using Branch and Bound},
  booktitle    = {{ICML}},
  series       = {Proceedings of Machine Learning Research},
  volume       = {267},
  publisher    = {{PMLR}},
  year         = {2025},
}
```

