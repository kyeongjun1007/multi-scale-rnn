# Multi-Scale Recurrent Neural Networks
Pytorch implementation of Multi-Scale Recurrent Neural Networks (MSRNN).

## Getting Started
Installation:
'''python
$ pip3 install -r requirements.txt
'''

## Example
'''python
import msrnn
import torch

n_input = 9
n_hidden = 32
n_layers = 3
cell_type = 'RNN'

model = msrnn.MSRNN(n_input, n_hidden, n_layers, cell_type=cell_type)

# batch size : B
# length of timeseires : T
# input dimension : I
# hidden dimension : H

input = torch.randn(2, 30, n_input)
# input shape : (B,T,I)

output, hidden = model(input)
# output : list of outputs from each layer (length of output is n_layers)
# hidden : list of hiddens from each layer (length of hidden is n_layers)
'''
