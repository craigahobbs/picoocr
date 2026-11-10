# Licensed under the MIT License
# https://github.com/craigahobbs/picoocr/blob/main/LICENSE

"""
The line recognizer - a convolutional network read by CTC (the recipe's model section):

- conv - 2-D convolutions (3x3, batch norm, ReLU), each [channels, pool] - the pool (max) [height, width] or null. The
  pools take the 32-pixel line to the last convolution's rows and its width to a step per 4 columns.
- flatten - the last convolution's channels and rows, a feature vector per step
- proj - a 1x1 convolution (batch norm, ReLU) to the sequence width
- dilations - residual 1-D convolutions along the steps (kernel 3, batch norm, ReLU, added to their input), one per
  dilation - each step sees its neighbors' context
- out - a 1x1 convolution to the classes: CTC's blank, then the alphabet

The export (export.py) folds each batch norm into its convolution and lists the network as the manifest's layers, which
lib/model.js runs - and run_layers below runs the same way in PyTorch (the export's quantized check and parity data).
"""

import torch
from torch import nn
from torch.nn import functional


# A convolution and its batch norm - 2-D (kernel 3) or 1-D (a kernel and a dilation)
class ConvNorm(nn.Module):

    def __init__(self, in_channels, out_channels, kernel=3, dilation=1, dims=2):
        super().__init__()
        conv = nn.Conv2d if dims == 2 else nn.Conv1d
        norm = nn.BatchNorm2d if dims == 2 else nn.BatchNorm1d
        self.conv = conv(in_channels, out_channels, kernel, padding=dilation * (kernel - 1) // 2, dilation=dilation, bias=False)
        self.norm = norm(out_channels)

    def forward(self, x):
        return self.norm(self.conv(x))

    # The convolution's weight and bias with the batch norm folded in (inference)
    def folded(self):
        norm = self.norm
        factor = norm.weight / torch.sqrt(norm.running_var + norm.eps)
        weight = self.conv.weight * factor.reshape(-1, *([1] * (self.conv.weight.dim() - 1)))
        return weight.detach(), (norm.bias - norm.running_mean * factor).detach()


# The recognizer network
class Recognizer(nn.Module):

    # classes - CTC's blank and the alphabet's characters; config - the recipe's model section
    def __init__(self, classes, config):
        super().__init__()
        self.config = config
        self.convs = nn.ModuleList()
        channels = 1
        rows = config['height']
        for out_channels, pool in config['conv']:
            self.convs.append(ConvNorm(channels, out_channels))
            channels = out_channels
            rows //= pool[0] if pool else 1
        self.proj = ConvNorm(channels * rows, config['proj'], kernel=1, dims=1)
        self.seq = nn.ModuleList(ConvNorm(config['proj'], config['proj'], dilation=dilation, dims=1) for dilation in config['dilations'])
        self.out = nn.Conv1d(config['proj'], classes, 1)

    # Class scores [batch, classes, steps] of line inputs [batch, 1, height, width]
    def forward(self, x):
        for conv, (_, pool) in zip(self.convs, self.config['conv']):
            x = functional.relu(conv(x))
            if pool:
                x = functional.max_pool2d(x, pool)
        batch, channels, rows, steps = x.shape
        x = functional.relu(self.proj(x.reshape(batch, channels * rows, steps)))
        for seq in self.seq:
            x = x + functional.relu(seq(x))
        return self.out(x)

    # The network as the manifest's layers (lib/model.js), and its folded convolutions {name: (weight, bias)}
    def export_layers(self):
        layers = []
        tensors = {}
        for index, (conv, (_, pool)) in enumerate(zip(self.convs, self.config['conv'])):
            name = f'conv{index}'
            layers.append({'op': 'conv2d', 'name': name, 'relu': True})
            tensors[name] = conv.folded()
            if pool:
                layers.append({'op': 'maxpool', 'size': list(pool)})
        layers.append({'op': 'flatten'})
        layers.append({'op': 'conv1d', 'name': 'proj', 'relu': True})
        tensors['proj'] = self.proj.folded()
        for index, (seq, dilation) in enumerate(zip(self.seq, self.config['dilations'])):
            name = f'seq{index}'
            layers.append({'op': 'conv1d', 'name': name, 'dilation': dilation, 'relu': True, 'residual': True})
            tensors[name] = seq.folded()
        layers.append({'op': 'conv1d', 'name': 'out'})
        tensors['out'] = (self.out.weight.detach(), self.out.bias.detach())
        return layers, tensors


# Run the manifest's layers (as lib/model.js does) on line inputs [batch, 1, height, width] - class scores [batch,
# classes, steps]. tensors - {name: (weight, bias)}
def run_layers(layers, tensors, x):
    for layer in layers:
        if layer['op'] == 'maxpool':
            x = functional.max_pool2d(x, layer['size'])
        elif layer['op'] == 'flatten':
            x = x.reshape(x.shape[0], x.shape[1] * x.shape[2], x.shape[3])
        else:
            weight, bias = tensors[layer['name']]
            if layer['op'] == 'conv2d':
                y = functional.conv2d(x, weight, bias, padding=(weight.shape[2] - 1) // 2)
            else:
                dilation = layer.get('dilation', 1)
                y = functional.conv1d(x, weight, bias, padding=dilation * (weight.shape[2] - 1) // 2, dilation=dilation)
            if layer.get('relu'):
                y = functional.relu(y)
            x = x + y if layer.get('residual') else y
    return x


# The texts of class scores [batch, classes, steps] by CTC's greedy rule, for steps (each line's) and an alphabet
def greedy_decode(scores, steps, alphabet):
    best = scores.argmax(dim=1).tolist()
    texts = []
    for classes, count in zip(best, steps):
        text = []
        previous = 0
        for cls in classes[:count]:
            if cls not in (0, previous):
                text.append(alphabet[cls - 1])
            previous = cls
        texts.append(''.join(text))
    return texts


# The edit distance between two strings
def edit_distance(first, second):
    previous = list(range(len(second) + 1))
    for index, char in enumerate(first, 1):
        current = [index]
        for jndex, other in enumerate(second, 1):
            current.append(min(previous[jndex] + 1, current[jndex - 1] + 1, previous[jndex - 1] + (char != other)))
        previous = current
    return previous[-1]


# The character error rate of texts against their references
def character_error_rate(texts, references):
    errors = sum(edit_distance(text, reference) for text, reference in zip(texts, references))
    return errors / max(1, sum(len(reference) for reference in references))
