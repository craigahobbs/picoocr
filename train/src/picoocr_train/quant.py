# Licensed under the MIT License
# https://github.com/craigahobbs/picoocr/blob/main/LICENSE

"""
Weight quantization - the one definition shared by the export (export.py) and the package's loader (lib/model.js):

- A convolution's weights are quantized per output channel (its row - the rest flattened): symmetric, bits wide,
  scale = max |w| / limit (limit = 2 ** (bits - 1) - 1; computed in float64, stored float32), each weight an integer
  q within +-limit, w' = q * scale. A row of zeros has scale 1.
- q is the nearest integer to w / scale, or - given the layer's input covariance (H = E[x x^T] over calibration lines'
  patches - layer_hessians) - GPTQ's: a row is rounded a column at a time, each column's rounding error compensated in
  the columns not yet rounded through H's inverse, so the layer's output (w x), not each weight, is kept. The
  evaluation (DESIGN.md) found nearest rounding lossless at 6 to 8 bits and lossy below; GPTQ lossless at 5.
- The integers are packed as unsigned values (q + limit) of bits bits, little-endian bit by bit (pack - lib/model.js
  unpacks them).
"""

import numpy
import torch
from torch.nn import functional

from .model import run_layers


# GPTQ's damping - this share of H's mean diagonal added to its diagonal
DAMP = 0.01


# The largest integer magnitude of a width
def bit_limit(bits):
    return (1 << (bits - 1)) - 1


# GPTQ's integers of rows [rows, columns] at their scales [rows] (float64), given the input covariance hessian
# [columns, columns] (float64) - within +-limit (int64)
def gptq(rows, scales, hessian, limit):
    hessian = hessian.copy()
    dead = numpy.diag(hessian) == 0
    hessian[dead, dead] = 1.0
    rows = rows.copy()
    rows[:, dead] = 0.0
    hessian += DAMP * numpy.mean(numpy.diag(hessian)) * numpy.eye(len(hessian))
    upper = numpy.linalg.cholesky(numpy.linalg.inv(hessian)).T
    integers = numpy.zeros(rows.shape, dtype=numpy.int64)
    for column in range(rows.shape[1]):
        values = rows[:, column]
        integers[:, column] = numpy.clip(numpy.round(values / scales), -limit, limit)
        error = (values - integers[:, column] * scales) / upper[column, column]
        rows[:, column + 1:] -= numpy.outer(error, upper[column, column + 1:])
    return integers


# Quantize a weight (a torch tensor, [rows, ...]) - (integers int64 [rows, row length], float32 scales [rows]); the
# nearest integers, or GPTQ's given the layer's input covariance [row length, row length]
def quantize(weight, bits, hessian=None):
    limit = bit_limit(bits)
    rows = weight.detach().reshape(weight.shape[0], -1).double().numpy()
    scales = numpy.abs(rows).max(axis=1) / limit
    scales[scales == 0] = 1.0
    if hessian is None:
        integers = numpy.clip(numpy.round(rows / scales[:, None]), -limit, limit).astype(numpy.int64)
    else:
        integers = gptq(rows, scales, hessian, limit)
    return integers, scales.astype(numpy.float32)


# The weight of integers and their scales (float32, the weight's shape) - as the package dequantizes it
def dequantize(integers, scales, shape):
    return torch.from_numpy((integers * scales[:, None].astype(numpy.float64)).astype(numpy.float32)).reshape(shape)


# Integers packed - unsigned (q + limit), bits bits each, little-endian bit by bit - as bytes
def pack(integers, bits):
    unsigned = (integers.reshape(-1) + bit_limit(bits)).astype(numpy.uint64)
    bit_values = (unsigned[:, None] >> numpy.arange(bits, dtype=numpy.uint64)) & numpy.uint64(1)
    return numpy.packbits(bit_values.reshape(-1).astype(numpy.uint8), bitorder='little').tobytes()


# A convolution layer's input patches [positions, row length] - each output position's inputs, as its weight's rows see
# them
def layer_patches(layer, x, weight):
    kernel = weight.shape[2]
    if layer['op'] == 'conv2d':
        unfolded = functional.unfold(x, kernel, padding=(kernel - 1) // 2)
    else:
        dilation = layer.get('dilation', 1)
        unfolded = functional.unfold(x.unsqueeze(2), (1, kernel), dilation=(1, dilation), padding=(0, dilation * (kernel - 1) // 2))
    return unfolded.transpose(1, 2).reshape(-1, weight[0].numel())


# Each convolution layer's input covariance (float64 numpy [row length, row length]) over calibration batches - the
# float network's inputs (layers - the manifest's; tensors - {name: (weight, bias)})
def layer_hessians(layers, tensors, batches):
    sums = {}
    counts = {}
    with torch.no_grad():
        for inputs, *_ in batches:
            x = inputs
            for layer in layers:
                if layer['op'] in ('conv2d', 'conv1d'):
                    patches = layer_patches(layer, x, tensors[layer['name']][0]).double()
                    sums[layer['name']] = sums.get(layer['name'], 0) + patches.T @ patches
                    counts[layer['name']] = counts.get(layer['name'], 0) + patches.shape[0]
                x = run_layers([layer], tensors, x)
    return {name: (total / counts[name]).numpy() for name, total in sums.items()}
