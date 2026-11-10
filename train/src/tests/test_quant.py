# Licensed under the MIT License
# https://github.com/craigahobbs/picoocr/blob/main/LICENSE

"""
The tests of picoocr_train.quant
"""

import unittest

import numpy
import torch

from picoocr_train.quant import bit_limit, dequantize, gptq, layer_hessians, pack, quantize

from .helpers import tiny_recipe
from .test_model import trained_recognizer


# A quantized layer's output error - the residual's quadratic form r H r^T, summed over the rows
def output_error(rows, integers, scales, hessian):
    residual = rows - integers * scales[:, None]
    return numpy.trace(residual @ hessian @ residual.T)


class TestQuant(unittest.TestCase):

    def test_bit_limit(self):
        self.assertEqual([bit_limit(bits) for bits in (3, 5, 8)], [3, 15, 127])

    def test_quantize(self):
        weight = torch.tensor([[1.0, -0.5, 0.25], [0.0, 0.0, 0.0]])
        integers, scales = quantize(weight, 8)
        self.assertEqual(integers.tolist(), [[127, -64, 32], [0, 0, 0]])
        self.assertEqual(scales.dtype, numpy.float32)
        self.assertEqual(scales.tolist(), [numpy.float32(1 / 127), 1.0])
        integers, _ = quantize(weight, 3)
        self.assertEqual(integers.tolist(), [[3, -2, 1], [0, 0, 0]])

        # Dequantized - the weight's shape
        values = dequantize(integers, scales, (2, 1, 3))
        self.assertEqual(tuple(values.shape), (2, 1, 3))
        self.assertEqual(values.dtype, torch.float32)

    def test_gptq(self):
        # With an identity covariance, GPTQ is nearest rounding
        rows = numpy.array([[0.9, -0.4, 0.26]])
        scales = numpy.array([0.3])
        self.assertEqual(gptq(rows, scales, numpy.eye(3), 3).tolist(), [[3, -1, 1]])

        # Correlated inputs - GPTQ keeps the layer's output: its output error (r H r^T, r the weight's residual) is below
        # nearest rounding's, and it rounds some weights differently
        rng = numpy.random.default_rng(0)
        inputs = rng.normal(size=(400, 24)) @ rng.normal(size=(24, 24))
        hessian = inputs.T @ inputs / len(inputs)
        rows = rng.normal(size=(16, 24))
        scales = numpy.abs(rows).max(axis=1) / 15
        nearest = numpy.clip(numpy.round(rows / scales[:, None]), -15, 15)
        rounded = gptq(rows, scales, hessian, 15)
        self.assertLess(output_error(rows, rounded, scales, hessian), 0.7 * output_error(rows, nearest, scales, hessian))
        self.assertFalse(numpy.array_equal(rounded, nearest))

        # A dead input (no variance) - its weights zeroed; through quantize
        dead = numpy.diag([1.0, 1.0, 0.0])
        integers, _ = quantize(torch.tensor([[0.44, 0.44, 0.5]]), 4, dead)
        self.assertEqual(integers.tolist(), [[6, 6, 0]])

    def test_pack(self):
        # 3 bits: 1, -2, 3, -3 - unsigned 4, 1, 6, 0, little-endian bit by bit
        self.assertEqual(pack(numpy.array([1, -2, 3, -3]), 3), bytes([0b10001100, 0b00001]))
        self.assertEqual(pack(numpy.array([[0, -127], [127, 5]]), 8), bytes([127, 0, 254, 132]))

    def test_layer_hessians(self):
        model = trained_recognizer()
        layers, tensors = model.export_layers()
        batches = [(torch.rand(2, 1, 32, 40),)]
        hessians = layer_hessians(layers, tensors, batches)
        self.assertEqual(sorted(hessians), sorted(tensors))
        for name, (weight, _) in tensors.items():
            hessian = hessians[name]
            self.assertEqual(hessian.shape, (weight[0].numel(), weight[0].numel()))
            self.assertTrue(numpy.allclose(hessian, hessian.T))
        self.assertEqual(len(tiny_recipe()['alphabet']) + 1, tensors['out'][0].shape[0])
