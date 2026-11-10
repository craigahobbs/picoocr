# Licensed under the MIT License
# https://github.com/craigahobbs/picoocr/blob/main/LICENSE

"""
The tests of picoocr_train.model
"""

import unittest

import torch

from picoocr_train.model import ConvNorm, Recognizer, character_error_rate, edit_distance, greedy_decode, run_layers

from .helpers import tiny_recipe


# A recognizer with its batch norms' statistics set (not the identity), in inference mode
def trained_recognizer():
    torch.manual_seed(1)
    recipe = tiny_recipe()
    model = Recognizer(len(recipe['alphabet']) + 1, recipe['model'])
    for module in model.modules():
        if isinstance(module, (torch.nn.BatchNorm1d, torch.nn.BatchNorm2d)):
            module.running_mean.uniform_(-0.5, 0.5)
            module.running_var.uniform_(0.5, 2)
            module.weight.data.uniform_(0.5, 1.5)
            module.bias.data.uniform_(-0.5, 0.5)
    return model.eval()


class TestModel(unittest.TestCase):

    def test_recognizer(self):
        model = trained_recognizer()
        scores = model(torch.rand(2, 1, 32, 40))
        self.assertEqual(tuple(scores.shape), (2, len(tiny_recipe()['alphabet']) + 1, 10))

    def test_unpooled(self):
        # A last convolution without a pool - its rows flattened with its channels
        config = dict(tiny_recipe()['model'], conv=[[4, [2, 2]], [4, [2, 2]], [8, [2, 1]], [8, [2, 1]], [8, None]])
        model = Recognizer(5, config).eval()
        layers, tensors = model.export_layers()
        self.assertEqual([layer['op'] for layer in layers[7:10]], ['maxpool', 'conv2d', 'flatten'])
        inputs = torch.rand(1, 1, 32, 16)
        with torch.no_grad():
            self.assertTrue(torch.allclose(run_layers(layers, tensors, inputs), model(inputs), atol=1e-5))

    def test_export_layers(self):
        # The manifest's layers run as lib/model.js runs them give the model's scores
        model = trained_recognizer()
        layers, tensors = model.export_layers()
        self.assertEqual([layer['op'] for layer in layers], [
            'conv2d', 'maxpool', 'conv2d', 'maxpool', 'conv2d', 'maxpool', 'conv2d', 'maxpool', 'flatten', 'conv1d', 'conv1d',
            'conv1d', 'conv1d'
        ])
        self.assertEqual(layers[10], {'op': 'conv1d', 'name': 'seq0', 'dilation': 1, 'relu': True, 'residual': True})
        self.assertEqual(sorted(tensors), ['conv0', 'conv1', 'conv2', 'conv3', 'out', 'proj', 'seq0', 'seq1'])
        inputs = torch.rand(2, 1, 32, 40)
        with torch.no_grad():
            self.assertTrue(torch.allclose(run_layers(layers, tensors, inputs), model(inputs), atol=1e-5))

    def test_conv_norm(self):
        conv = ConvNorm(2, 3, kernel=3, dilation=2, dims=1).eval()
        weight, bias = conv.folded()
        self.assertEqual(tuple(weight.shape), (3, 2, 3))
        self.assertEqual(tuple(bias.shape), (3,))

    def test_greedy_decode(self):
        # Steps' best classes a, a, blank, a, b - and a step beyond the line's
        best = [1, 1, 0, 1, 2, 2]
        scores = torch.zeros(1, 3, len(best))
        for step, cls in enumerate(best):
            scores[0, cls, step] = 1
        self.assertEqual(greedy_decode(scores, [5], 'ab'), ['aab'])

    def test_errors(self):
        self.assertEqual(edit_distance('kitten', 'sitting'), 3)
        self.assertEqual(edit_distance('', 'ab'), 2)
        self.assertEqual(character_error_rate(['abc', 'x'], ['abd', 'xy']), 0.4)
        self.assertEqual(character_error_rate([''], ['']), 0)
