# Licensed under the MIT License
# https://github.com/craigahobbs/picoocr/blob/main/LICENSE

"""
The tests of picoocr_train.render
"""

import random
import unittest
import unittest.mock

import numpy

from picoocr_train import render
from picoocr_train.fonts import load_font
from picoocr_train.image import LINE_HEIGHT, LINE_MARGIN
from picoocr_train.render import ctc_steps, photograph, render_coverage, render_sample
from picoocr_train.texts import TextSampler

from .helpers import PARAGRAPHS, test_font_path, test_fonts, tiny_recipe


# A random.Random whose random() returns values in turn (then 0.5), its other draws its own
class Draws(random.Random):

    def __init__(self, values):
        super().__init__(1)
        self.values = list(values)

    def random(self):
        return self.values.pop(0) if self.values else 0.5


class TestRender(unittest.TestCase):

    def test_ctc_steps(self):
        self.assertEqual(ctc_steps('abc'), 3)
        self.assertEqual(ctc_steps('aab'), 4)
        self.assertEqual(ctc_steps('...'), 5)

    def test_render_coverage(self):
        font = load_font(test_font_path(), 20)

        # Ligatures off, underlined, both context lines, tilted
        coverage, box = render_coverage(Draws([0.9, 0.01, 0.1, 0.1, 0.1]), 'hello', font, 20, lambda: 'context line')
        self.assertEqual(coverage.dtype, numpy.float64)
        self.assertTrue(0 <= coverage.min() and coverage.max() <= 1)
        left, top, width, height = box
        self.assertTrue(left > 0 and top > 0 and width > 20 and height > 5)

        # Plain - no context, not tilted
        _, plain = render_coverage(Draws([0.1, 0.9, 0.9, 0.9, 0.9]), 'hello', font, 20, lambda: 'x')
        self.assertTrue(plain[2] > 20)

        # No ink
        self.assertIsNone(render_coverage(Draws([0.1, 0.9, 0.9, 0.9, 0.9]), ' ', font, 20, lambda: 'x'))

    def test_photograph(self):
        coverage = numpy.zeros((20, 30))
        coverage[5:15, 5:25] = 1

        # Blurred, compressed
        gray = photograph(Draws([0.1, 0.1]), coverage, 20)
        self.assertEqual(gray.shape, (20, 30))
        self.assertEqual(gray.dtype, numpy.uint8)
        self.assertLess(gray[10, 15], gray[1, 1])

        # Sharp, uncompressed
        self.assertEqual(photograph(Draws([0.9, 0.9]), coverage, 20).shape, (20, 30))

    def test_render_sample(self):
        recipe = tiny_recipe()
        rng = random.Random(2)
        sampler = TextSampler(rng, PARAGRAPHS, recipe['alphabet'], recipe['data'])
        for _ in range(5):
            line, text = render_sample(rng, sampler, test_fonts(), recipe['data'])
            self.assertEqual(line.shape[0], LINE_HEIGHT)
            self.assertGreaterEqual((line.shape[1] - 2 * LINE_MARGIN) // 4, ctc_steps(text))
            self.assertTrue(0 <= line.min() and line.max() <= 1)

    def test_render_sample_retries(self):
        # Blank, then a box too small, then too short for CTC, then a line
        recipe = tiny_recipe()
        sampler = unittest.mock.Mock()
        sampler.sample.side_effect = ['a', 'b', 'c', 'd']
        rendered = [None, (numpy.zeros((40, 60)), (10, 10, 1, 1)), (numpy.zeros((40, 60)), (10, 10, 30, 20)),
                    (numpy.zeros((40, 200)), (10, 10, 150, 20))]
        lines = [numpy.zeros((32, 100)), numpy.zeros((32, 200))]
        with unittest.mock.patch.object(render, 'render_coverage', side_effect=rendered), \
             unittest.mock.patch.object(render, 'line_input', side_effect=lines), \
             unittest.mock.patch.object(render, 'BOX_JITTER', 0):
            sampler.sample.side_effect = ['a', 'b', 'ccccccccccccccc', 'd']
            line, text = render_sample(random.Random(1), sampler, test_fonts(), recipe['data'])
        self.assertEqual(text, 'd')
        self.assertEqual(line.shape, (32, 200))
