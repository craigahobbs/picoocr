# Licensed under the MIT License
# https://github.com/craigahobbs/picoocr/blob/main/LICENSE

"""
The tests of picoocr_train.data
"""

import itertools
import unittest
import unittest.mock

import numpy

from picoocr_train.data import POOL, LineBatches, collate, render_samples, sorted_batches

from .helpers import PARAGRAPHS, test_fonts, tiny_recipe


class TestData(unittest.TestCase):

    def test_collate(self):
        samples = [(numpy.ones((32, 8), dtype=numpy.float32), 'ab'), (numpy.ones((32, 12), dtype=numpy.float32), 'c')]
        inputs, steps, targets, target_lengths, texts = collate(samples, 'abc')
        self.assertEqual(tuple(inputs.shape), (2, 1, 32, 12))
        self.assertEqual(float(inputs[0, 0, 0, 8:].sum()), 0)
        self.assertEqual(steps.tolist(), [2, 3])
        self.assertEqual(targets.tolist(), [1, 2, 3])
        self.assertEqual(target_lengths.tolist(), [2, 1])
        self.assertEqual(texts, ['ab', 'c'])

    def test_render_samples(self):
        recipe = tiny_recipe()
        first = render_samples(recipe, test_fonts(), PARAGRAPHS, 5, 3)
        second = render_samples(recipe, test_fonts(), PARAGRAPHS, 5, 3)
        self.assertEqual([text for _, text in first], [text for _, text in second])

        # Sorted into batches
        batches = sorted_batches(first, recipe['alphabet'], 2)
        self.assertEqual([len(batch[4]) for batch in batches], [2, 1])
        self.assertLessEqual(batches[0][0].shape[3], batches[1][0].shape[3])

    def test_line_batches(self):
        recipe = tiny_recipe()
        dataset = LineBatches(recipe, test_fonts(), PARAGRAPHS, 3)

        # The main process's batches, and a worker's - different
        first = [batch[4] for batch in itertools.islice(iter(dataset), POOL + 1)]
        self.assertTrue(all(len(texts) == recipe['train']['batch'] for texts in first))
        worker = unittest.mock.Mock(id=1)
        with unittest.mock.patch('torch.utils.data.get_worker_info', return_value=worker):
            other = [batch[4] for batch in itertools.islice(iter(dataset), 1)]
        self.assertNotIn(other[0], first)
