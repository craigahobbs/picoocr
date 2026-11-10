# Licensed under the MIT License
# https://github.com/craigahobbs/picoocr/blob/main/LICENSE

"""
The tests of picoocr_train.train
"""

import io
import json
import os
import tempfile
import unittest
import unittest.mock

import torch

from picoocr_train import train as train_module
from picoocr_train.train import learning_rate, load_paragraphs, split_fonts, train

from .helpers import PARAGRAPHS, run_module, test_fonts, tiny_recipe


# The training's data patched to the tests' - the test font for both splits, and the tests' paragraphs
def patched_data():
    return unittest.mock.patch.multiple(train_module, split_fonts=lambda split: test_fonts(), load_paragraphs=lambda alphabet: PARAGRAPHS)


class TestTrain(unittest.TestCase):

    def test_learning_rate(self):
        config = {'lr': 1.0, 'warmup': 10, 'steps': 110}
        self.assertEqual(learning_rate(0, config), 0.1)
        self.assertEqual(learning_rate(9, config), 1.0)
        self.assertEqual(learning_rate(10, config), 1.0)
        self.assertAlmostEqual(learning_rate(60, config), 0.505)
        self.assertAlmostEqual(learning_rate(110, config), 0.01)
        self.assertAlmostEqual(learning_rate(200, config), 0.01)

    def test_split_fonts(self):
        with unittest.mock.patch.object(train_module, 'load_fonts', return_value=[{'name': 'A'}]), \
             unittest.mock.patch.object(train_module, 'font_path', return_value='/a.ttf'):
            self.assertEqual(split_fonts('train'), [({'name': 'A'}, '/a.ttf')])

    def test_load_paragraphs(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            path = os.path.join(temp_dir, 'book.txt')
            with open(path, 'w', encoding='utf-8-sig') as file:
                file.write('one two\n\nthree')
            with unittest.mock.patch.object(train_module, 'load_books', return_value=[{'name': 'b'}]), \
                 unittest.mock.patch.object(train_module, 'book_path', return_value=path):
                self.assertEqual(load_paragraphs('onetwhr '), ['one two', 'three'])

    def test_train(self):
        recipe = tiny_recipe()
        with tempfile.TemporaryDirectory() as temp_dir, patched_data(), unittest.mock.patch.object(train_module, 'LOG_EVERY', 1), \
             unittest.mock.patch('sys.stderr', io.StringIO()) as stderr:
            # Three steps, validated at 2 and 3 (the last) - the second validation worse (best.pt not written again)
            evaluate = train_module.evaluate
            errors = iter([0.5, 0.5, 0.9, 0.9])
            with unittest.mock.patch.object(train_module, 'evaluate', side_effect=lambda *args: (next(errors), evaluate(*args)[1])):
                best = train(temp_dir, recipe, workers=0)
            self.assertEqual(best, 0.5)
            with open(os.path.join(temp_dir, 'log.jsonl'), encoding='utf-8') as file:
                entries = [json.loads(line) for line in file]
            self.assertEqual([entry['step'] for entry in entries], [1, 2, 2, 3, 3])
            self.assertEqual(entries[2], {'step': 2, 'seen': 0.5, 'unseen': 0.5})
            self.assertEqual(torch.load(os.path.join(temp_dir, 'best.pt'), weights_only=False)['step'], 2)
            self.assertEqual(torch.load(os.path.join(temp_dir, 'last.pt'), weights_only=False)['step'], 3)
            self.assertIn('training', stderr.getvalue())

            # Continued from its checkpoint, to a longer schedule, stopped early
            best = train(temp_dir, recipe, steps=6, until=4, workers=0, init=os.path.join(temp_dir, 'last.pt'))
            self.assertEqual(torch.load(os.path.join(temp_dir, 'last.pt'), weights_only=False)['step'], 4)
            self.assertLessEqual(best, 0.5)

    def test_main(self):
        with tempfile.TemporaryDirectory() as temp_dir, patched_data(), unittest.mock.patch('sys.stderr', io.StringIO()):
            recipe_path = os.path.join(temp_dir, 'recipe.json')
            with open(recipe_path, 'w', encoding='utf-8') as file:
                json.dump(tiny_recipe(), file)
            threads = torch.get_num_threads()
            try:
                args = [temp_dir, '--recipe', recipe_path, '--steps', '1', '--workers', '0', '--threads', '1']
                self.assertEqual(train_module.main(args), 0)
            finally:
                torch.set_num_threads(threads)
            self.assertTrue(os.path.isfile(os.path.join(temp_dir, 'best.pt')))

            # PyTorch's default threads
            self.assertEqual(train_module.main([temp_dir, '--recipe', recipe_path, '--steps', '1', '--workers', '0']), 0)

    def test_main_module(self):
        code, stdout, _ = run_module('picoocr_train.train', ['--help'])
        self.assertEqual(code, 0)
        self.assertIn('usage: python -m picoocr_train.train', stdout)
