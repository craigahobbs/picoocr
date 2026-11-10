# Licensed under the MIT License
# https://github.com/craigahobbs/picoocr/blob/main/LICENSE

"""
The tests of picoocr_train.pages
"""

import json
import os
import random
import tempfile
import unittest
import unittest.mock

from picoocr_train import pages
from picoocr_train.fonts import load_font
from picoocr_train.pages import layout_column, render_page, wrap

from .helpers import run_module, test_font_path, test_fonts


# Long paragraphs (the pages draw 40 at a time)
PARAGRAPHS = [f'paragraph {index} ' + 'words in a line ' * 8 for index in range(45)]


# The pages' sizes patched small
def small_pages():
    return unittest.mock.patch.multiple(pages, PAGE_WIDTH=700, PAGE_HEIGHT=1400, BORDER=20)


class TestPages(unittest.TestCase):

    def test_wrap(self):
        font = load_font(test_font_path(), 20)
        lines = wrap('the quick brown fox jumps over the lazy dog', font, 100)
        self.assertGreater(len(lines), 2)
        self.assertTrue(all(font.getlength(line) <= 100 or ' ' not in line for line in lines))
        self.assertEqual(' '.join(lines), 'the quick brown fox jumps over the lazy dog')

    def test_layout_column(self):
        font = load_font(test_font_path(), 20)
        placed, texts, bottom = layout_column(['one two three', 'four five six seven eight nine ten'], font, 20, 10, 200, 0, 1000)
        self.assertEqual(len(texts), 2)
        self.assertEqual(placed[0], ('one two three', 10, 20))
        self.assertGreater(bottom, placed[-1][2])

        # The page's bottom - the paragraph that doesn't fit isn't placed
        placed, texts, _ = layout_column(['one', 'two ' * 40], font, 20, 10, 100, 0, 60)
        self.assertEqual(texts, ['one'])

    def test_render_page(self):
        with small_pages():
            for seed in range(3):
                rgba, text, details = render_page(random.Random(seed), PARAGRAPHS, test_fonts())
                self.assertEqual(rgba.shape[2], 4)
                self.assertGreater(len(text.split('\n')), 2)
                self.assertEqual(sorted(details), ['angle', 'columns', 'font', 'size'])

    def test_main(self):
        with tempfile.TemporaryDirectory() as temp_dir, small_pages(), \
             unittest.mock.patch.object(pages, 'load_paragraphs', return_value=PARAGRAPHS), \
             unittest.mock.patch.object(pages, 'load_fonts', return_value=[{'name': 'Test'}]), \
             unittest.mock.patch.object(pages, 'font_path', return_value=test_font_path()):
            self.assertEqual(pages.main([temp_dir, '--count', '1']), 0)
            with open(os.path.join(temp_dir, 'page0.json'), encoding='utf-8') as file:
                page = json.load(file)
            self.assertEqual(os.path.getsize(os.path.join(temp_dir, 'page0.rgba')), page['width'] * page['height'] * 4)
            self.assertTrue(os.path.isfile(os.path.join(temp_dir, 'page0.png')))

    def test_main_module(self):
        code, stdout, _ = run_module('picoocr_train.pages', ['--help'])
        self.assertEqual(code, 0)
        self.assertIn('usage: python -m picoocr_train.pages', stdout)
