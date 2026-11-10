# Licensed under the MIT License
# https://github.com/craigahobbs/picoocr/blob/main/LICENSE

"""
The tests of picoocr_train.fonts
"""

import unittest
import unittest.mock

from picoocr_train import fonts
from picoocr_train.fonts import font_path, load_font, load_fonts, missing_chars

from .helpers import run_module, test_font_path


class TestFonts(unittest.TestCase):

    def test_load_fonts(self):
        train = load_fonts('train')
        validation = load_fonts('validation')
        self.assertGreater(len(train), 40)
        self.assertEqual(len(validation), 4)
        self.assertFalse({font['name'] for font in train} & {font['name'] for font in validation})

    def test_font_path(self):
        font = {'url': 'https://example.com/fonts/Some%5Bwght%5D.ttf', 'sha256': 'x'}
        with unittest.mock.patch.object(fonts, 'pinned_path', side_effect=lambda pin, path: path):
            self.assertEqual(font_path(font, '/downloads'), '/downloads/fonts/Some[wght].ttf')

    def test_load_font(self):
        font = load_font(test_font_path(), 20)
        self.assertEqual(font.size, 20)
        self.assertIs(load_font(test_font_path(), 20), font)

        # A variable font's named instance
        with unittest.mock.patch('PIL.ImageFont.FreeTypeFont.set_variation_by_name') as set_variation:
            load_font(test_font_path(), 21, 'Bold')
        set_variation.assert_called_once_with('Bold')

    def test_missing_chars(self):
        self.assertEqual(missing_chars(load_font(test_font_path(), 20), 'a b\u4e2d'), ['\u4e2d'])

    def test_main(self):
        # Every font has every glyph
        with unittest.mock.patch.object(fonts, 'font_path', return_value=test_font_path()), \
             unittest.mock.patch.object(fonts, 'read_json', side_effect=[
                 {'alphabet': 'ab'}, {'fonts': [{'name': 'A', 'split': 'train'}]}, {'fonts': []}
             ]), \
             unittest.mock.patch('sys.stdout') as stdout:
            self.assertEqual(fonts.main(['check']), 0)
        self.assertEqual(''.join(call[0][0] for call in stdout.write.call_args_list), 'every font has every glyph\n')

        # A font is missing a glyph
        with unittest.mock.patch.object(fonts, 'font_path', return_value=test_font_path()), \
             unittest.mock.patch.object(fonts, 'read_json', side_effect=[
                 {'alphabet': 'a\u4e2d'}, {'fonts': [{'name': 'A', 'split': 'train'}]}, {'fonts': []}
             ]), \
             unittest.mock.patch('sys.stdout') as stdout:
            self.assertEqual(fonts.main(['check']), 1)
        self.assertIn("A: missing '\u4e2d'", ''.join(call[0][0] for call in stdout.write.call_args_list))

    def test_main_module(self):
        code, stdout, _ = run_module('picoocr_train.fonts', ['--help'])
        self.assertEqual(code, 0)
        self.assertIn('usage: python -m picoocr_train.fonts', stdout)
