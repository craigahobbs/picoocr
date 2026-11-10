# Licensed under the MIT License
# https://github.com/craigahobbs/picoocr/blob/main/LICENSE

"""
The tests' shared helpers - a test font, a tiny recipe, and a module run as a script
"""

import contextlib
import functools
import io
import os
import runpy
import sys
import tempfile
import unittest.mock
import warnings

from PIL import ImageFont


# Pillow's embedded TrueType font (ImageFont.load_default's) written to a file - the tests' font, with no download
@functools.cache
def test_font_path():
    captured = []
    truetype = ImageFont.truetype

    def capture(font, *args, **kwargs):
        captured.append(font.getvalue())
        return truetype(io.BytesIO(captured[-1]), *args, **kwargs)

    with unittest.mock.patch.object(ImageFont, 'truetype', capture):
        ImageFont.load_default(size=10)
    path = os.path.join(tempfile.mkdtemp(), 'test-font.ttf')
    with open(path, 'wb') as file:
        file.write(captured[0])
    return path


# The test fonts [(font, path)] - the test font, as a train font and a validation font
def test_fonts():
    return [({'name': 'Test', 'split': 'train'}, test_font_path())]


# A tiny recipe - a short alphabet, a small network, short lines, and a few steps
def tiny_recipe():
    return {
        'alphabet': " abcdefghijklmnopqrstuvwxyz.,'\"-",
        'model': {'height': 32, 'conv': [[4, [2, 2]], [4, [2, 2]], [8, [2, 1]], [8, [2, 1]]], 'proj': 8, 'dilations': [1, 2]},
        'data': {
            'maxChars': 12, 'minSize': 14, 'maxSize': 20, 'kinds': {'book': 0.6, 'format': 0.2, 'random': 0.2},
            'validationSamples': 4, 'validationSeed': 7
        },
        'train': {'batch': 2, 'steps': 3, 'lr': 0.002, 'warmup': 2, 'weightDecay': 0.01, 'clip': 1.0, 'validateEvery': 2, 'seed': 1},
        'export': {'bits': 5, 'gptq': True, 'calibrationSamples': 4, 'calibrationSeed': 3}
    }


# The tests' book paragraphs
PARAGRAPHS = [
    'it was the best of times, it was the worst of times.',
    "she said, 'come here.' and he came -- slowly.",
    'a short one.'
]


# Run a module as a script (python -m) with arguments - (exit code, stdout, stderr)
def run_module(name, args):
    stdout = io.StringIO()
    stderr = io.StringIO()
    with contextlib.redirect_stdout(stdout), contextlib.redirect_stderr(stderr), \
         unittest.mock.patch.object(sys, 'argv', [name] + args), warnings.catch_warnings():
        # runpy warns that the module, imported by the tests, is run again as __main__ - as intended here
        warnings.filterwarnings('ignore', message=f"'{name}' found in sys.modules", category=RuntimeWarning)
        code = 0
        try:
            runpy.run_module(name, run_name='__main__')
        except SystemExit as error:
            code = error.code
    return code, stdout.getvalue(), stderr.getvalue()
