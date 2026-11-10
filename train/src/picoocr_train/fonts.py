# Licensed under the MIT License
# https://github.com/craigahobbs/picoocr/blob/main/LICENSE

"""
The pinned fonts (train/data/fonts.json) - each font's file fetched (downloads.py) into downloads/fonts/, and loaded at
a pixel size (a variable font at its named instance). The fonts' split: train fonts render the training lines;
validation fonts are never trained on.

    python -m picoocr_train.fonts check     # fetch the fonts and check each has a glyph for every alphabet character
"""

import argparse
import functools
import os
import sys
import urllib.parse

from PIL import ImageFont

from . import DATA_DIR, DOWNLOADS_DIR, RECIPE_PATH, read_json
from .downloads import pinned_path


# The pinned fonts
FONTS_PATH = os.path.join(DATA_DIR, 'fonts.json')


# The pinned fonts of a split ('train' or 'validation')
def load_fonts(split, path=FONTS_PATH):
    return [font for font in read_json(path)['fonts'] if font['split'] == split]


# The local path of a font's file - fetched if missing
def font_path(font, downloads_dir=DOWNLOADS_DIR):
    name = os.path.basename(urllib.parse.unquote(urllib.parse.urlparse(font['url']).path))
    return pinned_path(font, os.path.join(downloads_dir, 'fonts', name))


# A font loaded at a pixel size (cached - a worker renders many lines in each)
@functools.lru_cache(maxsize=1024)
def load_font(path, size, variation=None):
    font = ImageFont.truetype(path, size, layout_engine=ImageFont.Layout.RAQM)
    if variation is not None:
        font.set_variation_by_name(variation)
    return font


# A rendered character's mask - its size and bytes
def _mask(font, char):
    mask = font.getmask(char)
    return mask.size, bytes(bytearray(mask))


# The characters of an alphabet a font has no glyph for (rendered as the missing glyph - a private-use character's)
def missing_chars(font, alphabet):
    missing = _mask(font, '')
    return [char for char in alphabet if char != ' ' and _mask(font, char) == missing]


def main(argv=None):
    parser = argparse.ArgumentParser(prog='python -m picoocr_train.fonts', description=__doc__.strip().split('\n\n', maxsplit=1)[0])
    parser.add_argument('command', choices=['check'])
    parser.parse_args(argv)
    alphabet = read_json(RECIPE_PATH)['alphabet']
    failed = False
    for split in ('train', 'validation'):
        for font in load_fonts(split):
            missing = missing_chars(load_font(font_path(font), 40, font.get('variation')), alphabet)
            if missing:
                failed = True
                print(f'{font["name"]}: missing {"".join(missing)!r}')
    print('missing glyphs' if failed else 'every font has every glyph')
    return 1 if failed else 0


if __name__ == '__main__':
    sys.exit(main())
