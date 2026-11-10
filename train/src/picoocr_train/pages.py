# Licensed under the MIT License
# https://github.com/craigahobbs/picoocr/blob/main/LICENSE

"""
Test pages - whole synthetic page photos and their text, for the package's end-to-end check (tools/pagecheck.js - the
layout, the line inputs, and the recognizer together, as the package reads a page):

    python -m picoocr_train.pages build/pages --count 8

Each page is a heading and paragraphs of book prose (one or two columns, word-wrapped, a validation font - never
trained on), tilted a few degrees on a darker table, and photographed (render.py's lighting, blur, noise, and JPEG).
Each page is written as <name>.json ({width, height, text} - its text's paragraphs separated by blank lines, its lines
by line breaks) and <name>.rgba (the image's bytes), and <name>.png (to look at).
"""

import argparse
import json
import os
import random
import sys

import numpy
from PIL import Image, ImageDraw

from . import RECIPE_PATH, read_json
from .fonts import font_path, load_font, load_fonts
from .render import photograph
from .train import load_paragraphs


# The page's size (pixels - a letter page about 200 pixels per inch) and the photo's border
PAGE_WIDTH = 1700
PAGE_HEIGHT = 2200
BORDER = 150


# Word-wrap a paragraph to a width in a font - its lines
def wrap(text, font, width):
    lines = []
    line = ''
    for word in text.split(' '):
        candidate = f'{line} {word}' if line else word
        if line and font.getlength(candidate) > width:
            lines.append(line)
            line = word
        else:
            line = candidate
    lines.append(line)
    return lines


# Lay out a column's paragraphs from a top - (the column's lines [(text, x, baseline)], its paragraphs' texts, the
# bottom), stopping at the page's bottom
def layout_column(paragraphs, font, size, left, width, top, bottom):
    pitch = int(size * 1.35)
    placed = []
    texts = []
    baseline = top + size
    for paragraph in paragraphs:
        lines = wrap(paragraph, font, width)
        if baseline + pitch * (len(lines) - 1) > bottom:
            break
        for line in lines:
            placed.append((line, left, baseline))
            baseline += pitch
        texts.append('\n'.join(lines))
        baseline += pitch // 2
    return placed, texts, baseline


# A test page - (the photo, an RGBA uint8 array [height, width, 4], its text)
def render_page(rng, paragraphs, fonts):
    font_entry, path = rng.choice(fonts)
    size = rng.randint(20, 40)
    font = load_font(path, size, font_entry.get('variation'))
    heading_font = load_font(path, int(size * 1.6), font_entry.get('variation'))
    page = Image.new('L', (PAGE_WIDTH, PAGE_HEIGHT), 0)
    draw = ImageDraw.Draw(page)
    margin = 150

    # The heading
    heading = ' '.join(rng.choice(paragraphs).split(' ')[:rng.randint(2, 5)]).strip(' ,;:')
    draw.text((margin, margin + int(size * 1.6)), heading, fill=255, font=heading_font, anchor='ls')
    top = margin + int(size * 3)

    # The columns
    columns = rng.choice([1, 1, 2])
    gutter = 60
    column_width = (PAGE_WIDTH - 2 * margin - (columns - 1) * gutter) // columns
    texts = [heading]
    candidates = [paragraph for paragraph in paragraphs if len(paragraph) > 80]
    start = rng.randrange(len(candidates) - 40)
    remaining = candidates[start:start + 40]
    for column in range(columns):
        placed, column_texts, _ = layout_column(
            remaining, font, size, margin + column * (column_width + gutter), column_width, top, PAGE_HEIGHT - margin
        )
        remaining = remaining[len(column_texts):]
        for line, x, baseline in placed:
            draw.text((x, baseline), line, fill=255, font=font, anchor='ls')
        texts.extend(column_texts)

    # Tilted on the table, and photographed
    angle = rng.uniform(-4, 4)
    coverage = page.rotate(angle, resample=Image.Resampling.BILINEAR, expand=True, fillcolor=0)
    photo = Image.new('L', (coverage.width + 2 * BORDER, coverage.height + 2 * BORDER), 0)
    photo.paste(coverage, (BORDER, BORDER))
    paper = Image.new('L', coverage.size, 0)
    paper.paste(Image.new('L', page.size, 255).rotate(angle, expand=True, fillcolor=0))
    paper_mask = numpy.zeros((photo.height, photo.width), dtype=numpy.float64)
    paper_mask[BORDER:BORDER + coverage.height, BORDER:BORDER + coverage.width] = numpy.asarray(paper) / 255
    gray = photograph(rng, numpy.asarray(photo, dtype=numpy.float64) / 255, size).astype(numpy.float64)
    table = rng.uniform(30, 90)
    gray = gray * paper_mask + table * (1 - paper_mask)
    rgba = numpy.zeros((photo.height, photo.width, 4), dtype=numpy.uint8)
    rgba[:, :, 0] = rgba[:, :, 1] = rgba[:, :, 2] = numpy.clip(gray, 0, 255).astype(numpy.uint8)
    rgba[:, :, 3] = 255
    return rgba, '\n\n'.join(texts), {'font': font_entry['name'], 'size': size, 'columns': columns, 'angle': angle}


def main(argv=None):
    parser = argparse.ArgumentParser(prog='python -m picoocr_train.pages', description=__doc__.strip().split('\n\n', maxsplit=1)[0])
    parser.add_argument('output_dir', help='the directory to write the pages to')
    parser.add_argument('--count', type=int, default=8, help='the pages (default 8)')
    parser.add_argument('--seed', type=int, default=1, help='the random seed (default 1)')
    args = parser.parse_args(argv)
    recipe = read_json(RECIPE_PATH)
    rng = random.Random(args.seed)
    paragraphs = load_paragraphs(recipe['alphabet'])
    fonts = [(font, font_path(font)) for font in load_fonts('validation')]
    os.makedirs(args.output_dir, exist_ok=True)
    for index in range(args.count):
        rgba, text, details = render_page(rng, paragraphs, fonts)
        name = os.path.join(args.output_dir, f'page{index}')
        with open(f'{name}.json', 'w', encoding='utf-8') as file:
            json.dump({'width': rgba.shape[1], 'height': rgba.shape[0], 'text': text, **details}, file)
        with open(f'{name}.rgba', 'wb') as file:
            file.write(rgba.tobytes())
        Image.fromarray(rgba[:, :, 0]).save(f'{name}.png')
    return 0


if __name__ == '__main__':
    sys.exit(main())
