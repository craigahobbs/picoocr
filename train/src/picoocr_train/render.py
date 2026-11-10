# Licensed under the MIT License
# https://github.com/craigahobbs/picoocr/blob/main/LICENSE

"""
A training sample - a text line as the package would see it in a phone photo of a page:

1. The line is rendered in a font at a pixel size, in its paragraph (the lines above and below, at a line pitch, so the
   line input's padding catches their ascenders and descenders as it does on a page), sometimes underlined, sometimes
   without ligatures.
2. The page is tilted slightly and stretched or squeezed (the residual skew and the perspective a leveled page keeps).
3. The ink is laid on paper - their levels, a lighting gradient across the page, the lens's blur, the sensor's noise,
   and JPEG's compression.
4. The line's ink box - its pixels at least INK_COVERAGE inked, as the package's binarization would find them - is
   jittered (the package's box is an estimate) and cut out by the line input (image.py - the package's lineInput).

A text whose line is too short for CTC (fewer steps within its margins than its characters and repeats) is never a
sample.
"""

import io
import math

import numpy
from PIL import Image, ImageDraw, ImageFilter

from .fonts import load_font
from .image import LINE_MARGIN, LINE_STRIDE, line_input


# A pixel is in the ink box if the line covers this fraction of it
INK_COVERAGE = 0.4

# The ink box's jitter, a fraction of its height
BOX_JITTER = 0.05


# The CTC steps a text needs - a step per character, and a blank between repeated characters
def ctc_steps(text):
    return len(text) + sum(1 for index in range(1, len(text)) if text[index] == text[index - 1])


# Text rendered on a coverage canvas (an 'L' image, 255 full ink) at a baseline origin
def draw_text(canvas, origin, text, font, features):
    ImageDraw.Draw(canvas).text(origin, text, fill=255, font=font, anchor='ls', features=features)


# A rendered line - (coverage [height, width] float 0 to 1, its ink box (x, y, width, height)), or None if it's blank
def render_coverage(rng, text, font, size, context):
    features = None if rng.random() < 0.7 else ['-liga']
    left, _, right, _ = font.getbbox(text, anchor='ls', features=features)
    pitch = size * rng.uniform(1.05, 1.6)
    margin_x = int(size * rng.uniform(0.3, 1.5))
    width = int(right - left) + 2 * margin_x
    baseline = int(size * 0.5 + pitch + size)
    height = int(baseline + pitch + size * 1.5)
    canvas = Image.new('L', (width, height), 0)
    line = Image.new('L', (width, height), 0)
    origin = (margin_x - left, baseline)
    draw_text(line, origin, text, font, features)

    # Underlined
    if rng.random() < 0.04:
        top = baseline + max(1, int(size * 0.1))
        ImageDraw.Draw(line).rectangle([origin[0], top, origin[0] + right - left, top + max(1, int(size * 0.05))], fill=255)

    # The paragraph's lines above and below
    for offset in (-pitch, pitch):
        if rng.random() < 0.6:
            draw_text(canvas, (margin_x - left + rng.uniform(-size, size), baseline + offset), context(), font, features)
    canvas = Image.fromarray(numpy.maximum(numpy.asarray(canvas), numpy.asarray(line)))

    # The residual skew, and the perspective's stretch
    if rng.random() < 0.5:
        angle = rng.uniform(-1, 1)
        canvas = canvas.rotate(angle, resample=Image.Resampling.BILINEAR)
        line = line.rotate(angle, resample=Image.Resampling.BILINEAR)
    stretch = rng.uniform(0.85, 1.15)
    stretched = (max(1, int(width * stretch)), height)
    canvas = canvas.resize(stretched, Image.Resampling.BILINEAR)
    line = line.resize(stretched, Image.Resampling.BILINEAR)

    # The ink box
    inked = numpy.asarray(line) >= INK_COVERAGE * 255
    rows = numpy.flatnonzero(inked.any(axis=1))
    columns = numpy.flatnonzero(inked.any(axis=0))
    if len(rows) == 0:
        return None
    box = (int(columns[0]), int(rows[0]), int(columns[-1] - columns[0] + 1), int(rows[-1] - rows[0] + 1))
    return numpy.asarray(canvas, dtype=numpy.float64) / 255, box


# A coverage canvas as a photographed gray page (uint8 [height, width])
def photograph(rng, coverage, size):
    height, width = coverage.shape
    paper = rng.uniform(140, 255)
    ink = rng.uniform(0, paper - 70)
    gray = paper - coverage * (paper - ink)

    # The lighting gradient
    angle = rng.uniform(0, 2 * math.pi)
    slope = rng.uniform(-0.4, 0.4) / math.hypot(width, height)
    ys, xs = numpy.mgrid[0:height, 0:width]
    gray = gray * (1 + slope * ((xs - width / 2) * math.cos(angle) + (ys - height / 2) * math.sin(angle)))

    # The lens's blur and the sensor's noise
    image = Image.fromarray(numpy.clip(gray, 0, 255).astype(numpy.uint8))
    if rng.random() < 0.7:
        image = image.filter(ImageFilter.GaussianBlur(size * rng.uniform(0, 0.05)))
    gray = numpy.asarray(image, dtype=numpy.float64)
    gray = gray + numpy.random.default_rng(rng.getrandbits(32)).normal(0, rng.uniform(0, 10), gray.shape)
    image = Image.fromarray(numpy.clip(gray, 0, 255).astype(numpy.uint8))

    # JPEG's compression
    if rng.random() < 0.4:
        buffer = io.BytesIO()
        image.save(buffer, format='JPEG', quality=rng.randint(25, 90))
        image = Image.open(io.BytesIO(buffer.getvalue()))
    return numpy.asarray(image)


# A training sample - (line input float32 [LINE_HEIGHT, width], text). rng - a random.Random; sampler - a
# texts.TextSampler; fonts - [(font, path)]; data - the recipe's data section.
def render_sample(rng, sampler, fonts, data):
    while True:
        text = sampler.sample()
        font, path = rng.choice(fonts)
        size = int(round(math.exp(rng.uniform(math.log(data['minSize']), math.log(data['maxSize'])))))
        loaded = load_font(path, size, font.get('variation'))
        rendered = render_coverage(rng, text, loaded, size, lambda: sampler.sample()[:len(text) + 10])
        if rendered is None:
            continue
        coverage, box = rendered
        page = photograph(rng, coverage, size).astype(numpy.float32)
        jitter = BOX_JITTER * box[3]
        top = box[1] + round(rng.uniform(-jitter, jitter))
        bottom = box[1] + box[3] + round(rng.uniform(-jitter, jitter))
        left = box[0] + round(rng.uniform(-jitter, jitter))
        right = box[0] + box[2] + round(rng.uniform(-jitter, jitter))
        if bottom - top < 2 or right - left < 1:
            continue
        line = line_input(page, (left, top, right - left, bottom - top))
        if (line.shape[1] - 2 * LINE_MARGIN) // LINE_STRIDE >= ctc_steps(text):
            return line, text
