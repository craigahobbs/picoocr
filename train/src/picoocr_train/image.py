# Licensed under the MIT License
# https://github.com/craigahobbs/picoocr/blob/main/LICENSE

"""
The port of lib/image.js - the recognizer's line input (line_input), computed exactly as the package computes it: the
same arithmetic (float64, stored as float32 where the package stores a Float32Array) in the same order, so a training
line is what the package would give the network for the same page. tools/parity.js checks the package against this
module's outputs (export.py's parity data).

A gray image is a float32 array [height, width] of luminance (0 black, 255 white).
"""

import math

import numpy


# The line input's constants (lib/image.js)
LINE_HEIGHT = 32
LINE_MARGIN = 48
LINE_STRIDE = 4
LINE_PAD = 0.1
INK_PERCENTILE = 0.01
PAPER_PERCENTILE = 0.9
MIN_CONTRAST = 16


# JavaScript's Math.round of a non-negative number
def js_round(value):
    return math.floor(value + 0.5)


# The resampling weights from a source axis to a destination axis (lib/image.js resampleWeights) - [(start, weights)],
# the weights float32
def resample_weights(src_size, dst_size):
    scale = src_size / dst_size
    support = max(1.0, scale)
    result = []
    for dst in range(dst_size):
        center = (dst + 0.5) * scale
        start = max(0, math.floor(center - support))
        end = min(src_size, math.ceil(center + support))
        weights = [max(0.0, 1 - abs((src + 0.5 - center) / support)) for src in range(start, end)]
        total = 0.0
        for weight in weights:
            total += weight
        result.append((start, numpy.array([weight / total for weight in weights], dtype=numpy.float32)))
    return result


# Resize a gray image (lib/image.js resize) - the rows first, each a float64 sum stored as float32; then the columns,
# accumulated in float32 a source row at a time
def resize(image, width, height):
    src_height, src_width = image.shape
    rows = numpy.zeros((src_height, width), dtype=numpy.float32)
    image64 = image.astype(numpy.float64)
    for x, (start, weights) in enumerate(resample_weights(src_width, width)):
        products = image64[:, start:start + len(weights)] * weights.astype(numpy.float64)
        total = numpy.zeros(src_height, dtype=numpy.float64)
        for index in range(len(weights)):
            total += products[:, index]
        rows[:, x] = total
    output = numpy.zeros((height, width), dtype=numpy.float32)
    for y, (start, weights) in enumerate(resample_weights(src_height, height)):
        for index, weight in enumerate(weights):
            output[y] = (output[y].astype(numpy.float64) + float(weight) * rows[start + index].astype(numpy.float64)).astype(numpy.float32)
    return output


# A region of a gray image - the edge pixels extended outside it (lib/image.js crop)
def crop(image, left, top, width, height):
    ys = numpy.clip(numpy.arange(top, top + height), 0, image.shape[0] - 1)
    xs = numpy.clip(numpy.arange(left, left + width), 0, image.shape[1] - 1)
    return image[numpy.ix_(ys, xs)].astype(numpy.float32)


# The recognizer's input for a text line (lib/image.js lineInput) - box (x, y, width, height), its ink box in the page -
# a float32 array [LINE_HEIGHT, width] of ink levels (0 paper to 1 ink)
def line_input(page, box):
    box_x, box_y, box_width, box_height = box
    pad = js_round(LINE_PAD * box_height)
    crop_width = box_width + 2 * pad
    crop_height = box_height + 2 * pad
    scaled_width = max(1, js_round(crop_width * LINE_HEIGHT / crop_height))
    line = resize(crop(page, box_x - pad, box_y - pad, crop_width, crop_height), scaled_width, LINE_HEIGHT)

    # The ink and paper levels
    values = numpy.sort(line.reshape(-1))
    ink = float(values[math.floor(INK_PERCENTILE * (len(values) - 1))])
    paper = float(values[math.floor(PAPER_PERCENTILE * (len(values) - 1))])
    contrast = max(MIN_CONTRAST, paper - ink)

    # The normalized line, framed by its margins
    width = math.ceil((scaled_width + 2 * LINE_MARGIN) / LINE_STRIDE) * LINE_STRIDE
    output = numpy.zeros((LINE_HEIGHT, width), dtype=numpy.float32)
    levels = (paper - line.astype(numpy.float64)) / contrast
    output[:, LINE_MARGIN:LINE_MARGIN + scaled_width] = numpy.clip(levels, 0, 1)
    return output
