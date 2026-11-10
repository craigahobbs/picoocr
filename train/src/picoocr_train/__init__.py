# Licensed under the MIT License
# https://github.com/craigahobbs/picoocr/blob/main/LICENSE

"""
PicoOCR's training pipeline - the line recognizer of the picoocr npm package (lib/models/recognizer.js), trained on text
lines it renders (no scanned or photographed data):

- downloads.py - the pinned downloads: a URL's bytes fetched once and verified by SHA-256
- fonts.py - the pinned fonts (train/data/fonts.json) - their train and validation splits
- texts.py - the training lines' texts: prose from the pinned books (train/data/books.json), formatted values (numbers,
  dates, prices, addresses), and random characters
- image.py - the port of lib/image.js: the line input (lineInput), exactly as the package computes it
- render.py - a training sample: a text line rendered in its paragraph, degraded as a phone photo is (lighting,
  blur, noise, JPEG), and cut out by the line input
- data.py - the training batches (worker processes, bucketed by width) and the fixed validation sets
- model.py - the network (the package's lib/model.js runs its export) and CTC's greedy decoding
- train.py - the training: CTC loss, AdamW, the validation's character error rate, the checkpoints
- quant.py - the weights' quantization: per output channel, GPTQ or nearest rounding, packed bits
- export.py - the checkpoint as the package's model module: batch norms folded, weights quantized (quant.py), the
  quantized model's validation, and the parity data for tools/parity.js

The recipe, train/data/train.json, holds every setting: the alphabet, the network's sizes, the data's ranges, and the
training's steps.
"""

import json
import os
import sys
import time


# The picoocr repository's root, the training data's directory, and the downloads' directory
ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
DATA_DIR = os.path.join(ROOT, 'train', 'data')
DOWNLOADS_DIR = os.path.join(ROOT, 'downloads')

# The recipe
RECIPE_PATH = os.path.join(DATA_DIR, 'train.json')


# A JSON file's value
def read_json(path):
    with open(path, encoding='utf-8') as file:
        return json.load(file)


# Log a message to stderr, with the time
def log(message):
    print(f'{time.strftime("%H:%M:%S")} {message}', file=sys.stderr, flush=True)
