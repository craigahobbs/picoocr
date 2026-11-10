# Licensed under the MIT License
# https://github.com/craigahobbs/picoocr/blob/main/LICENSE

"""
The training data - endless batches of rendered lines (render.py) from worker processes, and the fixed validation sets.

Each worker renders a pool of POOL batches' lines, sorts them by width, and yields them as batches in a random order -
so a batch's lines are of similar widths (little padding). A batch is padded with paper (0) to its widest line - more
margin, which the network already sees around every line.
"""

import random

import numpy
import torch

from .render import render_sample
from .texts import TextSampler


# The batches a worker sorts by width at a time
POOL = 8


# A batch of samples [(line, text)] - (inputs [batch, 1, height, width], steps [batch] (each line's), targets (the
# texts' classes, concatenated), target lengths [batch], texts)
def collate(samples, alphabet):
    index = {char: cls for cls, char in enumerate(alphabet, 1)}
    width = max(line.shape[1] for line, _ in samples)
    inputs = numpy.zeros((len(samples), 1, samples[0][0].shape[0], width), dtype=numpy.float32)
    for ix, (line, _) in enumerate(samples):
        inputs[ix, 0, :, :line.shape[1]] = line
    texts = [text for _, text in samples]
    return (
        torch.from_numpy(inputs),
        torch.tensor([line.shape[1] // 4 for line, _ in samples], dtype=torch.long),
        torch.tensor([index[char] for text in texts for char in text], dtype=torch.long),
        torch.tensor([len(text) for text in texts], dtype=torch.long),
        texts
    )


# Rendered samples - a count of them, from a seed
def render_samples(recipe, fonts, paragraphs, seed, count):
    rng = random.Random(seed)
    sampler = TextSampler(rng, paragraphs, recipe['alphabet'], recipe['data'])
    return [render_sample(rng, sampler, fonts, recipe['data']) for _ in range(count)]


# Batches of samples, sorted by width
def sorted_batches(samples, alphabet, batch_size):
    ordered = sorted(samples, key=lambda sample: sample[0].shape[1])
    return [collate(ordered[ix:ix + batch_size], alphabet) for ix in range(0, len(ordered), batch_size)]


# The training batches - endless, each worker (torch's DataLoader) seeded by its id. (An iterable dataset has no
# __getitem__ - pylint sees the abstract method of its base, Dataset.)
class LineBatches(torch.utils.data.IterableDataset):  # pylint: disable=abstract-method

    def __init__(self, recipe, fonts, paragraphs, seed):
        super().__init__()
        self.recipe = recipe
        self.fonts = fonts
        self.paragraphs = paragraphs
        self.seed = seed

    def __iter__(self):
        info = torch.utils.data.get_worker_info()
        worker = info.id if info is not None else 0
        rng = random.Random(self.seed * 1000 + worker)
        sampler = TextSampler(rng, self.paragraphs, self.recipe['alphabet'], self.recipe['data'])
        batch_size = self.recipe['train']['batch']
        while True:
            samples = [render_sample(rng, sampler, self.fonts, self.recipe['data']) for _ in range(batch_size * POOL)]
            batches = sorted_batches(samples, self.recipe['alphabet'], batch_size)
            rng.shuffle(batches)
            yield from batches
