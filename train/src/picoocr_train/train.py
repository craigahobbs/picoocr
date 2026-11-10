# Licensed under the MIT License
# https://github.com/craigahobbs/picoocr/blob/main/LICENSE

"""
The recognizer's training - CTC loss on endless rendered batches (data.py), AdamW with a linear warmup and a cosine
decay, and the gradient's norm clipped. Every validateEvery steps the character error rate (greedy decoding) is
measured on the two fixed validation sets - "seen" (the train fonts) and "unseen" (the validation fonts, never trained
on) - and the checkpoints are written to the run directory: last.pt, and best.pt (the lowest unseen error). Each
step's loss and each validation are logged to the run directory's log.jsonl.

    python -m picoocr_train.train build/train/run [--steps N] [--until N] [--workers N] [--threads N] [--init CHECKPOINT]
"""

import argparse
import json
import math
import os
import random
import sys
import time

import torch
from torch.nn import functional

from . import RECIPE_PATH, log, read_json
from .data import LineBatches, render_samples, sorted_batches
from .fonts import font_path, load_fonts
from .model import Recognizer, character_error_rate, greedy_decode
from .texts import book_paragraphs, book_path, load_books


# The steps between the training log's lines
LOG_EVERY = 50


# The recipe's fonts of a split, fetched - [(font, path)]
def split_fonts(split):
    return [(font, font_path(font)) for font in load_fonts(split)]


# The pinned books' paragraphs
def load_paragraphs(alphabet):
    paragraphs = []
    for book in load_books():
        with open(book_path(book), encoding='utf-8-sig') as file:
            paragraphs.extend(book_paragraphs(file.read(), alphabet))
    return paragraphs


# The learning rate at a step - a linear warmup, then a cosine decay to 1% of the peak
def learning_rate(step, config):
    if step < config['warmup']:
        return config['lr'] * (step + 1) / config['warmup']
    progress = min(1.0, (step - config['warmup']) / max(1, config['steps'] - config['warmup']))
    return config['lr'] * (0.01 + 0.99 * 0.5 * (1 + math.cos(math.pi * progress)))


# The CTC loss of a batch
def ctc_loss(model, batch):
    inputs, steps, targets, target_lengths, _ = batch
    scores = functional.log_softmax(model(inputs), dim=1).permute(2, 0, 1)
    return functional.ctc_loss(scores, targets, steps, target_lengths, zero_infinity=True)


# The character error rate of the model on batches
def evaluate(model, batches, alphabet):
    model.eval()
    texts = []
    references = []
    with torch.no_grad():
        for inputs, steps, _, _, batch_texts in batches:
            texts.extend(greedy_decode(model(inputs), steps.tolist(), alphabet))
            references.extend(batch_texts)
    model.train()
    return character_error_rate(texts, references), list(zip(texts, references))


# The validation sets - {name: batches}
def validation_sets(recipe, paragraphs):
    data = recipe['data']
    sets = {}
    for name, split, seed in (('seen', 'train', data['validationSeed']), ('unseen', 'validation', data['validationSeed'] + 1)):
        samples = render_samples(recipe, split_fonts(split), paragraphs, seed, data['validationSamples'])
        sets[name] = sorted_batches(samples, recipe['alphabet'], recipe['train']['batch'])
    return sets


# Train the recognizer
def train(run_dir, recipe, steps=None, until=None, workers=3, init=None):
    train_config = dict(recipe['train'], **({'steps': steps} if steps else {}))
    last = min(until, train_config['steps']) if until else train_config['steps']
    alphabet = recipe['alphabet']
    torch.manual_seed(train_config['seed'])
    random.seed(train_config['seed'])
    os.makedirs(run_dir, exist_ok=True)

    # The model, from scratch or a checkpoint
    model = Recognizer(len(alphabet) + 1, recipe['model'])
    optimizer = torch.optim.AdamW(model.parameters(), lr=train_config['lr'], weight_decay=train_config['weightDecay'])
    start = 0
    best = math.inf
    if init is not None:
        checkpoint = torch.load(init, weights_only=False)
        model.load_state_dict(checkpoint['model'])
        optimizer.load_state_dict(checkpoint['optimizer'])
        start = checkpoint['step']
        best = checkpoint['best']
    log(f'training {sum(param.numel() for param in model.parameters())} parameters, steps {start} to {last} of {train_config["steps"]}')

    # The data - a continued training's seeded by its start, so it doesn't repeat the batches it trained on
    paragraphs = load_paragraphs(alphabet)
    validation = validation_sets(recipe, paragraphs)
    loader = torch.utils.data.DataLoader(
        LineBatches(recipe, split_fonts('train'), paragraphs, train_config['seed'] + start), batch_size=None, num_workers=workers,
        prefetch_factor=4 if workers else None, persistent_workers=workers > 0
    )

    # The training loop
    with open(os.path.join(run_dir, 'log.jsonl'), 'a', encoding='utf-8') as log_file:
        losses = []
        started = time.time()
        for step, batch in zip(range(start, last), loader):
            rate = learning_rate(step, train_config)
            for group in optimizer.param_groups:
                group['lr'] = rate
            loss = ctc_loss(model, batch)
            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), train_config['clip'])
            optimizer.step()
            losses.append(loss.item())

            # Log the loss
            if (step + 1) % LOG_EVERY == 0:
                seconds = (time.time() - started) / len(losses)
                entry = {'step': step + 1, 'loss': sum(losses) / len(losses), 'lr': rate, 'seconds': seconds}
                log_file.write(json.dumps(entry) + '\n')
                log_file.flush()
                log(f'step {step + 1} loss {entry["loss"]:.4f} lr {rate:.6f} {seconds:.3f} s/step')
                losses = []
                started = time.time()

            # Validate and checkpoint
            if (step + 1) % train_config['validateEvery'] == 0 or step + 1 == last:
                errors = {name: evaluate(model, batches, alphabet) for name, batches in validation.items()}
                entry = {'step': step + 1, **{name: error for name, (error, _) in errors.items()}}
                log_file.write(json.dumps(entry) + '\n')
                log_file.flush()
                log(f'step {step + 1} validation seen {entry["seen"]:.4f} unseen {entry["unseen"]:.4f}')
                for text, reference in [pair for pair in errors['unseen'][1] if pair[0] != pair[1]][:5]:
                    log(f'    {reference!r} -> {text!r}')
                best = min(best, entry['unseen'])
                checkpoint = {
                    'model': model.state_dict(), 'optimizer': optimizer.state_dict(), 'step': step + 1, 'best': best,
                    'recipe': recipe, 'validation': entry
                }
                torch.save(checkpoint, os.path.join(run_dir, 'last.pt'))
                if entry['unseen'] == best:
                    torch.save(checkpoint, os.path.join(run_dir, 'best.pt'))
                started = time.time()
    return best


def main(argv=None):
    parser = argparse.ArgumentParser(prog='python -m picoocr_train.train', description=__doc__.strip().split('\n\n', maxsplit=1)[0])
    parser.add_argument('run_dir', help='the run directory - its checkpoints and log')
    parser.add_argument('--steps', type=int, help="the training's steps - its schedule's length (the recipe's by default)")
    parser.add_argument('--until', type=int, help='stop at this step, the schedule unchanged (a pilot run)')
    parser.add_argument('--workers', type=int, default=3, help='the data worker processes (default 3)')
    parser.add_argument('--threads', type=int, help="PyTorch's threads (its default by default)")
    parser.add_argument('--init', help='a checkpoint to continue from')
    parser.add_argument('--recipe', default=RECIPE_PATH, help='the recipe (train/data/train.json by default)')
    args = parser.parse_args(argv)
    if args.threads:
        torch.set_num_threads(args.threads)
    best = train(args.run_dir, read_json(args.recipe), steps=args.steps, until=args.until, workers=args.workers, init=args.init)
    log(f'best unseen character error rate {best:.4f}')
    return 0


if __name__ == '__main__':
    sys.exit(main())
