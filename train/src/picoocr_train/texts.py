# Licensed under the MIT License
# https://github.com/craigahobbs/picoocr/blob/main/LICENSE

"""
The training lines' texts - a sampler (TextSampler) of three kinds, mixed by the recipe's data.kinds:

- book - a span of words from the pinned books' prose (train/data/books.json), sometimes typeset - curly quotes and
  apostrophes, em dashes - and sometimes in capitals or title case
- format - a formatted value: numbers, prices, dates, times, phone numbers, percentages, addresses, references
- random - random characters of the alphabet in words - every character seen, with no language to lean on

Every text is at most the recipe's data.maxChars characters, single-spaced, and of the alphabet's characters.
"""

import os
import re

from . import DATA_DIR, DOWNLOADS_DIR, read_json
from .downloads import pinned_path


# The pinned books
BOOKS_PATH = os.path.join(DATA_DIR, 'books.json')

# The Project Gutenberg text's body markers
_START = re.compile(r'^\*\*\* ?START OF (?:THE|THIS) PROJECT GUTENBERG.*$', re.MULTILINE)
_END = re.compile(r'^\*\*\* ?END OF (?:THE|THIS) PROJECT GUTENBERG.*$', re.MULTILINE)

# A short text's length limit
SHORT_CHARS = 20

# Runs of whitespace
_SPACES = re.compile(r'\s+')


# The pinned books
def load_books(path=BOOKS_PATH):
    return read_json(path)['books']


# The local path of a book's text - fetched if missing
def book_path(book, downloads_dir=DOWNLOADS_DIR):
    return pinned_path(book, os.path.join(downloads_dir, 'books', f'{book["name"]}.txt'))


# A Project Gutenberg text's paragraphs (its body's, single-spaced) of only the alphabet's characters - italics'
# underscores dropped
def book_paragraphs(text, alphabet):
    start = _START.search(text)
    end = _END.search(text)
    body = text[start.end() if start else 0:end.start() if end else len(text)]
    chars = set(alphabet)
    paragraphs = []
    for block in re.split(r'\n\s*\n', body.replace('\r\n', '\n')):
        paragraph = _SPACES.sub(' ', block.replace('_', '')).strip()
        if paragraph and set(paragraph) <= chars:
            paragraphs.append(paragraph)
    return paragraphs


# A text typeset - straight quotes curled (opening after a space or an opening bracket, closing otherwise) and double
# hyphens as em dashes
def typeset(text):
    result = []
    for index, char in enumerate(text):
        opening = index == 0 or text[index - 1] in ' ([{—'
        if char == '"':
            result.append('“' if opening else '”')
        elif char == "'":
            result.append('‘' if opening else '’')
        else:
            result.append(char)
    return ''.join(result).replace('--', '—')


# Words of a text as a span of at most max_chars characters, starting at a word (the first word cut if it alone is
# longer)
def word_span(words, start, max_chars):
    span = words[start][:max_chars]
    for word in words[start + 1:]:
        if len(span) + 1 + len(word) > max_chars:
            break
        span += ' ' + word
    return span


# The month names, for dates
_MONTHS = ['January', 'February', 'March', 'April', 'May', 'June', 'July', 'August', 'September', 'October',
           'November', 'December']


# The training lines' texts
class TextSampler:

    # rng - a random.Random; paragraphs - the books' paragraphs; recipe - the recipe's data section
    def __init__(self, rng, paragraphs, alphabet, data):
        self.rng = rng
        self.paragraphs = [paragraph.split(' ') for paragraph in paragraphs]
        self.alphabet = [char for char in alphabet if char != ' ']
        self.charset = set(alphabet)
        self.max_chars = data['maxChars']
        self.kinds = list(data['kinds'].keys())
        self.weights = list(data['kinds'].values())

    # A text's length limit - mostly long lines (SHORT_CHARS to maxChars), sometimes short ones (up to SHORT_CHARS)
    def limit(self):
        short = min(SHORT_CHARS, self.max_chars)
        if self.rng.random() < 0.3:
            return self.rng.randint(1, short)
        return self.rng.randint(short, self.max_chars)

    # A text of a random kind - of the alphabet's characters (a text a case change took outside it is drawn again)
    def sample(self):
        while True:
            kind = self.rng.choices(self.kinds, self.weights)[0]
            if kind == 'book':
                text = self.book()
            elif kind == 'format':
                text = self.formatted()
            else:
                text = self.random()
            text = _SPACES.sub(' ', text).strip()[:self.max_chars].strip()
            if text and set(text) <= self.charset:
                return text

    # A span of book prose
    def book(self):
        words = self.rng.choice(self.paragraphs)
        text = word_span(words, self.rng.randrange(len(words)), self.limit())
        style = self.rng.random()
        if style < 0.5:
            text = typeset(text)
        if 0.5 <= style < 0.58:
            text = text.upper()
        elif 0.58 <= style < 0.63:
            text = text.title()
        return text

    # Random characters in words
    def random(self):
        limit = self.limit()
        words = []
        while sum(len(word) + 1 for word in words) < limit:
            words.append(''.join(self.rng.choice(self.alphabet) for _ in range(self.rng.randint(1, 10))))
        return ' '.join(words)[:limit]

    # Digits
    def digits(self, count):
        return ''.join(str(self.rng.randint(0, 9)) for _ in range(count))

    # A number with thousands separators
    def number(self):
        return f'{self.rng.randint(0, 10 ** self.rng.randint(1, 7)):,}'

    # A formatted value, sometimes with words of prose around it
    def formatted(self):
        rng = self.rng
        values = [
            lambda: f'${self.number()}.{self.digits(2)}',
            lambda: f'{rng.randint(1, 12)}:{self.digits(2)} {rng.choice(["a.m.", "p.m.", "AM", "PM"])}',
            lambda: f'{rng.choice(_MONTHS)} {rng.randint(1, 31)}, {rng.randint(1700, 2099)}',
            lambda: f'{rng.randint(1, 12)}/{rng.randint(1, 31)}/{self.digits(rng.choice([2, 4]))}',
            lambda: f'{rng.randint(1900, 2099)}-{rng.randint(1, 12):02d}-{rng.randint(1, 31):02d}',
            lambda: f'({self.digits(3)}) {self.digits(3)}-{self.digits(4)}',
            lambda: f'{rng.randint(0, 100)}.{self.digits(1)}%',
            lambda: f'{self.number()} {rng.choice(["Main St.", "Oak Avenue", "Elm Street", "Route", "Suite"])}',
            lambda: f'{rng.choice(["No.", "p.", "pp.", "Vol.", "Fig.", "Table", "§", "#", "Chapter"])} {rng.randint(1, 999)}',
            lambda: f'{rng.choice(["Page", "Item", "Step"])} {rng.randint(1, 99)} of {rng.randint(1, 99)}',
            lambda: f'{rng.randint(1, 99)}°{rng.choice(["F", "C", ""])}',
            lambda: f'• {word_span(rng.choice(self.paragraphs), 0, rng.randint(5, 30))}',
            lambda: f'{rng.randint(1, 20)}. {word_span(rng.choice(self.paragraphs), 0, rng.randint(5, 30))}',
            lambda: f'{self.random()[:8].lower()}@{self.random()[:8].lower()}.{rng.choice(["com", "org", "net"])}',
            lambda: f'£{self.number()} €{self.number()}',
            self.number
        ]
        text = rng.choice(values)()
        if rng.random() < 0.5:
            words = rng.choice(self.paragraphs)
            prose = word_span(words, rng.randrange(len(words)), rng.randint(5, 30))
            text = f'{prose} {text}' if rng.random() < 0.5 else f'{text} {prose}'
        return text[:self.limit()]
