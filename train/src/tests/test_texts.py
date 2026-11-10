# Licensed under the MIT License
# https://github.com/craigahobbs/picoocr/blob/main/LICENSE

"""
The tests of picoocr_train.texts
"""

import random
import unittest
import unittest.mock

from picoocr_train import texts
from picoocr_train.texts import TextSampler, book_paragraphs, book_path, load_books, typeset, word_span

from .helpers import PARAGRAPHS, tiny_recipe


class TestTexts(unittest.TestCase):

    def test_load_books(self):
        books = load_books()
        self.assertGreater(len(books), 5)
        self.assertTrue(all(len(book['sha256']) == 64 for book in books))

    def test_book_path(self):
        with unittest.mock.patch.object(texts, 'pinned_path', side_effect=lambda pin, path: path):
            self.assertEqual(book_path({'name': 'pg1'}, '/downloads'), '/downloads/books/pg1.txt')

    def test_book_paragraphs(self):
        text = (
            'Header\r\n*** START OF THE PROJECT GUTENBERG EBOOK X ***\r\n\r\nThe _first_\r\nparagraph.\r\n\r\n'
            'Not \u4e2d ascii.\r\n\r\n   \r\n\r\nLast one.\r\n*** END OF THE PROJECT GUTENBERG EBOOK X ***\r\nLicense'
        )
        self.assertEqual(book_paragraphs(text, tiny_recipe()['alphabet'] + 'TNL'), ['The first paragraph.', 'Last one.'])

        # No markers - the whole text
        self.assertEqual(book_paragraphs('a b\n\nc', 'abc '), ['a b', 'c'])

    def test_typeset(self):
        self.assertEqual(
            typeset('"Hi," she said -- \'no\' it\'s ("x")'),
            '\u201cHi,\u201d she said \u2014 \u2018no\u2019 it\u2019s (\u201cx\u201d)'
        )

    def test_word_span(self):
        words = ['one', 'two', 'three', 'fourteen']
        self.assertEqual(word_span(words, 0, 9), 'one two')
        self.assertEqual(word_span(words, 1, 100), 'two three fourteen')
        self.assertEqual(word_span(words, 3, 4), 'four')

    def test_sampler(self):
        recipe = tiny_recipe()
        for kind in ('book', 'format', 'random'):
            data = dict(recipe['data'], kinds={kind: 1}, maxChars=48)
            alphabet = recipe['alphabet'] + '0123456789$:/()%#@.\u00a7\u00b0\u2022\u00a3\u20acABCDEFGHIJKLMNOPQRSTUVWXYZ'
            sampler = TextSampler(random.Random(3), PARAGRAPHS, alphabet, data)
            for _ in range(200):
                text = sampler.sample()
                self.assertTrue(0 < len(text) <= 48, text)
                self.assertEqual(text, text.strip())
                self.assertNotIn('  ', text)
                self.assertTrue(set(text) <= sampler.charset, text)

    def test_sampler_redraw(self):
        # A text outside the alphabet (a case change) is drawn again
        recipe = tiny_recipe()
        sampler = TextSampler(random.Random(1), PARAGRAPHS, recipe['alphabet'], dict(recipe['data'], kinds={'book': 1}))
        with unittest.mock.patch.object(sampler, 'book', side_effect=['ABC', '', 'abc']):
            self.assertEqual(sampler.sample(), 'abc')
