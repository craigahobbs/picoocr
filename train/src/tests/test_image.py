# Licensed under the MIT License
# https://github.com/craigahobbs/picoocr/blob/main/LICENSE

"""
The tests of picoocr_train.image
"""

import unittest

import numpy

from picoocr_train.image import LINE_HEIGHT, LINE_MARGIN, LINE_STRIDE, crop, js_round, line_input, resample_weights, resize


class TestImage(unittest.TestCase):

    def test_js_round(self):
        self.assertEqual([js_round(value) for value in (0.5, 1.5, 2.4, 2.5)], [1, 2, 2, 3])

    def test_resample_weights(self):
        same = resample_weights(3, 3)
        self.assertEqual([start for start, _ in same], [0, 0, 1])
        self.assertEqual(same[1][1].tolist(), [0, 1, 0])
        half = resample_weights(4, 2)
        self.assertEqual([start for start, _ in half], [0, 1])
        self.assertEqual([round(weight * 7, 4) for weight in half[0][1].tolist()], [3, 3, 1])

    def test_resize(self):
        image = numpy.array([[1, 2, 3], [4, 5, 6]], dtype=numpy.float32)
        self.assertEqual(resize(image, 3, 2).tolist(), image.tolist())
        smaller = resize(numpy.array([[0, 0, 100, 100]] * 2, dtype=numpy.float32), 2, 1)
        self.assertEqual([round(value, 4) for value in smaller[0].tolist()], [14.2857, 85.7143])
        self.assertEqual(resize(numpy.array([[0, 100]], dtype=numpy.float32), 4, 1).tolist(), [[0, 25, 75, 100]])

    def test_crop(self):
        image = numpy.array([[1, 2], [3, 4]], dtype=numpy.float32)
        self.assertEqual(crop(image, -1, -1, 4, 3).tolist(), [[1, 1, 2, 2], [1, 1, 2, 2], [3, 3, 4, 4]])

    def test_line_input(self):
        # The same page as the package's test (test/testImage.js)
        page = numpy.full((40, 100), 200, dtype=numpy.float32)
        page[15:25, 10:90] = 20
        line = line_input(page, (10, 15, 80, 10))
        self.assertEqual(line.shape, (LINE_HEIGHT, 316))
        self.assertEqual(line.shape[1] % LINE_STRIDE, 0)
        self.assertEqual(line[16, LINE_MARGIN + 100], 1)
        self.assertEqual(line[0, LINE_MARGIN + 100], 0)
        self.assertEqual(line[16, 0], 0)

        # Blank
        self.assertEqual(line_input(numpy.full((40, 100), 128, dtype=numpy.float32), (0, 0, 10, 10)).max(), 0)
