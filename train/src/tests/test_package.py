# Licensed under the MIT License
# https://github.com/craigahobbs/picoocr/blob/main/LICENSE

"""
The tests of picoocr_train
"""

import io
import os
import re
import tempfile
import unittest
import unittest.mock

import picoocr_train
from picoocr_train import DATA_DIR, ROOT, read_json


class TestPackage(unittest.TestCase):

    def test_paths(self):
        self.assertTrue(os.path.isfile(os.path.join(ROOT, 'package.json')))
        self.assertTrue(os.path.isfile(os.path.join(DATA_DIR, 'train.json')))

    def test_read_json(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            path = os.path.join(temp_dir, 'a.json')
            with open(path, 'w', encoding='utf-8') as file:
                file.write('{"a": 1}')
            self.assertEqual(read_json(path), {'a': 1})

    def test_log(self):
        with unittest.mock.patch('sys.stderr', io.StringIO()) as stderr:
            picoocr_train.log('hello')
        self.assertRegex(stderr.getvalue(), r'^\d\d:\d\d:\d\d hello\n$')

    def test_requirements(self):
        # The package's dependencies are pinned as requirements.txt pins them
        with open(os.path.join(ROOT, 'train', 'requirements.txt'), encoding='utf-8') as file:
            requirements = {line.strip() for line in file if line.strip() and not line.startswith('#')}
        with open(os.path.join(ROOT, 'train', 'pyproject.toml'), encoding='utf-8') as file:
            dependencies = {match.replace(' ', '') for match in re.findall(r'^    "([^"]+)"', file.read(), re.MULTILINE)}
        self.assertEqual(dependencies, requirements)
