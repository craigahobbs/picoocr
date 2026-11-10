# Licensed under the MIT License
# https://github.com/craigahobbs/picoocr/blob/main/LICENSE

"""
The tests of picoocr_train.downloads
"""

import os
import tempfile
import unittest
import unittest.mock

from picoocr_train import downloads
from picoocr_train.downloads import PinError, pinned_path, sha256


class TestDownloads(unittest.TestCase):

    def test_fetch_url(self):
        response = unittest.mock.MagicMock()
        response.__enter__.return_value.read.return_value = b'data'
        with unittest.mock.patch('urllib.request.urlopen', return_value=response) as urlopen:
            self.assertEqual(downloads.fetch_url('https://example.com/a'), b'data')
        self.assertEqual(urlopen.call_args[0][0].get_header('User-agent'), 'picoocr-train')

    def test_pinned_path(self):
        pin = {'url': 'https://example.com/a', 'sha256': sha256(b'good'), 'archive': 'https://archive.org/a'}
        with tempfile.TemporaryDirectory() as temp_dir:
            path = os.path.join(temp_dir, 'sub', 'a.txt')

            # Fetched - the URL's bytes are wrong, the archive's right
            with unittest.mock.patch.object(downloads, 'fetch_url', side_effect=[b'bad', b'good']) as fetch_url:
                self.assertEqual(pinned_path(pin, path), path)
            self.assertEqual([call[0][0] for call in fetch_url.call_args_list], [pin['url'], pin['archive']])
            with open(path, 'rb') as file:
                self.assertEqual(file.read(), b'good')

            # Already there
            with unittest.mock.patch.object(downloads, 'fetch_url') as fetch_url:
                self.assertEqual(pinned_path(pin, path), path)
            fetch_url.assert_not_called()

            # There, but changed - fetched again
            with open(path, 'wb') as file:
                file.write(b'changed')
            with unittest.mock.patch.object(downloads, 'fetch_url', return_value=b'good'):
                self.assertEqual(pinned_path(pin, path), path)

            # No source matches
            with unittest.mock.patch.object(downloads, 'fetch_url', return_value=b'bad'):
                with self.assertRaises(PinError) as context:
                    pinned_path({'url': pin['url'], 'sha256': pin['sha256']}, os.path.join(temp_dir, 'b.txt'))
            self.assertEqual(str(context.exception), f'https://example.com/a does not match its pin (sha256 {pin["sha256"]})')
