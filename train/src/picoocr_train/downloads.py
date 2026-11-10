# Licensed under the MIT License
# https://github.com/craigahobbs/picoocr/blob/main/LICENSE

"""
Pinned downloads - a URL's bytes fetched once into the downloads directory and verified by their SHA-256. A download
whose hash doesn't match is never used: the fetch tries the pin's archived copy, if any, and otherwise fails.
"""

import hashlib
import os
import urllib.request


# The downloads' User-Agent
USER_AGENT = 'picoocr-train'


# A download's failure - its hash doesn't match its pin
class PinError(Exception):
    pass


# The SHA-256 of bytes (hex)
def sha256(data):
    return hashlib.sha256(data).hexdigest()


# A URL's bytes
def fetch_url(url):
    request = urllib.request.Request(url, headers={'User-Agent': USER_AGENT})
    with urllib.request.urlopen(request, timeout=60) as response:
        return response.read()


# The path of a pinned download (a pin's url, sha256, and optional archive) - fetched if it isn't there
def pinned_path(pin, path):
    if os.path.isfile(path):
        with open(path, 'rb') as file:
            if sha256(file.read()) == pin['sha256']:
                return path
    for url in [pin['url']] + ([pin['archive']] if 'archive' in pin else []):
        data = fetch_url(url)
        if sha256(data) == pin['sha256']:
            os.makedirs(os.path.dirname(path), exist_ok=True)
            with open(f'{path}.tmp', 'wb') as file:
                file.write(data)
            os.replace(f'{path}.tmp', path)
            return path
    raise PinError(f'{pin["url"]} does not match its pin (sha256 {pin["sha256"]})')
