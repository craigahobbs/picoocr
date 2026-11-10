# PicoOCR

PicoOCR reads the text of page photos in pure JavaScript. A small neural network recognizes each text line, and its
weights are part of the code - no network requests, no files to fetch, no WebAssembly, no dependencies. Import it and
it works, in the browser (a page or a Web Worker), offline, or in Node.

**Try it:** [PicoRead](https://craigahobbs.github.io/picoread/) - photograph a document's pages with your phone and
listen to them. PicoOCR reads each page and PicoVox speaks it, all on your device.

PicoOCR is part of the PicoVox family ([PicoVox](https://github.com/craigahobbs/picovox) speaks text; PicoOCR reads it).


## Usage

~~~ javascript
import {recognize} from 'picoocr';

// An ImageData (or any {width, height, data} of RGBA bytes) - e.g. from a canvas
const result = recognize(imageData, {'onProgress': (done, total) => console.log(`${done}/${total} lines`)});
console.log(result.text);
~~~

`recognize(image, options)` returns:

- `text` - the page's text in reading order: a line break between lines, a blank line between paragraphs
- `lines` - each line's `text`, `confidence` (0 to 1), `paragraph` (starts one), and `box` (in the leveled page)
- `angle` - the page's measured skew (radians), and `scale` - the scale it was analyzed at
- `orientation` - the page as photographed: `upright`, `mirrored`, `upside-down`, or `upside-down mirrored` (a
  mirrored or upside-down page is read upright)

Its options: `onProgress(done, total)` - called after each line is read; `orient` - detect a mirrored or upside-down
page (default true); `model` - a model loaded by `loadRecognizer(module)` (the package's by default).

It runs synchronously - call it in a Web Worker to keep a page responsive. The model loads on first use
(`loadRecognizer()` loads it ahead). `pageText(lines)` joins lines as `recognize` does.


## How it works

1. **The page** - grayscale, at most 2400 pixels on its longer side, its skew measured from the ink's connected
   components and leveled
2. **The lines** - binarized (Sauvola), the components grouped into lines, small marks (dots, accents, quotes) attached,
   the lines put in reading order (columns, and the titles that span them) and paragraphs
3. **Each line** - its ink box padded and scaled to 32 pixels high, its contrast normalized, and framed by margins
4. **The network** - four convolutions and three residual 1-D convolutions (187k parameters, 5-bit GPTQ weights - a
   172 KB module), its characters decoded by CTC's greedy rule

[DESIGN.md](DESIGN.md) is the design: the training and inference plan, the limits chosen and why, and the
measurements behind them.


## Development

PicoOCR is a [javascript-build](https://github.com/craigahobbs/javascript-build#readme) package; its training
(`train/`, the Python package `picoocr_train`) is a [python-build](https://github.com/craigahobbs/python-build#readme)
package:

~~~
make commit            # tests, lint, docs, 100% coverage - and the training package's gate
make train             # train the recognizer (build/train/run)
make train-export      # its best checkpoint as build/train/recognizer.js, checked against the package (make parity)
make train-install     # into lib/models/
~~~

The training renders text with Pillow's Raqm layout, which loads FriBiDi at runtime - on macOS, `brew install fribidi`
first (Linux distributions and the Docker images have it).
