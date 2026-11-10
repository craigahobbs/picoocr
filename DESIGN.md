# PicoOCR design

PicoOCR reads printed text from phone photos of pages, in pure JavaScript, with its network in its code. This is its
design - the inference and its API, the training, and the limits chosen ahead of time and why - and the measurements
that checked them.


## Constraints

- **Pure code** - no runtime dependencies, nothing fetched or read from disk; the weights are a generated module
- **Small and fast** - a phone should read a page in seconds, in a Web Worker
- **Few development dependencies** - the package's tools are javascript-build's; the training's are PyTorch, numpy,
  and Pillow
- **Trained on what it renders** - no scanned or photographed data sets; pinned fonts and texts, so a training
  reproduces
- **Simple** - one definition of each idea, shared by the training and the package


## The pipeline

~~~
page photo (RGBA)
  -> gray, at most 2400 px                        lib/image.js grayImage, limitSize
  -> skew measured and leveled                    lib/layout.js skewAngle; lib/image.js rotate
  -> text lines, in reading order                 lib/layout.js findLines
  -> each line's input (32 px high)               lib/image.js lineInput   == picoocr_train/image.py line_input
  -> class scores per 4 columns                   lib/model.js forward     == picoocr_train/model.py (run_layers)
  -> text (CTC greedy)                            lib/model.js decode
~~~

The layout is classical - there's no network for it. A page's lines are found well by geometry, and its failures are
easy to see and fix. The network does what geometry can't: read the line.


## Inference

### Layout (lib/layout.js)

1. **Binarize** - Sauvola's local threshold (k 0.1 - below the usual 0.2, for dim photos' faint strokes; R 128) over a window of 1/100 of the page's longer side (8 to 40
   pixels); its sums kept by column as the rows advance, so memory is two rows, not two integral images.
2. **Components** - 8-connected, one pass with union-find over the labels, each label's box merged into its root's.
   Components touching the image's edge (the table around the page) and specks are dropped.
3. **Skew** - the bottoms of the character-size components (0.5 to 4 median heights) projected across each candidate
   angle; the sharpest histogram (sum of squared bin counts) wins: coarse (0.5 degree steps to 15 degrees, bins a
   quarter of the median height), then fine (0.05 degree steps, 1-pixel bins), the middle of a tie. The page is rotated
   level if the skew is at least 0.1 degree, and analyzed again.
4. **Lines** - character-size components (0.5 to 4 median heights - the median is about the x-height, and every
   character is at least half of it), left to right, each joining the line whose band (the median top and bottom of
   its last 5 members - so a gently curved line holds together) it overlaps most (at least half the smaller height),
   within 1.5 band heights. Lines started twice are merged. Marks - smaller components: dots, accents, quotes, commas,
   dashes - attach, left to right, to the nearest line within 0.6 line heights (vertically, of its characters; across,
   of its characters and the marks attached so far), and the lines are merged again (a line split where a
   dash spans the gap between words is one); larger components (figures) are dropped. Then the noise: a line lower
   than 0.4 median line heights (a page's edge), or lower than 0.6 with at most two components (specks), is dropped.
5. **Reading order** - a group with a column gap (at least a median component height, no line across it) is read a
   column at a time. A group without one may have lines spanning its columns (a title, a caption): its widest lines are
   set aside one at a time (at most half) until the rest have columns; the spanning lines then divide the rest into
   bands, each read in turn. Otherwise top to bottom. A line gap of 0.75 line heights starts a paragraph.

   (Row cuts - classic XY-cuts - were the first design; a unit test showed their flaw: a paragraph break in one column
   after the other column ends cuts the page into rows, and the first column's last paragraph is read after the second
   column.)

   Top to bottom is a row at a time: the lines whose centers are within a row's first line are its, read left to right
   (the pieces of one printed line - split where a thin dash didn't survive binarization - stay in order).

### The line input (lib/image.js lineInput - the one definition shared with training)

The line's ink box, padded by 10% of its height on each side, is scaled to 32 pixels high (a tent filter as wide as the
scale - an average when shrinking, bilinear when enlarging); its contrast normalized - ink is the 1st percentile, paper
the 90th, a range of at least 16 levels - to ink 1, paper 0; and framed by 48 columns of paper on each side, padded to
a multiple of 4. The training's port (picoocr_train/image.py) computes it in the same float64 operations in the same
order, stored as float32 where the package stores a Float32Array: tools/parity.js finds the outputs identical, bit for
bit.

### The network (lib/model.js)

| Layer | Shape | Output (rows x columns) |
|---|---|---|
| conv0 3x3, ReLU, max pool 2x2 | 1 -> 14 | 16 x W/2 |
| conv1 3x3, ReLU, max pool 2x2 | 14 -> 28 | 8 x W/4 |
| conv2 3x3, ReLU, max pool 2x1 | 28 -> 56 | 4 x W/4 |
| conv3 3x3, ReLU, max pool 2x1 | 56 -> 56 | 2 x W/4 |
| flatten | 56 x 2 -> 112 | W/4 steps |
| proj 1x1, ReLU | 112 -> 112 | |
| seq0-2 kernel 3, dilations 1, 2, 4, ReLU, residual | 112 -> 112 | |
| out 1x1 | 112 -> 126 classes | (CTC's blank and 125 characters) |

186,872 parameters; 124k multiply-adds per input column. The batch norms are folded into the convolutions at export, so
inference is convolutions, ReLUs, max pools, and additions - four layer ops, listed in the manifest (the manifest is the
network; lib/model.js interprets it, as model.py run_layers does in PyTorch).

**The receptive field** is 82 columns (radius 41): conv0-conv3 give 26, the dilated convolutions add 2 x (1 + 2 + 4)
steps of 4 columns. The margin (48) exceeds the radius, so the network's zero padding at its edges never reaches the
text: a line read alone in the package sees what it saw in a training batch (whose extra width is paper too).

**Every convolution is im2col and one matrix product** (`gemm`), computed in tiles of 8 output channels by 4 columns:
32 sums in locals, each weight and column value loaded once per depth step, and a step whose 4 column values are all
zero skipped (ReLU leaves 39 to 49% of the convolutions' inputs zero, often in whole regions - 28% of column quads are
all zero; a skipped term adds exactly 0, so the result is the same). A 1x1 convolution's columns are its input.

**Weights** are 5-bit integers, packed, with a float32 scale per output channel (max |w| / 15), rounded by GPTQ given
each layer's input covariance over 512 calibration lines (rendered apart from the validation sets) - dequantized once
at load (float32 inference); biases float32. The module stores the bytes as base64: deflate barely shrinks them, and
static hosts' HTTP compression takes back most of base64's third. 172 KB, 125 KB gzipped.

**Decoding** is CTC's greedy rule (each step's best class, repeats collapsed, blanks dropped); a line's confidence is
the mean probability of its characters' steps.

### The API (lib/index.js)

`recognize(image, {onProgress, model, orient})` - synchronous (run it in a Web Worker); `loadRecognizer(module)` -
the package's model, loaded once, or another; `pageText(lines)` - lines as text. The result is `{text, lines, angle,
scale, orientation}`; each line `{text, confidence, paragraph, box}`. With `orient` (the default), the page's three
longest lines (their first 10 line heights) are read four ways - as photographed, flipped horizontally (mirrored),
flipped vertically, and both (turned 180 degrees) - and the page is read the way whose mean confidence leads the
photographed way's by at least 0.02.


## Training

### Data (picoocr_train/render.py, texts.py)

Every sample is rendered: a text line in its paragraph, degraded as a phone photo is, and cut out by the line input -
the package's own function (its port), so the network trains on exactly what it will be given.

- **Fonts** (train/data/fonts.json - 55 pinned files): the URW base35 fonts (metric clones of Times, Helvetica,
  Courier, Century Schoolbook, Palatino, Bookman, Avant Garde), Carlito and Caladea (Calibri's and Cambria's clones),
  and 20 Google Fonts families - serif, sans, mono, condensed, light, bold, italic. Four families (PT Serif, PT Sans,
  Gelasio, IBM Plex Sans) are held out for validation. Every font has a glyph for every alphabet character
  (`picoocr_train.fonts check`).
- **Texts** (train/data/books.json - 12 pinned Project Gutenberg books, PicoVox's pins): 70% prose spans (half
  typeset - curly quotes, em dashes; some capitals and title case), 15% formatted values (prices, dates, times, phone
  numbers, percentages, addresses, references, list items, emails), 15% random characters - every class seen, with no
  language to lean on. Up to 48 characters; 30% of texts are short (up to 20).
- **Rendering**: a font size of 12 to 64 pixels (log-uniform - photos' text from far to near); Raqm layout (ligatures
  on 70% of the time); the lines above and below at a pitch of 1.05 to 1.6 sizes (60% each - the line input's padding
  catches their ascenders and descenders, as on a page); underlined 4%.
- **The photo**: a tilt of up to 1 degree (50% - the residual skew), a stretch of 0.85 to 1.15 (perspective, condensed
  and expanded faces); paper 140 to 255 and ink at least 70 darker; a lighting gradient of up to 40% across the canvas;
  a Gaussian blur of up to 5% of the size (70%); sensor noise up to 10 levels; JPEG at quality 25 to 90 (40%).
- **The ink box** - pixels at least 40% covered (as binarization finds them), each edge jittered by up to 5% of the
  height (the package's box is an estimate).

### The recipe (train/data/train.json) and the run (picoocr_train/train.py)

CTC loss; AdamW (learning rate 0.002, weight decay 0.01), a 1000-step linear warmup and a cosine decay to 1%; batches of
32 bucketed by width (each worker renders 8 batches' lines, sorts them, and shuffles the batches); gradient norm clipped
at 1. Every 2000 steps the character error rate (greedy) is measured on two fixed validation sets of 512 lines - seen
fonts and unseen fonts - and best.pt kept by the unseen error.

### Export (picoocr_train/export.py)

The batch norms are folded, the weights quantized; the quantized network - run as the package runs it - is validated
(the export fails if quantization costs more than 0.002 character error rate); the module is written; and the parity
data (line inputs and their quantized scores, a page's line inputs) is checked against the package (`make parity`:
tools/parity.js).


## Limits chosen ahead of time

| Limit | Value | Why |
|---|---|---|
| Line height | 32 px | With the 10% pad, an em is about 28 px - x-height about 13 px, enough for punctuation; every doubling costs twice the compute |
| Stride | 4 columns | CTC needs a step per character plus a blank between repeats. Measured over every font at the line input's scale: prose has 2.2 to 2.8 steps per needed step; the worst - Nimbus Sans Narrow on "illicit lilt: ill-fill! I\|l;i,j." - 1.34 (1.14 squeezed by the 0.85 stretch). Stride 8 would fail; stride 2 doubles the sequence's cost for nothing |
| Margin | 48 columns | The receptive field's radius is 41; a multiple of the stride |
| Page side | 2400 px | A phone photo of a letter page at this size puts body text at about the line height - no detail lost to the line input, no wasted pixels |
| Network | 187k parameters, 124k MAC/column | Chosen by a width sweep (Network size, below): the narrowest width as accurate as the widest trained |
| Weights | 5 bits, GPTQ | The smallest width that's lossless against float, end to end (Quantization, below) |
| Text length | 48 characters | Lines are read by a convolutional network - it doesn't learn a length; page lines (60 to 90 characters) read as well as training lines |
| Steps | 40,000 (1.3M lines) | The pilot's loss and error fell fastest in the first 2000 steps; 40,000 at about 0.27 s a step is 3 hours on 6 CPU cores |


## Verification

- **Parity** (tools/parity.js): the package's line inputs equal the training's port bit for bit; its class scores
  match PyTorch's (the quantized network, as run_layers runs the manifest) within 3e-5, and the decoded texts are the
  same.
- **End to end** (picoocr_train.pages, tools/pagecheck.js): synthetic page photos - a heading and paragraphs of book
  prose in an unseen font, one or two columns, tilted up to 4 degrees on a darker table, photographed (the training's
  degradations) - read by the package as an application would. The character error rate counts the words and their
  order (whitespace runs as single spaces).

### Results

The trained recognizer (the 0.875 width - Network size, below; best.pt, by the unseen validation, step 38,000 of
40,000), 5-bit GPTQ:

| | Seen fonts | Unseen fonts |
|---|---|---|
| Validation lines (512 each - 15% random characters, heavily degraded) | 1.31% (float 1.27%) | 0.96% (float 0.94%) |

| Test pages (the package reading whole page photos, unseen fonts) | Pages | Lines | Character error rate |
|---|---|---|---|
| Fresh (seed 4 - unused until the binarization's sensitivity was chosen) | 12 | 642 | 0.04% |
| Held out (seed 2 - 20 to 40 px text) | 12 | 564 | 0.04% |
| Development (seed 3) | 8 | 414 | 0.06% |
| The first set (seed 1) | 6 | 310 | 0.14% |

The fresh pages' 0.04% is the honest number - the others have guided the layout's fixes:

- **Marks attach left to right** - the held-out pages' evaluation (0.13%) found a closing quote, two marks beyond its
  short line's characters, attached to the line below's box. Each mark now extends its line's reach across (a dash
  brings the quote after it in reach): 0.04%.
- **Binarization's sensitivity, 0.1** - the first set's page 5 (20 px text, ink only 60 levels darker than its paper,
  heavy sensor noise) read as 1,102 lines and a 16% error rate: Sauvola's usual k of 0.2 put the threshold below the
  thin strokes, breaking them into specks (a blur first made it worse - the strokes are 1 to 2 pixels wide). The
  sets' error rates by k (first / development / held out):

  | k | 0.05 | 0.075 | 0.1 | 0.125 | 0.15 | 0.2 |
  |---|---|---|---|---|---|---|
  | | 0.41 / 3.07 / 14.4% | 0.14 / 0.07 / 0.06% | 0.14 / 0.06 / 0.04% | 0.13 / 0.07 / 0.04% | 0.42 / 0.09 / 0.04% | 16.3 / 0.19 / 0.04% |

  0.1 is the plateau's middle; on the fresh pages, 0.09% at 0.2 and 0.04% at 0.1.

The training's validation curve (unseen fonts): 5.1% at step 2,000, 2.4% at 10,000, 1.7% at 20,000, 1.1% at 30,000,
0.94% at 38,000 - 2.8 hours on 6 CPU cores. The run is deterministic: the sweep's run of this width and the final one
matched step for step.

### Quantization

Quantization must be lossless against float. Evaluated on the 242k network (Network size, below), on the 1,024
validation lines, by nearest rounding and by GPTQ, each against float (1.29% / 0.95%):

| Bits | Weights | Nearest | GPTQ |
|---|---|---|---|
| 8 | 234 KB | +0.02 / -0.01 | +0.01 / -0.01 |
| 7 | 205 KB | +0.03 / -0.03 | +0.01 / -0.03 |
| 6 | 176 KB | +0.02 / -0.03 | -0.02 / -0.03 |
| 5 | 146 KB | +0.02 / +0.05 | +0.00 / -0.07 |
| 4 | 117 KB | +0.51 / +0.28 | +0.11 / +0.01 |

(percentage points, seen / unseen; the sets' noise is about +-0.05). End to end - the 26 test pages, 1,278 lines, read
by the package with each model and with the float export (tools/pagecheck.js --reference): 8-bit nearest - the same
error rates, 1 line read differently; 5-bit GPTQ - the same error rates (0.13%, 0.21%, 0.00%), 3 lines; 4-bit GPTQ -
0.16% on the held-out pages (float 0.13%), 11 lines. So 5-bit GPTQ: GPTQ adds nothing at 6 bits and up (nearest is
lossless there), and is what keeps 5 bits lossless.

### Network size

The first size was set by a 2000-step pilot of two sizes (483k parameters and 261k MAC/column: 5.24% / 4.36%, seen /
unseen fonts; 242k and 160k: 5.35% / 4.47% - 2% worse for 39% less compute and half the size). Then a sweep of the 242k
network's width - every layer's channels scaled by one factor, each trained the full 40,000 steps on the same recipe,
each quantized by GPTQ at 4, 5, and 6 bits and checked against its float export (the validation lines and the 26 test
pages); the speed on the development page 0, idle (Speed, below):

| Width | Parameters | Float: seen / unseen | 5-bit GPTQ: seen / unseen | 5-bit pages: first / dev / held out | 5-bit module | The page (JIT) | 8 lines (no JIT) |
|---|---|---|---|---|---|---|---|
| 1 (16, 32, 64, 64; 128) | 242k | 1.29% / 0.95% | 1.29% / 0.88% | 0.00% / 0.20% / 0.04% | 218 KB | 3.59 s | 22.1 s |
| 0.875 (14, 28, 56, 56; 112) | 187k | 1.29% / 0.95% | 1.35% / 0.97% | 0.00% / 0.20% / 0.04% | 172 KB | 2.95 s | 19.1 s |
| 0.75 (12, 24, 48, 48; 96) | 139k | 1.61% / 1.19% | 1.64% / 1.21% | 0.01% / 0.22% / 0.05% | 131 KB | 2.18 s | 15.2 s |
| 0.5 (8, 16, 32, 32; 64) | 65k | 2.14% / 1.69% | 2.21% / 1.55% | 0.07% / 0.22% / 0.10% | 67 KB | 1.23 s | 10.5 s |

The error is flat to 0.875 and rises below it (a quarter more at 0.75, two thirds more at 0.5), so 0.875: the widest
network's accuracy (one run each - equal within the runs' noise) at 21% less size and 1.2x the speed (less than the
compute's 1.3x - the page's layout costs the same at every width). 5 bits is every width's floor: 6 bits gains nothing
measurable; 4 bits fails the export's gate at 0.75 and 0.5 and costs 0.12 to 0.14 points at 1 and 0.875 (31 KB saved
at 0.875). The sweet spot may lie between 0.875 and 0.75 - another run's worth for perhaps a tenth more speed.

### Speed

Inference on the development page 0 (1965 x 2400 analyzed, 32 lines), Node 24 on the development machine (6 cores,
one used), idle:

| | Before the optimizations (242k) | After (242k) | The 0.875 width (187k) |
|---|---|---|---|
| With the JIT (the whole page) | 3.5 s | 3.4 s | 2.9 s |
| Without a JIT (node --jitless - an iPhone in Lockdown Mode, say; 8 lines) | 31.2 s | 22.1 s (1.41x) | 19.6 s |

Without a JIT, by stage: the skew 4.8x (measured at half size), the rotation 1.8x and the layout 1.5x (no Math calls in
per-pixel loops, no per-neighbor function calls), and the convolutions 1.3 to 1.5x (the zero-skipping 8 x 4 tiles).
Every change but the half-size skew is exact - the page's text the same; the half-size skew measures as accurately
(at most 0.043 degrees from the truth on 14 pages, at full size 0.037). Tried and dropped: Float64Array weights or
activations (no gain without a JIT; 2.5x slower with one), skipping single zero inputs (1.4x faster without a JIT, 1.6x
slower with one), and the layout at half size (the error rate 0.18% to 0.42%). Not yet: Winograd convolutions (about
half the 3x3 convolutions' multiplies, at some complexity and slightly different arithmetic), reusing buffers (the
garbage collector's 4%).


## Not yet

- **Curved pages** - a book's pages bend near the spine; the line grouping follows a curve, but the line input crops a
  rectangle (a tall one for a curved line). A per-line baseline fit and a dewarp would fix it.
- **A text-line detector network** - for pages geometry can't segment (photos with text over pictures, tables)
- **A language model** - CTC beam search with a small word list would fix the common confusions (0/O, 1/l/I, rn/m)
- **Inverted text, rotated pages** (90 and 180 degrees - EXIF orientation is applied by the browser), and scripts
  beyond Latin
