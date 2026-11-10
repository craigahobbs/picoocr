// Licensed under the MIT License
// https://github.com/craigahobbs/picoocr/blob/main/LICENSE

// The parity check - the package against the training's port and PyTorch (picoocr_train.export's parity data): the
// line inputs of a page (lib/image.js lineInput against picoocr_train/image.py - identical), and an exported model's
// class scores (lib/model.js forward against model.py run_layers - within SCORE_TOLERANCE) and decoded texts (identical).
//
//     node tools/parity.js build/train/recognizer.js build/train/parity.json

/* eslint-disable no-console */

import {decode, forward, loadModel} from '../lib/model.js';
import {lineInput} from '../lib/image.js';
import process from 'node:process';
import {readFileSync} from 'node:fs';
import {resolve} from 'node:path';


// The largest class score difference allowed (float32 arithmetic in different orders)
const SCORE_TOLERANCE = 1e-3;


// The largest absolute difference of two arrays
function maxDifference(first, second) {
    if (first.length !== second.length) {
        return Infinity;
    }
    let max = 0;
    for (let ix = 0; ix < first.length; ix++) {
        max = Math.max(max, Math.abs(first[ix] - second[ix]));
    }
    return max;
}


async function main([modulePath, parityPath]) {
    const model = loadModel(await import(resolve(modulePath)));
    const parity = JSON.parse(readFileSync(parityPath, 'utf-8'));
    let failed = false;

    // The page's line inputs
    const page = {'width': parity.page.width, 'height': parity.page.height, 'data': Float32Array.from(parity.page.data)};
    for (const {box, line} of parity.boxes) {
        const [column, row, width, height] = box;
        const difference = maxDifference(lineInput(page, {'x': column, 'y': row, width, height}).data, line);
        console.log(`lineInput ${JSON.stringify(box)}: max difference ${difference}`);
        failed ||= difference !== 0;
    }

    // The lines' class scores and texts
    for (const line of parity.lines) {
        const scores = forward(model, {'width': line.width, 'height': model.manifest.height, 'data': Float32Array.from(line.data)});
        const difference = maxDifference(scores.data, line.scores);
        const {text} = decode(model, scores);
        console.log(`forward ${JSON.stringify(text)}: max score difference ${difference.toExponential(2)}`);
        failed ||= difference > SCORE_TOLERANCE || text !== line.text;
    }
    console.log(failed ? 'parity FAILED' : 'parity OK');
    process.exitCode = failed ? 1 : 0;
}


await main(process.argv.slice(2));
