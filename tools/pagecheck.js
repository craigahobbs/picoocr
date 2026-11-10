// Licensed under the MIT License
// https://github.com/craigahobbs/picoocr/blob/main/LICENSE

// The end-to-end check - the package reading test pages (picoocr_train.pages: page photos and their text): each page's
// character error rate (its text and the page's, whitespace runs as single spaces - the words and their order) and
// its seconds. With --reference, a second model reads each page too (the float export - quantization's end-to-end
// check): its error rate, and the lines the two read differently.
//
//     node tools/pagecheck.js build/pages [--model build/train/recognizer.js] [--reference build/train/float.js] [--show]

/* eslint-disable no-console */

import {join, resolve} from 'node:path';
import {loadRecognizer, recognize} from '../lib/index.js';
import {readFileSync, readdirSync} from 'node:fs';
import process from 'node:process';


// The edit distance of two strings (one row at a time)
export function editDistance(first, second) {
    let previous = Int32Array.from({'length': second.length + 1}, (_value, ix) => ix);
    let current = new Int32Array(second.length + 1);
    for (let ix = 1; ix <= first.length; ix++) {
        current[0] = ix;
        for (let jx = 1; jx <= second.length; jx++) {
            const substitute = previous[jx - 1] + (first[ix - 1] === second[jx - 1] ? 0 : 1);
            current[jx] = Math.min(previous[jx] + 1, current[jx - 1] + 1, substitute);
        }
        [previous, current] = [current, previous];
    }
    return previous[second.length];
}


// A text's words, single-spaced
function words(text) {
    return text.replace(/\s+/gu, ' ').trim();
}


async function main(args) {
    const [dir] = args;
    const modelIndex = args.indexOf('--model');
    const model = modelIndex === -1 ? loadRecognizer() : loadRecognizer(await import(resolve(args[modelIndex + 1])));
    const referenceIndex = args.indexOf('--reference');
    const reference = referenceIndex === -1 ? null : loadRecognizer(await import(resolve(args[referenceIndex + 1])));
    const show = args.includes('--show');
    let errors = 0;
    let characters = 0;
    let seconds = 0;
    let referenceErrors = 0;
    let lines = 0;
    let differing = 0;
    for (const file of readdirSync(dir).filter((name) => name.endsWith('.json')).sort()) {
        const page = JSON.parse(readFileSync(join(dir, file), 'utf-8'));
        const data = new Uint8ClampedArray(readFileSync(join(dir, file.replace(/\.json$/u, '.rgba'))));
        const started = performance.now();
        const result = recognize({'width': page.width, 'height': page.height, data}, {model});
        const elapsed = (performance.now() - started) / 1000;
        const truth = words(page.text);
        const distance = editDistance(words(result.text), truth);
        errors += distance;
        characters += truth.length;
        seconds += elapsed;
        console.log(
            `${file}: CER ${(distance / truth.length).toFixed(4)}, ${result.lines.length} lines, ${elapsed.toFixed(1)} s ` +
            `(${page.font} ${page.size}px, ${page.columns} column(s), angle ${page.angle.toFixed(2)} measured ` +
            `${((-result.angle * 180) / Math.PI).toFixed(2)})`
        );
        if (reference !== null) {
            const referenceResult = recognize({'width': page.width, 'height': page.height, data}, {'model': reference});
            const referenceDistance = editDistance(words(referenceResult.text), truth);
            const different = result.lines.filter((line, ix) => line.text !== referenceResult.lines[ix]?.text);
            referenceErrors += referenceDistance;
            lines += result.lines.length;
            differing += different.length + Math.max(0, referenceResult.lines.length - result.lines.length);
            console.log(`    reference CER ${(referenceDistance / truth.length).toFixed(4)}, ${different.length} line(s) read differently`);
            for (const line of different) {
                console.log(`        ${JSON.stringify(line.text)}`);
            }
        }
        if (show) {
            console.log(result.text);
        }
    }
    console.log(`total CER ${(errors / characters).toFixed(4)}, ${seconds.toFixed(1)} s`);
    if (reference !== null) {
        console.log(`reference CER ${(referenceErrors / characters).toFixed(4)}, ${differing} of ${lines} lines read differently`);
    }
}


await main(process.argv.slice(2));
