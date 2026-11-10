// Licensed under the MIT License
// https://github.com/craigahobbs/picoocr/blob/main/LICENSE

// The inference benchmark - a test page (picoocr_train.pages) read as lib/index.js recognize reads it, each stage
// timed: the gray page and its leveling, the layout, the line inputs, and each network layer (summed over the lines).
// The best of --runs runs is reported (other work on the machine only slows a run), and the page's text's hash (an
// optimization must leave it unchanged). Run it under node --jitless for an engine without a JIT (an iPhone in
// Lockdown Mode, say), and with --cpu-prof for a profile.
//
//     node --jitless tools/bench.js build/pages-dev/page0 [--lines N] [--runs N] [--model build/train/recognizer.js]

/* eslint-disable no-console */

import {MAX_PAGE_SIDE, loadRecognizer} from '../lib/index.js';
import {binarize, findLines, skewAngle, textComponents} from '../lib/layout.js';
import {conv1d, conv2d, decode, maxpool} from '../lib/model.js';
import {grayImage, halve, limitSize, lineInput, rotate} from '../lib/image.js';
import {createHash} from 'node:crypto';
import process from 'node:process';
import {readFileSync} from 'node:fs';
import {resolve} from 'node:path';


// The value of a command-line option (a number), or a default
function option(args, name, fallback) {
    const index = args.indexOf(name);
    return index === -1 ? fallback : Number(args[index + 1]);
}


// Time a function - [its result, milliseconds]
function timed(times, name, callback) {
    const started = performance.now();
    const result = callback();
    times[name] = (times[name] ?? 0) + performance.now() - started;
    return result;
}


// Run a network layer on an activation (as lib/model.js forward does)
function runLayer(model, layer, activation) {
    if (layer.op === 'conv2d') {
        return conv2d(model, layer, activation);
    } else if (layer.op === 'maxpool') {
        return maxpool(layer, activation);
    } else if (layer.op === 'flatten') {
        const {channels, height, width, data} = activation;
        return {'channels': channels * height, 'height': 1, width, data};
    }
    return conv1d(model, layer, activation);
}


// Read the page once - {times, text}
function run(model, image, maxLines) {
    const times = {};
    let page = timed(times, 'gray', () => limitSize(grayImage(image), MAX_PAGE_SIDE));
    const angle = timed(times, 'skew', () => {
        const measured = Math.max(page.width, page.height) > 1200 ? halve(page) : page;
        const {boxes, medianHeight} = textComponents(binarize(measured), measured.width, measured.height);
        return skewAngle(boxes, medianHeight);
    });
    page = timed(times, 'rotate', () => rotate(page, angle));
    const {lines} = timed(times, 'layout', () => findLines(page));
    const texts = [];
    for (const box of lines.slice(0, maxLines)) {
        const line = timed(times, 'lineInput', () => lineInput(page, box));
        let activation = {'channels': 1, 'height': line.height, 'width': line.width, 'data': line.data};
        for (const layer of model.manifest.layers) {
            activation = timed(times, layer.name ?? layer.op, runLayer.bind(null, model, layer, activation));
        }
        const scores = {'classes': activation.channels, 'steps': activation.width, 'data': activation.data};
        texts.push(timed(times, 'decode', () => decode(model, scores).text));
    }
    return {times, 'text': texts.join('\n')};
}


async function main(args) {
    const [pagePath] = args;
    const maxLines = option(args, '--lines', Infinity);
    const runs = option(args, '--runs', 3);
    const {width, height} = JSON.parse(readFileSync(`${pagePath}.json`, 'utf-8'));
    const image = {width, height, 'data': new Uint8ClampedArray(readFileSync(`${pagePath}.rgba`))};
    const modelIndex = args.indexOf('--model');
    const model = modelIndex === -1 ? loadRecognizer() : loadRecognizer(await import(resolve(args[modelIndex + 1])));
    let best = null;
    let text = null;
    for (let ix = 0; ix < runs; ix++) {
        const result = run(model, image, maxLines);
        ({text} = result);
        const fastest = (times) => Object.fromEntries(Object.entries(times).map(([name, ms]) => [name, Math.min(ms, result.times[name])]));
        best = best === null ? result.times : fastest(best);
    }
    const total = Object.values(best).reduce((sum, ms) => sum + ms, 0);
    for (const [name, ms] of Object.entries(best)) {
        console.log(`${name.padEnd(10)} ${ms.toFixed(0).padStart(8)} ms  ${((100 * ms) / total).toFixed(1).padStart(5)}%`);
    }
    console.log(`${'total'.padEnd(10)} ${total.toFixed(0).padStart(8)} ms   (${process.execArgv.join(' ') || 'jit'})`);
    console.log(`text sha256 ${createHash('sha256').update(text).digest('hex').slice(0, 16)}`);
}


await main(process.argv.slice(2));
