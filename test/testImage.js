// Licensed under the MIT License
// https://github.com/craigahobbs/picoocr/blob/main/LICENSE

import {
    LINE_HEIGHT, LINE_MARGIN, LINE_STRIDE, crop, flip, grayImage, halve, limitSize, lineInput, resampleWeights, resize, rotate
} from '../lib/image.js';
import {strict as assert} from 'node:assert';
import test from 'node:test';


// A gray image of rows of values
function image(values) {
    return {'width': values[0].length, 'height': values.length, 'data': Float32Array.from(values.flat())};
}


// A gray image's rows (rounded to 4 places)
function rows(gray) {
    const result = [];
    for (let row = 0; row < gray.height; row++) {
        result.push(Array.from(gray.data.slice(row * gray.width, (row + 1) * gray.width), (value) => Math.round(value * 1e4) / 1e4));
    }
    return result;
}


test('grayImage', () => {
    const rgba = {'width': 2, 'height': 1, 'data': new Uint8ClampedArray([255, 255, 255, 255, 100, 0, 0, 255])};
    assert.deepEqual(rows(grayImage(rgba)), [[255, 29.9]]);
});


test('resampleWeights', () => {
    // The same size - the identity
    const same = resampleWeights(3, 3);
    assert.deepEqual(Array.from(same.starts), [0, 0, 1]);
    assert.deepEqual(Array.from(same.counts), [2, 3, 2]);
    assert.deepEqual(Array.from(same.weights.slice(same.maxCount, same.maxCount + 3)), [0, 1, 0]);

    // Halved - each destination pixel a tent-weighted average of the source pixels within the scale (2) of its center
    const half = resampleWeights(4, 2);
    assert.equal(half.maxCount, 6);
    assert.deepEqual(Array.from(half.starts), [0, 1]);
    assert.deepEqual(Array.from(half.counts), [3, 3]);
    assert.deepEqual(Array.from(half.weights.slice(0, 3), (weight) => Math.round(weight * 7 * 1e4) / 1e4), [3, 3, 1]);
});


test('resize', () => {
    // The same size
    assert.deepEqual(rows(resize(image([[1, 2, 3], [4, 5, 6]]), 3, 2)), [[1, 2, 3], [4, 5, 6]]);

    // Smaller
    assert.deepEqual(rows(resize(image([[0, 0, 100, 100], [0, 0, 100, 100]]), 2, 1)), [[14.2857, 85.7143]]);

    // Larger
    assert.deepEqual(rows(resize(image([[0, 100]]), 4, 1)), [[0, 25, 75, 100]]);
});


test('limitSize', () => {
    const small = image([[1, 2], [3, 4]]);
    assert.equal(limitSize(small, 2), small);
    const limited = limitSize(image([[0, 0, 100, 100]]), 2);
    assert.equal(limited.width, 2);
    assert.equal(limited.height, 1);
});


test('rotate', () => {
    const gray = image([[1, 2, 3], [4, 5, 6]]);
    assert.deepEqual(rows(rotate(gray, 0)), [[1, 2, 3], [4, 5, 6]]);
    assert.deepEqual(rows(rotate(gray, Math.PI)), [[6, 5, 4], [3, 2, 1]]);

    // A small angle - interpolated, the edges extended
    assert.deepEqual(rows(rotate(image([[0, 0, 0], [0, 90, 0], [0, 0, 0]]), 0.1))[1], [0.4047, 90, 0.4047]);
});


test('crop', () => {
    const gray = image([[1, 2], [3, 4]]);
    assert.deepEqual(rows(crop(gray, 1, 0, 1, 2)), [[2], [4]]);
    assert.deepEqual(rows(crop(gray, -1, -1, 4, 3)), [[1, 1, 2, 2], [1, 1, 2, 2], [3, 3, 4, 4]]);
});


test('lineInput', () => {
    // A page with a dark bar on light paper
    const width = 100;
    const height = 40;
    const data = new Float32Array(width * height).fill(200);
    for (let row = 15; row < 25; row++) {
        data.fill(20, row * width + 10, row * width + 90);
    }
    const line = lineInput({width, height, data}, {'x': 10, 'y': 15, 'width': 80, 'height': 10});
    assert.equal(line.height, LINE_HEIGHT);
    assert.equal(line.width % LINE_STRIDE, 0);

    // The 1-pixel pad scales the 12-pixel crop to 32 rows - 82 columns scale to 219, plus the margins (padded to 316)
    assert.equal(line.width, 316);
    assert.equal(line.data[16 * line.width + LINE_MARGIN + 100], 1);
    assert.equal(line.data[LINE_MARGIN + 100], 0);
    assert.equal(line.data[16 * line.width], 0);
    assert.equal(line.data[16 * line.width + line.width - 1], 0);

    // A blank line - no contrast to stretch
    const flat = {width, height, 'data': new Float32Array(width * height).fill(128)};
    const blank = lineInput(flat, {'x': 0, 'y': 0, 'width': 10, 'height': 10});
    assert.ok(blank.data.every((value) => value === 0));
});


test('flip', () => {
    const gray = image([[1, 2, 3], [4, 5, 6]]);
    assert.deepEqual(rows(flip(gray, false, false)), [[1, 2, 3], [4, 5, 6]]);
    assert.deepEqual(rows(flip(gray, true, false)), [[3, 2, 1], [6, 5, 4]]);
    assert.deepEqual(rows(flip(gray, false, true)), [[4, 5, 6], [1, 2, 3]]);
    assert.deepEqual(rows(flip(gray, true, true)), [[6, 5, 4], [3, 2, 1]]);
});


test('halve', () => {
    assert.deepEqual(rows(halve(image([[1, 3, 5, 7, 9], [5, 7, 9, 11, 13], [0, 0, 0, 0, 0]]))), [[4, 8]]);
});
