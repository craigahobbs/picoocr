// Licensed under the MIT License
// https://github.com/craigahobbs/picoocr/blob/main/LICENSE

import {conv1d, conv2d, decode, decodeBase64, forward, loadModel, maxpool, meanConfidence} from '../lib/model.js';
import {makeModule, random} from './helpers.js';
import {Buffer} from 'node:buffer';
import {strict as assert} from 'node:assert';
import test from 'node:test';


// A random activation
function activation(channels, height, width, seed = 7) {
    const next = random(seed);
    return {channels, height, width, 'data': Float32Array.from({'length': channels * height * width}, () => next() * 2 - 1)};
}


// A 2-D convolution, the plain way ('same' zero padding)
function referenceConv2d(weight, bias, shape, input, relu) {
    const [outChannels, inChannels, kernel] = shape;
    const half = (kernel - 1) / 2;
    const {height, width} = input;
    const data = new Float32Array(outChannels * height * width);
    for (let oc = 0; oc < outChannels; oc++) {
        for (let row = 0; row < height; row++) {
            for (let column = 0; column < width; column++) {
                let sum = bias[oc];
                for (let ic = 0; ic < inChannels; ic++) {
                    for (let ky = 0; ky < kernel; ky++) {
                        for (let kx = 0; kx < kernel; kx++) {
                            const inRow = row + ky - half;
                            const inColumn = column + kx - half;
                            if (inRow >= 0 && inRow < height && inColumn >= 0 && inColumn < width) {
                                const weightIndex = ((oc * inChannels + ic) * kernel + ky) * kernel + kx;
                                sum += weight[weightIndex] * input.data[(ic * height + inRow) * width + inColumn];
                            }
                        }
                    }
                }
                data[(oc * height + row) * width + column] = relu ? Math.max(0, sum) : sum;
            }
        }
    }
    return data;
}


// A 1-D convolution, the plain way ('same' zero padding at a dilation)
function referenceConv1d(weight, bias, shape, input, dilation, relu, residual) {
    const [outChannels, inChannels, kernel] = shape;
    const half = (kernel - 1) / 2;
    const {width} = input;
    const data = new Float32Array(outChannels * width);
    for (let oc = 0; oc < outChannels; oc++) {
        for (let column = 0; column < width; column++) {
            let sum = bias[oc];
            for (let ic = 0; ic < inChannels; ic++) {
                for (let tap = 0; tap < kernel; tap++) {
                    const inColumn = column + (tap - half) * dilation;
                    if (inColumn >= 0 && inColumn < width) {
                        sum += weight[(oc * inChannels + ic) * kernel + tap] * input.data[ic * width + inColumn];
                    }
                }
            }
            const value = relu ? Math.max(0, sum) : sum;
            data[oc * width + column] = residual ? value + input.data[oc * width + column] : value;
        }
    }
    return data;
}


// The largest difference of two arrays
function maxDifference(first, second) {
    assert.equal(first.length, second.length);
    return Math.max(...Array.from(first, (value, ix) => Math.abs(value - second[ix])));
}


test('decodeBase64', () => {
    assert.deepEqual(Array.from(decodeBase64('AAEC\n/w==')), [0, 1, 2, 255]);
});


test('loadModel', () => {
    // Packed weights of several widths (5 and 3 bits cross bytes) - each its integer times its output channel's scale
    for (const bits of [8, 5, 3]) {
        const module = makeModule([], {'conv': {'shape': [2, 1, 3, 3]}}, ['a'], 1, bits);
        const model = loadModel(module);
        assert.equal(model.manifest, module.manifest);
        const {integers, scales} = module.values.conv;
        const expected = integers.map((integer, ix) => Math.fround(integer * scales[Math.floor(ix / 9)]));
        assert.deepEqual(Array.from(model.tensors['conv.weight']), expected);
        assert.equal(model.tensors['conv.bias'].length, 2);
    }

    // Float32 weights
    const floats = new Float32Array([0.5, -1.25]);
    const manifest = {'tensors': [{'name': 'w', 'shape': [2], 'dtype': 'float32', 'offset': 0}]};
    const model = loadModel({manifest, 'weights': Buffer.from(floats.buffer).toString('base64')});
    assert.deepEqual(Array.from(model.tensors.w), [0.5, -1.25]);
});


test('conv2d', () => {
    const shape = [3, 2, 3, 3];
    const model = loadModel(makeModule([], {'conv': {shape}}));
    const input = activation(2, 4, 5);
    for (const relu of [false, true]) {
        const output = conv2d(model, {'name': 'conv', relu}, input);
        assert.deepEqual([output.channels, output.height, output.width], [3, 4, 5]);
        const expected = referenceConv2d(model.tensors['conv.weight'], model.tensors['conv.bias'], shape, input, relu);
        assert.ok(maxDifference(output.data, expected) < 1e-5);
    }

    // A zero weight is skipped - the same result
    model.tensors['conv.weight'][4] = 0;
    const expected = referenceConv2d(model.tensors['conv.weight'], model.tensors['conv.bias'], shape, input, false);
    assert.ok(maxDifference(conv2d(model, {'name': 'conv'}, input).data, expected) < 1e-5);

    // The wrong channels
    assert.throws(() => conv2d(model, {'name': 'conv'}, activation(3, 4, 5)), {'message': 'layer conv expects 2 channels, got 3'});

    // A missing tensor
    assert.throws(() => conv2d(model, {'name': 'other'}, input), {'message': 'model tensor other.weight not found'});
});


test('conv1d', () => {
    const shape = [3, 3, 3];
    const model = loadModel(makeModule([], {'seq': {shape}}));
    const input = activation(3, 1, 9);
    for (const [dilation, relu, residual] of [[undefined, false, false], [2, true, false], [4, true, true]]) {
        const output = conv1d(model, {'name': 'seq', dilation, relu, residual}, input);
        assert.deepEqual([output.channels, output.height, output.width], [3, 1, 9]);
        const weight = model.tensors['seq.weight'];
        const expected = referenceConv1d(weight, model.tensors['seq.bias'], shape, input, dilation ?? 1, relu, residual);
        assert.ok(maxDifference(output.data, expected) < 1e-5);
    }

    // More than one row
    assert.throws(() => conv1d(model, {'name': 'seq'}, activation(3, 2, 9)), {'message': 'layer seq expects 3 channels of one row'});
});


test('maxpool', () => {
    const input = {'channels': 1, 'height': 2, 'width': 5, 'data': Float32Array.from([1, 5, 2, 0, 9, 3, -1, 4, 8, 7])};
    assert.deepEqual(Array.from(maxpool({'size': [2, 2]}, input).data), [5, 8]);
    assert.deepEqual(Array.from(maxpool({'size': [2, 1]}, input).data), [3, 5, 4, 8, 9]);
});


test('forward', () => {
    // A small network - a convolution, a pool, the flatten, and the 1-D convolutions
    const layers = [
        {'op': 'conv2d', 'name': 'conv0', 'relu': true},
        {'op': 'maxpool', 'size': [2, 2]},
        {'op': 'flatten'},
        {'op': 'conv1d', 'name': 'seq0', 'dilation': 1, 'relu': true, 'residual': true},
        {'op': 'conv1d', 'name': 'out'}
    ];
    const tensors = {'conv0': {'shape': [2, 1, 3, 3]}, 'seq0': {'shape': [4, 4, 3]}, 'out': {'shape': [3, 4, 1]}};
    const model = loadModel(makeModule(layers, tensors));
    const line = activation(1, 4, 8);
    const scores = forward(model, {'width': line.width, 'height': line.height, 'data': line.data});
    assert.equal(scores.classes, 3);
    assert.equal(scores.steps, 4);

    // The same, layer by layer
    const conv = conv2d(model, layers[0], {...line, 'channels': 1});
    const pooled = maxpool(layers[1], conv);
    const flat = {'channels': 4, 'height': 1, 'width': 4, 'data': pooled.data};
    const expected = conv1d(model, layers[4], conv1d(model, layers[3], flat));
    assert.deepEqual(Array.from(scores.data), Array.from(expected.data));

    // An unknown layer
    const unknown = loadModel(makeModule([{'op': 'lstm'}], {}));
    assert.throws(() => forward(unknown, {'width': 4, 'height': 4, 'data': new Float32Array(16)}), {'message': 'unknown layer op lstm'});
});


test('decode', () => {
    const model = {'manifest': {'alphabet': ['a', 'b']}};

    // Scores [classes, steps] - steps' best classes a, a, blank, a, b, b, blank: "aab"
    const best = [1, 1, 0, 1, 2, 2, 0];
    const steps = best.length;
    const data = new Float32Array(3 * steps);
    for (const [step, cls] of best.entries()) {
        data[cls * steps + step] = 5;
    }
    const {text, confidence} = decode(model, {'classes': 3, steps, data});
    assert.equal(text, 'aab');
    assert.equal(Math.round(confidence * 1e6) / 1e6, Math.round((1 / (1 + 2 * Math.exp(-5))) * 1e6) / 1e6);

    // Blanks only
    const blanks = {'classes': 3, 'steps': 2, 'data': Float32Array.from([1, 1, 0, 0, 0, 0])};
    assert.deepEqual(decode(model, blanks), {'text': '', 'confidence': 1});
});


test('meanConfidence', () => {
    // Steps [classes, steps]: certain (5 against 0, 0), and even (all 0)
    const scores = {'classes': 3, 'steps': 2, 'data': Float32Array.from([5, 0, 0, 0, 0, 0])};
    const certain = 1 / (1 + 2 * Math.exp(-5));
    assert.ok(Math.abs(meanConfidence(scores) - (certain + 1 / 3) / 2) < 1e-6);
    assert.ok(Math.abs(meanConfidence(scores, 0, 1) - certain) < 1e-6);
    assert.ok(Math.abs(meanConfidence(scores, 1) - 1 / 3) < 1e-6);
    assert.equal(meanConfidence(scores, 1, 1), 0);
});
