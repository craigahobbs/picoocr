// Licensed under the MIT License
// https://github.com/craigahobbs/picoocr/blob/main/LICENSE

// The tests' shared helpers

import {Buffer} from 'node:buffer';


// A deterministic pseudo-random number generator (0 to 1)
export function random(seed) {
    let state = seed;
    return () => {
        state = (state * 1103515245 + 12345) % 2147483648;
        return state / 2147483648;
    };
}


// Integers packed as picoocr_train/quant.py packs them - unsigned (q + limit), bits bits each, little-endian bit by bit
export function packIntegers(integers, bits) {
    const limit = (1 << (bits - 1)) - 1;
    const bytes = new Uint8Array(Math.ceil((integers.length * bits) / 8));
    let bit = 0;
    for (const integer of integers) {
        const unsigned = integer + limit;
        for (let bitIndex = 0; bitIndex < bits; bitIndex++, bit++) {
            bytes[bit >> 3] |= ((unsigned >> bitIndex) & 1) << (bit & 7);
        }
    }
    return bytes;
}


// A model module of tensors {name: {shape}} - each weight random integers of bits bits with a scale per output channel,
// packed, and each bias float32 - as picoocr_train.export writes it; and its integers and scales {name: {integers,
// scales}} (the test's - the loader ignores them)
export function makeModule(layers, tensors, alphabet = ['a', 'b'], seed = 1, bits = 5) {
    const next = random(seed);
    const limit = (1 << (bits - 1)) - 1;
    const parts = [];
    const entries = [];
    const values = {};
    let offset = 0;
    const append = (bytes) => {
        const padding = (4 - (offset % 4)) % 4;
        parts.push(new Uint8Array(padding), bytes);
        offset += padding;
        const start = offset;
        offset += bytes.length;
        return start;
    };
    for (const [name, {shape}] of Object.entries(tensors)) {
        const size = shape.reduce((product, dim) => product * dim, 1);
        const integers = Array.from({'length': size}, () => Math.floor(next() * (2 * limit + 1)) - limit);
        const scales = Float32Array.from({'length': shape[0]}, () => 0.01 + next() * 0.02);
        const bias = Float32Array.from({'length': shape[0]}, () => next() - 0.5);
        const weightOffset = append(packIntegers(integers, bits));
        const scaleOffset = append(new Uint8Array(scales.buffer));
        entries.push({'name': `${name}.weight`, shape, 'dtype': 'packed', bits, 'offset': weightOffset, scaleOffset});
        entries.push({'name': `${name}.bias`, 'shape': [shape[0]], 'dtype': 'float32', 'offset': append(new Uint8Array(bias.buffer))});
        values[name] = {integers, scales};
    }
    const bytes = Buffer.concat(parts);
    const manifest = {'version': 1, alphabet, 'height': 4, 'margin': 0, 'stride': 2, layers, 'tensors': entries};
    return {manifest, 'weights': bytes.toString('base64').replace(/.{1,60}/g, '$&\n'), values};
}
