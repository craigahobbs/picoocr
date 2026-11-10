// Licensed under the MIT License
// https://github.com/craigahobbs/picoocr/blob/main/LICENSE

import {binarize, components, findLines, groupLines, median, readingOrder, skewAngle, textComponents} from '../lib/layout.js';
import {strict as assert} from 'node:assert';
import test from 'node:test';


// A gray page (paper 230) with dark boxes (ink 30) - each [x, y, width, height]
export function page(width, height, rects) {
    const data = new Float32Array(width * height).fill(230);
    for (const [left, top, rectWidth, rectHeight] of rects) {
        for (let row = top; row < top + rectHeight; row++) {
            data.fill(30, row * width + left, row * width + left + rectWidth);
        }
    }
    return {width, height, data};
}


// A component box
function box(x0, y0, width, height) {
    return {x0, y0, 'x1': x0 + width, 'y1': y0 + height, 'area': width * height};
}


// A line of character boxes - count of them, from x, at a top and height, a step apart
function characters(count, left, top, height, step = 15) {
    return Array.from({'length': count}, (_value, ix) => [left + ix * step, top, 10, height]);
}


test('binarize', () => {
    const gray = page(60, 60, [[20, 20, 10, 10]]);
    const binary = binarize(gray);
    assert.equal(binary[25 * 60 + 25], 1);
    assert.equal(binary[5 * 60 + 5], 0);
    assert.equal(binary[25 * 60 + 35], 0);
    assert.equal(binary.reduce((sum, value) => sum + value, 0), 100);

    // A wide page - the window's radius scales with the page
    assert.equal(binarize(page(1200, 20, [[600, 5, 10, 10]])).reduce((sum, value) => sum + value, 0), 100);
});


test('components', () => {
    // A U shape (its arms meet only at the bottom - their labels joined), a diagonal pair, and a dot
    const width = 10;
    const pixels = [
        '#.#.......',
        '#.#...#...',
        '###....#..',
        '........#.'
    ];
    const binary = Uint8Array.from(pixels.join(''), (char) => (char === '#' ? 1 : 0));
    const boxes = components(binary, width, pixels.length).sort((box0, box1) => box0.x0 - box1.x0);
    assert.deepEqual(boxes, [
        {'x0': 0, 'y0': 0, 'x1': 3, 'y1': 3, 'area': 7},
        {'x0': 6, 'y0': 1, 'x1': 9, 'y1': 4, 'area': 3}
    ]);

    // Labels joined right to left (a later label's root the smaller)
    const join = ['..#', '.##', '##.'];
    const joined = components(Uint8Array.from(join.join(''), (char) => (char === '#' ? 1 : 0)), 3, 3);
    assert.deepEqual(joined, [{'x0': 0, 'y0': 0, 'x1': 3, 'y1': 3, 'area': 5}]);

    // Labels joined left to right (the first label's root the smaller)
    const join2 = ['#.#', '###'];
    assert.deepEqual(components(Uint8Array.from(join2.join(''), (char) => (char === '#' ? 1 : 0)), 3, 2), [
        {'x0': 0, 'y0': 0, 'x1': 3, 'y1': 2, 'area': 5}
    ]);

    // Three labels found to be one - a merged label's box merged into its root's (the later labels' after)
    const comb = ['#.#.#', '#####'];
    assert.deepEqual(components(Uint8Array.from(comb.join(''), (char) => (char === '#' ? 1 : 0)), 5, 2), [
        {'x0': 0, 'y0': 0, 'x1': 5, 'y1': 2, 'area': 8}
    ]);
});


test('median', () => {
    assert.equal(median([]), 0);
    assert.equal(median([3, 1, 2]), 2);
    assert.equal(median([4, 1, 3, 2]), 2);
});


test('textComponents', () => {
    // A character, a speck (too small), and a box at the page's edge (the surroundings)
    const gray = page(80, 60, [[20, 20, 6, 12], [40, 20, 1, 2], [0, 40, 10, 10]]);
    const {boxes, medianHeight} = textComponents(binarize(gray), 80, 60);
    assert.deepEqual(boxes, [{'x0': 20, 'y0': 20, 'x1': 26, 'y1': 32, 'area': 72}]);
    assert.equal(medianHeight, 12);
});


test('skewAngle', () => {
    // Three lines of characters whose bottoms descend 2 degrees to the right
    const slope = Math.tan((2 * Math.PI) / 180);
    const boxes = [];
    for (const lineTop of [100, 160, 220]) {
        for (let ix = 0; ix < 40; ix++) {
            const left = 20 + ix * 15;
            const top = Math.round(lineTop + left * slope);
            boxes.push(box(left, top, 10, 20));
        }
    }
    const angle = (skewAngle(boxes, 20) * 180) / Math.PI;
    assert.ok(Math.abs(angle - 2) < 0.1, `angle ${angle}`);

    // Too few components
    assert.equal(skewAngle([box(10, 10, 5, 20)], 20), 0);
});


test('groupLines', () => {
    const boxes = [
        // Two lines of characters
        ...characters(10, 20, 100, 20).map(([left, top, width, height]) => box(left, top, width, height)),
        ...characters(8, 20, 150, 20).map(([left, top, width, height]) => box(left, top, width, height)),

        // A dot above the first line, a comma below its end, and a mark far from any line
        box(22, 92, 4, 4),
        box(160, 118, 3, 6),
        box(500, 400, 3, 3),

        // A figure (too tall)
        box(300, 20, 100, 200)
    ];
    const lines = groupLines(boxes, 20).sort((line0, line1) => line0.y0 - line1.y0);
    assert.equal(lines.length, 2);
    assert.deepEqual([lines[0].x0, lines[0].y0, lines[0].x1, lines[0].y1], [20, 92, 165, 124]);
    assert.equal(lines[0].members.length, 10);
    assert.equal(lines[0].marks.length, 2);
    assert.deepEqual([lines[1].x0, lines[1].y0, lines[1].x1, lines[1].y1], [20, 150, 135, 170]);

    // A line split at a long mark (an em dash between words) - one line, the dash attached
    const dashed = [
        ...characters(4, 20, 100, 20).map(([left, top, width, height]) => box(left, top, width, height)),
        box(78, 108, 20, 3),
        ...characters(4, 125, 100, 20).map(([left, top, width, height]) => box(left, top, width, height))
    ];
    const joined = groupLines(dashed, 20);
    assert.equal(joined.length, 1);
    assert.deepEqual([joined[0].x0, joined[0].x1, joined[0].members.length, joined[0].marks.length], [20, 180, 8, 1]);

    // A short line ending in a dash and a closing quote (two marks, the second beyond the line's characters' reach) -
    // the marks chain to it, not to the long line above
    const chained = groupLines([
        ...characters(20, 20, 100, 20).map(([left, top, width, height]) => box(left, top, width, height)),
        ...characters(3, 20, 128, 20).map(([left, top, width, height]) => box(left, top, width, height)),
        box(64, 135, 18, 3), box(84, 128, 3, 6), box(92, 128, 3, 6)
    ], 20).sort((line0, line1) => line0.y0 - line1.y0);
    assert.deepEqual(chained.map((line) => [line.y0, line.y1, line.x1, line.marks.length]), [[100, 120, 315, 0], [128, 148, 95, 3]]);

    // Noise: specks (two small components alone), and a low line of fragments (a page's edge)
    const noisy = [
        ...characters(10, 20, 100, 20).map(([left, top, width, height]) => box(left, top, width, height)),
        ...characters(10, 20, 150, 20).map(([left, top, width, height]) => box(left, top, width, height)),
        box(400, 300, 4, 11), box(410, 300, 4, 11),
        ...characters(20, 20, 400, 7).map(([left, top, width, height]) => box(left, top, width, height))
    ];
    assert.deepEqual(groupLines(noisy, 20).map((line) => line.y0).sort((y0, y1) => y0 - y1), [100, 150]);

    // Only lines of few components - the median of them all
    assert.equal(groupLines([box(20, 100, 10, 20), box(35, 100, 10, 20)], 20).length, 1);

    // A line started twice (a character overlapping the band too little), merged
    const twice = [box(20, 100, 10, 20), box(35, 111, 10, 20), box(50, 105, 10, 20), box(65, 105, 10, 20)];
    const merged = groupLines(twice, 20);
    assert.equal(merged.length, 1);
    assert.deepEqual([merged[0].x0, merged[0].y0, merged[0].x1, merged[0].y1], [20, 100, 75, 131]);
    assert.deepEqual(merged[0].members.map((member) => member.x0), [20, 35, 50, 65]);
});


test('readingOrder', () => {
    // A title across two columns, the left column with a paragraph break
    const line = (x0, y0, x1, y1) => ({x0, y0, x1, y1});
    const title = line(100, 50, 900, 90);
    const left = [line(100, 200, 480, 230), line(100, 120, 480, 150), line(100, 160, 480, 190), line(100, 260, 480, 290)];
    const right = [line(520, 125, 900, 155), line(520, 165, 900, 195)];
    const ordered = readingOrder([...right, title, ...left], 20);
    assert.deepEqual(ordered.map((ordered_) => [ordered_.x0, ordered_.y0]), [
        [100, 50], [100, 120], [100, 160], [100, 200], [100, 260], [520, 125], [520, 165]
    ]);
    assert.deepEqual(ordered.map((ordered_) => ordered_.paragraph), [true, true, false, false, true, true, false]);

    // Two pieces of a row (no column gap between them, the right one taller) - read left to right
    const row = readingOrder([line(60, 8, 100, 32), line(10, 10, 55, 30), line(10, 40, 100, 60)], 20);
    assert.deepEqual(row.map((ordered_) => [ordered_.x0, ordered_.paragraph]), [[10, true], [60, false], [10, false]]);

        // One line; two lines at the same height without a column gap between them (left first)
    assert.deepEqual(readingOrder([line(10, 10, 100, 30)], 20).map((ordered_) => ordered_.x0), [10]);
    assert.deepEqual(readingOrder([line(60, 10, 100, 30), line(10, 10, 55, 30)], 20).map((ordered_) => ordered_.x0), [10, 60]);

    // No lines
    assert.deepEqual(readingOrder([], 20), []);
});


test('findLines', () => {
    // Two lines of "characters" on a page
    const gray = page(400, 200, [...characters(12, 30, 50, 20), ...characters(10, 30, 100, 20)]);
    const {medianHeight, lines} = findLines(gray);
    assert.equal(medianHeight, 20);
    assert.deepEqual(lines, [
        {'x': 30, 'y': 50, 'width': 175, 'height': 20, 'paragraph': true},
        {'x': 30, 'y': 100, 'width': 145, 'height': 20, 'paragraph': true}
    ]);
});
