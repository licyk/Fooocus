import test from 'node:test';
import assert from 'node:assert/strict';
import splitTags from '../src/utils/splitTags.js';
import hostWeights from '../src/utils/hostWeights.js';

test('Upstream nested numeric weights stay together', () => {
    assert.deepEqual(splitTags('cat, (blue eyes, (red hair:1.2):0.8), dog'), ['cat', '(blue eyes, (red hair:1.2):0.8)', 'dog']);
});
test('Upstream LoRA syntax separates adjacent LoRAs from text', () => {
    assert.deepEqual(splitTags('cat <lora:animals/cat.safetensors:0.7><lora:style:1>'), ['cat', '<lora:animals/cat.safetensors:0.7>', '<lora:style:1>']);
});
test('Upstream scheduled bracket and LoRA block weights remain intact', () => {
    assert.deepEqual(splitTags('[cat:dog:0.5], <lora:a:1:1,0,0>'), ['[cat:dog:0.5]', '<lora:a:1:1,0,0>']);
});
test('Upstream multilingual punctuation and linebreak editing', () => {
    assert.deepEqual(splitTags('猫，狗；树\n草'), ['猫', '狗', '树', '\n', '草']);
});
test('Upstream facial expressions are protected from bracket splitting', () => {
    assert.deepEqual(splitTags('>_<, :-(, :-)'), ['>_<', ':-(', ':-)']);
});
test('Upstream BREAK and empty prompt handling', () => {
    assert.deepEqual(splitTags('cat BREAK dog'), ['cat', 'BREAK', 'dog']);
    assert.deepEqual(splitTags(''), []);
});

test('Fooocus emission converts upstream down-weight symbols and keeps numeric nested weights', () => {
    assert.equal(hostWeights('[cat], [[dog]], (bird:1.5)'), '(cat:0.909091), (dog:0.826446), (bird:1.5)');
    assert.equal(hostWeights('[cat:0.8]'), '(cat:0.727273)');
});
test('NovelAI symbol emission is opt-in and leaves scheduling syntax and escapes alone', () => {
    assert.equal(hostWeights('{cat}'), '{cat}');
    assert.equal(hostWeights('{{cat}}', true), '(cat:1.1025)');
    assert.equal(hostWeights('[cat:dog:0.5], [cat|dog], \\[dog\\], __hair/color__, <lora:a:0.8>'), '[cat:dog:0.5], [cat|dog], \\[dog\\], __hair/color__, <lora:a:0.8>');
});
