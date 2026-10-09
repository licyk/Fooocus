const test = require('node:test');
const assert = require('node:assert/strict');
const fs = require('node:fs');
const vm = require('node:vm');
const path = require('node:path');

const context = vm.createContext({window: {localization: {
    'Ready: {count} entries.': '已就绪：{count} 条。',
    'Literal': '中文',
    'Run history ({count})': '运行历史（{count}）',
    'Cannot load {name}': '无法加载 {name}',
}}, onUiLoaded() {}});
vm.runInContext(fs.readFileSync(path.join(__dirname, '../javascript/localization.js'), 'utf8'), context);
const {t} = context.window.FooocusI18n;

test('dynamic messages support arbitrary counts and zero', () => {
    assert.equal(t('Ready: {count} entries.', {count: '141,079'}), '已就绪：141,079 条。');
    assert.equal(t('Ready: {count} entries.', {count: 0}), '已就绪：0 条。');
});
test('missing translations fall back to formatted English', () => {
    assert.equal(t('Unknown: {name}', {name: 'model.safetensors'}), 'Unknown: model.safetensors');
    assert.equal(t('Literal'), '中文');
    assert.equal(t('Ready: {count} entries.'), '已就绪：{count} 条。');
});
test('parameters stay literal, with no recursive substitution or JSON rewriting', () => {
    assert.equal(t('Unknown: {name}', {name: '$& {count}', count: 5}), 'Unknown: $& {count}');
    assert.equal(t('{"name":"cat"}', {name: '狗'}), '{"name":"cat"}');
    assert.equal(t('Unknown: {count}', Object.create({count: 9})), 'Unknown: {count}');
});
test('third-party numeric labels support escaped punctuation and grouped counts', () => {
    assert.equal(t('Run history (12345)'), '运行历史（12345）');
    assert.equal(t('Run history (12,345)'), '运行历史（12,345）');
    assert.equal(t('Run history (12\u202f345)'), '运行历史（12\u202f345）');
    assert.equal(t('Run history (0)'), '运行历史（0）');
});
test('automatic template matching excludes arbitrary words and does not match substrings', () => {
    assert.equal(t('Run history (word)'), 'Run history (word)');
    assert.equal(t('Prefix Run history (123)'), 'Prefix Run history (123)');
    assert.equal(t('Cannot load model.safetensors'), 'Cannot load model.safetensors');
});
test('empty dictionaries and prototype names have a safe English fallback', () => {
    context.window.localization = {};
    assert.equal(t('constructor'), 'constructor');
    assert.equal(t('Unknown: {count}', {count: 3}), 'Unknown: 3');
    delete context.window.localization;
    assert.equal(t('Unknown: {count}', {count: 4}), 'Unknown: 4');
});
