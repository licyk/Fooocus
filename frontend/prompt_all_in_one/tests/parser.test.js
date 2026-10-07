import {test} from 'node:test';
import assert from 'node:assert/strict';
import {parse, serialize, snapshot, restore, setWeight, weight, append, applyTranslation} from '../src/parser.js';
const samples = [
  '', ' cat ,  dog,\n  natural language\n\nparagraph.  ',
  '(cat, (blue eyes:1.2):0.8), dog', 'cat\\,dog, escaped\\(parentheses\\)',
  '<lora:animals/cat:0.7>, embedding:sub/name, __hair/color__',
  '中文，句子；自然语言。\r\nEnglish sentence.\t\t',
  'brackets[a,b], {literal, braces}, <angle,a>, [unclosed,',
  JSON.stringify({prompt: 'cat, dog', steps: 40}, null, 2),
];
for (const value of samples) test('round trip: ' + JSON.stringify(value), () => {
  assert.equal(serialize(parse(value)), value);
  assert.equal(serialize(restore(snapshot(parse(value)))), value);
});
test('nested and escaped delimiters are not split', () => {
  assert.equal(parse('(cat, (dog:1.3):.8), __hair/color__').length, 2);
  assert.equal(parse('cat\\,dog, cat\\\\, dog').length, 3);
});
test('disabled tags stay in metadata, omit from generation', () => {
  const tags = parse('cat, (dog:1.2), bird'); tags[1].disabled = true;
  assert.equal(serialize(tags), 'cat, bird');
  assert.equal(serialize(tags, true), 'cat, (dog:1.2), bird');
  assert.equal(restore(snapshot(tags))[1].disabled, true);
});
test('weights use native syntax with whitespace and nested groups retained', () => {
  assert.equal(setWeight(' (cat, (dog:1.3):.8) ', .9), ' (cat, (dog:1.3):0.9) ');
  assert.equal(setWeight('<lora:animals/cat:0.7>', .8), '<lora:animals/cat:0.8>');
  assert.equal(weight('(cat)').value, 1.1);
  assert.equal(setWeight('cat', NaN), 'cat');
});
test('append preserves existing multiline input', () => {
  const tags = parse('first line\nsecond line'); append(tags, 'cat, dog');
  assert.equal(serialize(tags), 'first line\nsecond line, cat, dog');
});
test('translation retains weights and ignores network syntax', () => {
  assert.equal(applyTranslation(' (猫:1.2) ', 'cat'), ' (cat:1.2) ');
  for (const value of ['<lora:animals/cat:0.7>', '__hair/color__', 'embedding:name']) assert.equal(applyTranslation(value, 'translated'), value);
});
test('editing visible text preserves disabled tag metadata', async () => {
  const {edit} = await import('../src/parser.js');
  const tags = parse('cat, dog, bird'); tags[1].disabled = true;
  const changed = edit('cat, tree', tags);
  assert.equal(serialize(changed), 'cat, tree');
  assert.equal(changed.find(tag => tag.disabled).raw, 'dog');
  assert.equal(changed.filter(tag => tag.disabled).length, 1);
  assert.equal(edit('{"prompt":"metadata"}', changed).length, 1);
});
