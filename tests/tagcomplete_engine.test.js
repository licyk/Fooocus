const assert = require('node:assert/strict');
const fs = require('node:fs');
const engine = require('../javascript/tagcomplete/engine.js');
const defaults = JSON.parse(fs.readFileSync(0, 'utf8'));
const catalog = {loras: [{name: 'one/cat', weight: .7}], embeddings: [{name: 'test'}],
    wildcards: [{name: 'hair/color'}], styles: [{name: 'Fooocus V2'}]};
const datasets = {main: 'blue_hair,0,100,"azure_hair,blue_tresses"\nred_hair,0,200,\nfoo_artist,1,50,\natago_(azur_lane),4,20,\n',
    translation: 'blue_hair,蓝发\nazure_hair,蔚蓝头发\n', extra: 'masterpiece,5,Quality,,\n',
    chants: JSON.stringify([{name:'Negative', terms:'bad quality', content:'blurry, low quality', color:5}])};
let count = 0;
function check(name, fn) { fn(); count++; console.log('PASS', name); }
function search(query, kind='tag', custom={}) {
    const settings = {...defaults, ...custom};
    return engine.search(engine.buildIndex(datasets, catalog, settings), {kind, query, role:'positive'}, settings, {embeddings:true});
}
check('quoted commas, escaped quotes, CRLF and Unicode CSV', () => {
    assert.deepEqual(engine.parseCSV('\uFEFFname,0,10,"alias,别名"\r\n"a""b",1,2,\n'), [['name','0','10','alias,别名'],['a"b','1','2','']]);
    assert.throws(() => engine.parseCSV('a,"unfinished'));
});
check('Chinese, aliases, substring and category searches', () => {
    assert.equal(search('蓝发')[0].name, 'blue_hair');
    assert.equal(search('蔚蓝')[0].name, 'blue_hair');
    assert.equal(search('azure')[0].matchedAlias, 'azure_hair');
    assert.equal(search('*hair').length, 2);
    assert.equal(search('hair', 'tag', {match_mode:'prefix'}).length, 0);
    assert.equal(search('*hair', 'tag', {categories:['1']}).length, 0);
});
check('artist, LoRA, embeddings, snippets and Fooocus styles', () => {
    assert.equal(search('foo','artist')[0].name,'foo_artist');
    assert.equal(search('cat','lora')[0].name,'one/cat');
    assert.equal(search('test','embedding')[0].name,'test');
    assert.equal(search('bad','chant')[0].content,'blurry, low quality');
    assert.equal(search('Fooocus','style')[0].name,'Fooocus V2');
});
check('Anima capabilities suppress embeddings, not LoRAs', () => {
    const index=engine.buildIndex(datasets,catalog,defaults);
    assert.equal(engine.search(index,{kind:'embedding',query:'',role:'positive'},defaults,{embeddings:false}).length,0);
    assert.equal(engine.search(index,{kind:'network',query:'',role:'positive'},defaults,{embeddings:false}).some(x=>x.kind==='lora'),true);
});
check('mid-prompt replacement spans preserve weighted suffixes and following tags', () => {
    const text='solo, (blue_hair:1.2), red_hair';
    const c=engine.context(text,13);
    assert.equal(text.slice(c.start,c.end),'blue_hair');
    assert.equal(text.slice(0,c.start)+'new'+text.slice(c.end),'solo, (new:1.2), red_hair');
    const literal='atago_(azur_lane), solo';
    assert.equal(engine.context(literal,5).end,literal.indexOf(','));
});
check('network, wildcard contents and selected ranges', () => {
    assert.equal(engine.context('<lora:one/c',11).kind,'lora');
    assert.equal(engine.context('embedding:tes',13).kind,'embedding');
    assert.equal(engine.context('__hair/color__red',17).kind,'wildcard_value');
    assert.equal(engine.context('__hair/color__ ',15),null);
    assert.deepEqual(engine.context('cat, solo',0,3),{kind:'tag',query:'cat',start:0,end:3});
});
check('Fooocus insertion syntax, artist prefix and literal escaping', () => {
    assert.equal(engine.sanitize({name:'one/cat',kind:'lora',weight:.7},defaults,{kind:'lora'}),'<lora:one/cat:0.7>');
    assert.equal(engine.sanitize({name:'test',kind:'embedding'},defaults,{}),'embedding:test');
    assert.equal(engine.sanitize({name:'hair/color',kind:'wildcard'},defaults,{}),'__hair/color__');
    assert.equal(engine.sanitize({name:'foo_artist',kind:'tag',category:1},defaults,{kind:'artist'}),'@foo artist');
    assert.equal(engine.sanitize({name:'atago_(azur_lane)',kind:'tag',category:4},defaults,{}),'atago \\(azur lane\\)');
});
check('frequency boost respects negative role, expiry and cap', () => {
    const settings={...defaults,frequency:true,frequency_min:1,frequency_function:'usage_first'};
    const index=engine.buildIndex(datasets,catalog,settings);
    const usage=[{name:'blue_hair',kind:'tag',negative:0,count:10,last_used:new Date().toISOString()}];
    assert.equal(engine.search(index,{kind:'tag',query:'*hair'},settings,{},usage)[0].name,'blue_hair');
    assert.equal(engine.search(index,{kind:'tag',query:'*hair',negative:true},settings,{},usage)[0].boosted,false);
    usage[0].last_used='2000-01-01T00:00:00Z';
    assert.equal(engine.search(index,{kind:'tag',query:'*hair'},settings,{},usage)[0].boosted,false);
});
check('extra priority and wildcard content order are configurable', () => {
    assert.equal(search('*', 'tag', {extra_mode:'before'})[0].name,'masterpiece');
    assert.equal(search('*', 'tag', {extra_mode:'after'}).at(-1).name,'masterpiece');
    assert.deepEqual(engine.search([], {kind:'wildcard_value',query:'',wildcard:'test'}, {...defaults,wildcard_sort:false}, {}, [], ['z','a']).map(x=>x.name),['z','a']);
});
check('browser preferences validate types, ranges, paths and keyboard JSON', () => {
    const schema = [{key:'delay_ms',kind:'int',bounds:[0,2000]}, {key:'tag_file',kind:'csv'},
        {key:'keymap',kind:'object'}, {key:'colors',kind:'object'}];
    const valid = engine.validateSettings({...defaults,future:'ignored'}, defaults, schema);
    assert.equal(valid.delay_ms, defaults.delay_ms);
    assert.equal(valid.future, undefined);
    for (const extra of [{delay_ms:-1},{delay_ms:'150'},{tag_file:'../secret.csv'},
            {colors:[]},{keymap:{...defaults.keymap,choose:'Tab'}}]) {
        assert.throws(() => engine.validateSettings({...defaults,...extra},defaults,schema));
    }
});
console.log(`${count} frontend engine checks passed`);
