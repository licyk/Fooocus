// Lossless native Fooocus grammar. Original splitTags normalized user text;
// this port retains all separators/escapes and modifies only explicit edits.
let nextId = 0;
export function parse(text) {
    // Pasted generation metadata must reach Fooocus's JSON parser unchanged.
    if (text.trimStart().startsWith('{')) {
        try { JSON.parse(text); return [{id: ++nextId, raw: text, separator: '', disabled: false, metadata: true}]; } catch (_) { /* Ordinary prompt. */ }
    }
    const tags = []; let start = 0, depth = [], escaped = false;
    for (let i = 0; i < text.length; i++) {
        const char = text[i];
        if (escaped) { escaped = false; continue; }
        if (char === '\\') { escaped = true; continue; }
        if ('([<{'.includes(char)) depth.push(char);
        else if (')]>}'.includes(char) && depth.at(-1) === '([<{'[')]>}'.indexOf(char)]) depth.pop();
        if (!depth.length && (char === ',' || char === '\n' || char === '\r')) {
            let end = i + 1;
            while (end < text.length && /[,\s]/.test(text[end])) end++;
            tags.push({id: ++nextId, raw: text.slice(start, i), separator: text.slice(i, end), disabled: false});
            start = end; i = end - 1;
        }
    }
    tags.push({id: ++nextId, raw: text.slice(start), separator: '', disabled: false});
    return tags;
}
export function serialize(tags, includeDisabled = false) {
    if (includeDisabled || !tags.some(tag => tag.disabled)) return tags.map(tag => tag.raw + tag.separator).join('');
    // Separators belong to the previous enabled item; retain line boundaries.
    const enabled = tags.filter(tag => !tag.disabled);
    return enabled.map((tag, index) => tag.raw + (index < enabled.length - 1 ? (tag.separator || ', ') : '')).join('');
}
export function edit(text, previous) {
    const result = parse(text);
    if (result.some(tag => tag.metadata)) return result;
    previous.forEach((tag, index) => {
        if (tag.disabled) result.splice(Math.min(index, result.length), 0, tag);
    });
    return result;
}
export function snapshot(tags) {
    return tags.map(({raw, separator, disabled}) => ({raw, separator, disabled}));
}
export function restore(tags) {
    return tags.map(tag => ({...tag, id: ++nextId}));
}
export function weight(raw) {
    const trimmed = raw.trim();
    const explicit = /^\((.*):(-?(?:\d+(?:\.\d*)?|\.\d+))\)$/s.exec(trimmed);
    if (explicit) return {text: explicit[1], value: Number(explicit[2])};
    if (trimmed.startsWith('(') && trimmed.endsWith(')')) return {text: trimmed.slice(1, -1), value: 1.1};
    const lora = /^<lora:(.*):(-?(?:\d+(?:\.\d*)?|\.\d+))>$/.exec(trimmed);
    if (lora) return {text: lora[1], value: Number(lora[2]), lora: true};
    return {text: trimmed, value: 1};
}
export function setWeight(raw, value) {
    if (!Number.isFinite(value) || !raw.trim() || raw.trimStart().startsWith('{')) return raw;
    const parsed = weight(raw), number = Math.round(value * 1000) / 1000;
    const leading = raw.match(/^\s*/)[0], trailing = raw.match(/\s*$/)[0];
    return leading + (parsed.lora ? `<lora:${parsed.text}:${number}>` : `(${parsed.text}:${number})`) + trailing;
}
export function translationText(raw) {
    const trimmed = raw.trim();
    if (!trimmed || /^(<|embedding:|__|\{)/.test(trimmed)) return '';
    return weight(raw).text;
}
export function applyTranslation(raw, translated) {
    const current = weight(raw);
    if (!translationText(raw)) return raw;
    const leading = raw.match(/^\s*/)[0], trailing = raw.match(/\s*$/)[0];
    return leading + (raw.trim().startsWith('(') ? `(${translated}:${current.value})` : translated) + trailing;
}
export function append(tags, text, separator = ', ') {
    const addition = parse(text);
    if (tags.length && tags.at(-1).raw && !tags.at(-1).separator) tags.at(-1).separator = separator;
    if (tags.length === 1 && tags[0].raw === '' && !tags[0].separator) tags.splice(0);
    tags.push(...addition);
}
