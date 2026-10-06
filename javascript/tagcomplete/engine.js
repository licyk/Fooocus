// Adapted from a1111-sd-webui-tagcomplete (MIT), Dominik Reh.
// Host-independent CSV, token-range and matching logic, also used in the Worker.
(function (root) {
    'use strict';
    const normalize = value => String(value ?? '').replace(/\\([()[\]\\])/g, '$1').replaceAll('_', ' ').toLowerCase();
    const escapeRegex = value => value.replace(/[.*+?^${}()|[\]\\]/g, '\\$&');

    function validateSettings(values, defaults, schema) {
        const result = {};
        for (const {key, kind, bounds} of schema) {
            const value = values[key] ?? defaults[key];
            let valid;
            if (kind === 'bool') valid = typeof value === 'boolean';
            else if (kind === 'int' || kind === 'float') valid = typeof value === 'number' && Number.isFinite(value) &&
                value >= bounds[0] && value <= bounds[1] && (kind !== 'int' || Number.isInteger(value));
            else if (kind === 'choice') valid = bounds.includes(value);
            else if (kind === 'categories') valid = Array.isArray(value) && value.every(x => /^-1$|^[0-8]$/.test(String(x)));
            else if (kind === 'object') {
                valid = Boolean(value) && typeof value === 'object' && !Array.isArray(value);
                if (valid) {
                    const entries = Object.values(value);
                    valid = key === 'keymap' ? Object.keys(value).sort().join() === Object.keys(defaults[key]).sort().join() &&
                        entries.every(x => typeof x === 'string' && x.length <= 64) &&
                        entries.filter(Boolean).length === new Set(entries.filter(Boolean)).size :
                        entries.every(x => Array.isArray(x) && x.length === 2 && x.every(c => typeof c === 'string' && c.length <= 64));
                }
            } else {
                valid = typeof value === 'string' && value.length <= 4096;
                if (valid && ['csv', 'json_file'].includes(kind) && value !== 'None') valid = !/[\/\\]/.test(value) &&
                    value.endsWith(kind === 'csv' ? '.csv' : '.json');
            }
            if (!valid) throw new Error(`Invalid completion setting: ${key}.`);
            result[key] = kind === 'categories' ? value.map(String) : value;
        }
        if (result.only_alias) result.search_aliases = true;
        return result;
    }

    function parseCSV(text) {
        const rows = [];
        let row = [], field = '', quoted = false;
        text = text.replace(/^\uFEFF/, '');
        for (let i = 0; i < text.length; i++) {
            const ch = text[i];
            if (ch === '"') {
                if (quoted && text[i + 1] === '"') { field += '"'; i++; }
                else if (!field || quoted) quoted = !quoted;
                else field += ch;
            } else if (!quoted && ch === ',') { row.push(field); field = ''; }
            else if (!quoted && (ch === '\n' || ch === '\r')) {
                if (ch === '\r' && text[i + 1] === '\n') i++;
                row.push(field); if (row.some(x => x.trim())) rows.push(row);
                row = []; field = '';
            } else field += ch;
        }
        if (quoted) throw new Error('Unclosed quote in tag CSV.');
        row.push(field); if (row.some(x => x.trim())) rows.push(row);
        return rows;
    }

    function escaped(text, index) {
        let count = 0;
        while (index > 0 && text[--index] === '\\') count++;
        return count % 2 !== 0;
    }

    function context(text, cursor, selectionEnd = cursor) {
        if (selectionEnd > cursor) return {kind: 'tag', query: text.slice(cursor, selectionEnd), start: cursor, end: selectionEnd};
        let start = cursor, end = cursor;
        while (start > 0 && !((',\n\r'.includes(text[start - 1])) && !escaped(text, start - 1))) start--;
        while (end < text.length && !((',\n\r'.includes(text[end])) && !escaped(text, end))) end++;
        while (start < cursor && /\s/.test(text[start])) start++;
        const before = text.slice(start, cursor);
        const angle = before.lastIndexOf('<');
        if (angle >= 0 && !before.slice(angle).includes('>')) {
            start += angle;
            const closed = text.indexOf('>', cursor);
            if (closed >= 0 && closed < end) end = closed + 1;
            let query = text.slice(start + 1, cursor);
            let kind = 'network';
            for (const [prefix, type] of [['lora:', 'lora'], ['l:', 'lora'], ['e:', 'embedding'], ['chant:', 'chant'], ['c:', 'chant']]) {
                if (query.startsWith(prefix)) { query = query.slice(prefix.length); kind = type; break; }
            }
            if (query.includes(':') || /^(h:|hypernet:|lyco:|\[)/.test(query)) return null;
            const weight = text.slice(start, end).match(/:([+-]?(?:\d+(?:\.\d*)?|\.\d+))>$/)?.[1];
            return {kind, query, start, end, weight};
        }
        const wc = before.indexOf('__');
        if (wc >= 0) {
            const token = text.slice(start + wc, end), close = token.indexOf('__', 2);
            const typed = cursor - start - wc;
            if (close >= 0 && typed >= close + 2) {
                const query = token.slice(close + 2, typed);
                if (/\s/.test(query)) return null;
                return {kind: 'wildcard_value', wildcard: token.slice(2, close), query, start: start + wc, end};
            }
            return {kind: 'wildcard', query: text.slice(start + wc + 2, cursor), start: start + wc, end};
        }
        while ('(['.includes(text[start]) && start < cursor) start++;
        // Remove only a real numeric weight suffix; literal parentheses belong to the tag.
        const segment = text.slice(start, end);
        const weight = segment.match(/:([+-]?(?:\d+(?:\.\d*)?|\.\d+))\s*[)\]]*$/);
        if (weight) end = start + weight.index;
        else if (start > 0 && '(['.includes(text[start - 1])) {
            const closer = text[start - 1] === '(' ? ')' : ']';
            if (text[end - 1] === closer) end--;
        }
        if (cursor > end) return null;
        while (end > cursor && /\s/.test(text[end - 1])) end--;
        let query = text.slice(start, cursor), kind = 'tag';
        if (query.startsWith('embedding:')) { kind = 'embedding'; query = query.slice(10); }
        else if (query.startsWith('@')) { kind = 'artist'; query = query.slice(1); }
        else if (query.startsWith('$')) {
            kind = 'style'; query = query.slice(1).replace(/^\(/, '');
            if (query.endsWith(')')) query = query.slice(0, -1);
        }
        if (!query && kind === 'tag') return null;
        return {kind, query, start, end};
    }

    function sanitize(result, settings, ctx) {
        let text = result.name;
        if (result.kind === 'lora') return `<lora:${text}:${ctx.weight ?? result.weight ?? settings.lora_weight}>`;
        if (result.kind === 'embedding') return `embedding:${text}`;
        if (result.kind === 'wildcard') return `__${text}__`;
        if (result.kind === 'chant') return result.content;
        if (result.kind === 'style') return '';
        if (result.kind === 'wildcard_value') return text;
        const exceptions = settings.underscore_exclusions.split(',').map(x => x.trim());
        if (settings.replace_underscores && !exceptions.includes(text)) text = text.replaceAll('_', ' ');
        if (settings.escape_parentheses) text = text.replace(/(?<!\\)([()])/g, '\\$1');
        if (Number(result.category) === 1 && (settings.artist_prefix === 'always' ||
                (settings.artist_prefix === 'when_triggered' && ctx.kind === 'artist'))) text = '@' + text;
        return text;
    }

    function buildIndex(datasets, catalog, settings) {
        const translations = new Map();
        for (const row of parseCSV(datasets.translation || '')) {
            if (row[0]) translations.set(row[0].trim(), row[settings.translation_old_format ? 2 : 1] || '');
        }
        const fromCSV = (csv, kind) => parseCSV(csv || '').filter(x => x[0]?.trim()).map(row => ({
            name: row[0].trim(), kind, category: Number(row[1] || 0), count: Number(row[2]) || 0,
            meta: kind === 'extra' ? row[2] || 'Custom tag' : '', aliases: (row[3] || '').split(',').filter(Boolean),
            translation: row[4] || translations.get(row[0].trim()) || '', source: settings.tag_file,
        }));
        const main = fromCSV(datasets.main, 'tag'), extras = fromCSV(datasets.extra, 'extra');
        for (const extra of extras) if (extra.translation) translations.set(extra.name, extra.translation);
        const names = new Set();
        const tags = (settings.extra_mode === 'before' ? [...extras, ...main] : [...main, ...extras]).filter(row => {
            if (names.has(row.name)) return false;
            names.add(row.name); return true;
        });
        let chants = [];
        if (datasets.chants) {
            const rows = JSON.parse(datasets.chants);
            if (!Array.isArray(rows)) throw new Error('Chants must be a JSON array.');
            chants = rows.filter(row => typeof row.name === 'string' && typeof row.content === 'string').map(row => ({
                name: row.name, content: row.content, terms: String(row.terms || ''), kind: 'chant', category: row.color,
            }));
        }
        const records = [...tags, ...catalog.loras.map(x => ({...x, kind: 'lora'})),
            ...catalog.embeddings.map(x => ({...x, kind: 'embedding'})),
            ...catalog.wildcards.map(x => ({...x, kind: 'wildcard'})),
            ...catalog.styles.map(x => ({...x, kind: 'style'})), ...chants];
        return records.map(row => ({...row, normalized: normalize(row.name), aliasKeys: (row.aliases || []).map(normalize),
            aliasTranslations: (row.aliases || []).map(x => normalize(translations.get(x) || '')),
            translated: normalize(row.translation), normalizedTerms: normalize(row.terms)}));
    }

    function search(index, ctx, settings, capabilities = {}, usage = [], wildcardValues = []) {
        if (!ctx) return [];
        let query = normalize(ctx.query), substring = query.startsWith('*');
        if (substring) query = query.slice(1);
        if (ctx.kind === 'tag' && !substring && query.length < settings.min_chars) return [];
        const excluded = settings.wildcard_exclusions.split(',').map(x => x.trim()).filter(Boolean);
        const wordRegex = new RegExp('(^|[^a-zA-Z0-9])' + escapeRegex(query), 'i');
        const matches = value => {
            if (!query) return true;
            if (ctx.kind !== 'tag' && ctx.kind !== 'artist') return value.includes(query);
            if (substring || settings.match_mode === 'substring') return value.includes(query);
            if (settings.match_mode === 'prefix') return value.startsWith(query);
            return wordRegex.test(value);
        };
        let candidates = ctx.kind === 'wildcard_value'
            ? wildcardValues.map(name => ({name, kind: 'wildcard_value', normalized: normalize(name), meta: ctx.wildcard})) : index;
        candidates = candidates.filter(row => {
            const k = row.kind;
            if (k === 'lora' && !settings.use_loras) return false;
            if (k === 'embedding' && (!settings.use_embeddings || capabilities.embeddings === false)) return false;
            if (k.startsWith('wildcard') && !settings.use_wildcards) return false;
            if (k === 'wildcard' && row.name.split('/').slice(0, -1).some(x => excluded.includes(x))) return false;
            if (k === 'style' && (!settings.use_styles || ctx.role !== 'positive')) return false;
            if (ctx.kind === 'network') return ['lora', 'embedding', 'chant'].includes(k);
            if (ctx.kind === 'artist') return settings.artist_trigger && ['tag', 'extra'].includes(k) && row.category === 1;
            if (ctx.kind === 'tag') return ['tag', 'extra'].includes(k) || (k === 'embedding' && settings.embeddings_in_tags);
            return k === ctx.kind;
        });
        const history = new Map(usage.filter(x => Boolean(x.negative) === Boolean(ctx.negative)).map(x => [JSON.stringify([x.name, x.kind]), x]));
        const cutoff = settings.frequency_days ? Date.now() - settings.frequency_days * 86400000 : 0;
        const results = [];
        for (const row of candidates) {
            if (['tag', 'extra'].includes(row.kind) && settings.categories.length && !settings.categories.includes(String(row.category))) continue;
            let found = matches(row.normalized), matchedAlias = '';
            if (!found && settings.search_aliases) {
                const i = (row.aliasKeys || []).findIndex(matches);
                if (i >= 0) { found = true; matchedAlias = row.aliases[i]; }
            }
            if (!found && settings.search_translations) found = matches(row.translated || '\0') || (row.aliasTranslations || []).some(matches);
            if (!found && row.kind === 'chant') found = matches(row.normalizedTerms);
            if (!found) continue;
            const result = {...row, matchedAlias};
            delete result.normalized; delete result.aliasKeys; delete result.aliasTranslations;
            delete result.translated; delete result.normalizedTerms;
            const record = history.get(JSON.stringify([row.name, row.kind]));
            result.uses = settings.frequency && record && record.count >= settings.frequency_min &&
                Date.parse(record.last_used) >= cutoff && (!matchedAlias || settings.frequency_aliases) ? record.count : 0;
            result.rank = row.kind === 'extra' ? (settings.extra_mode === 'before' ? 2 : -1) : 0;
            result.score = Math.log1p(row.count || 0);
            results.push(result);
        }
        const boosted = results.filter(x => x.uses).sort((a, b) => b.uses - a.uses);
        const eligible = new Set((settings.frequency_cap ? boosted.slice(0, settings.frequency_cap) : boosted).map(x => x.name + '\0' + x.kind));
        for (const row of results) {
            row.boosted = eligible.has(row.name + '\0' + row.kind);
            if (row.boosted) row.score = settings.frequency_function === 'usage_first' ? row.uses :
                row.score + (settings.frequency_function === 'strong' ? 2 : 1) * Math.log1p(row.uses);
            else if (settings.frequency_function === 'usage_first' && settings.frequency) row.score = 0;
        }
        results.sort((a, b) => {
            if (a.rank !== b.rank) return b.rank - a.rank;
            if (a.score !== b.score) return b.score - a.score;
            if (a.kind === 'wildcard_value' && !settings.wildcard_sort) return 0;
            if (settings.model_sort !== 'name' && (a.modified || b.modified)) return settings.model_sort === 'newest' ? b.modified - a.modified : a.modified - b.modified;
            return a.name.localeCompare(b.name);
        });
        return settings.show_all ? results : results.slice(0, settings.max_results);
    }

    const api = {parseCSV, context, sanitize, normalize, buildIndex, search, validateSettings};
    root.FooocusTagEngine = api;
    if (typeof module !== 'undefined') module.exports = api;
})(typeof self !== 'undefined' ? self : globalThis);
