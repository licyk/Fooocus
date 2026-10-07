// Native Fooocus host for Tag Autocomplete (MIT), inspired by Dominik Reh.
(function () {
    'use strict';
    const engine = window.FooocusTagEngine;
    const scripts = new URL('./', document.currentScript.src);
    const apiRoot = new URL('./tagcomplete/v1/', window.location.href);
    const states = new WeakMap(), textareas = new Set(), assets = new Map();
    const storageKey = 'fooocus.tagcomplete.v1';
    let settings, defaults, schema, catalog, capabilities = {}, usage = [], worker, ready = false;
    let active, sequence = 0, request = 0, loading = 0, syncing = false, controlsSynced = false, rebuildTimer, started = false;

    function status(message) {
        const element = document.getElementById('tagcomplete_status');
        if (element) element.textContent = message;
    }
    async function api(path, options = {}) {
        const response = await fetch(new URL(path, apiRoot), {credentials: 'same-origin', ...options});
        if (!response.ok) throw new Error(`Completion service: HTTP ${response.status}`);
        return response.json();
    }
    async function asset(entry) {
        const key = entry.id + ':' + entry.modified;
        if (!assets.has(key)) assets.set(key, fetch(new URL(`asset/${entry.id}`, apiRoot), {credentials: 'same-origin'})
            .then(response => { if (!response.ok) throw new Error(`Cannot load ${entry.name}`); return response.text(); })
            .catch(error => { assets.delete(key); throw error; }));
        return assets.get(key);
    }
    function saveLocal() {
        try { localStorage.setItem(storageKey, JSON.stringify(settings)); } catch (_) { /* Private browser mode. */ }
    }
    function bridge(id, value) {
        const field = document.querySelector(`#${id} textarea, #${id} input`);
        if (!field) return false;
        field.value = value;
        field.dispatchEvent(new Event('input', {bubbles: true}));
        return true;
    }
    function hide(state = active) {
        if (!state) return;
        state.popup.hidden = true;
        state.area.setAttribute('aria-expanded', 'false');
        state.area.removeAttribute('aria-activedescendant');
        state.selected = -1;
    }
    function allowed(state) {
        if (!settings?.enabled || !settings[state.role] || (state.negative && !settings.negative)) return false;
        const models = settings.model_list.split(',').map(x => x.trim()).filter(Boolean);
        if (models.length) {
            const found = models.includes(capabilities.base);
            if ((settings.model_list_mode === 'blacklist' && found) || (settings.model_list_mode === 'whitelist' && !found)) return false;
        }
        return true;
    }
    function position(state) {
        if (state.popup.hidden) return;
        const rect = state.area.getBoundingClientRect();
        let left = rect.left, top = rect.bottom;
        if (settings.follow_cursor && window.FooocusTagCaret) {
            const caret = window.FooocusTagCaret(state.area, state.area.selectionStart);
            left += caret.left - state.area.scrollLeft;
            top = rect.top + Math.min(rect.height, Math.max(0, caret.top - state.area.scrollTop) + (caret.height || 20));
        }
        const popup = state.popup;
        popup.style.maxHeight = `${Math.min(innerHeight * .55, settings.max_results * 40 + 12)}px`;
        popup.style.maxWidth = `${Math.min(640, innerWidth - 16)}px`;
        const box = popup.getBoundingClientRect();
        popup.style.left = `${Math.max(8, Math.min(left, innerWidth - box.width - 8))}px`;
        popup.style.top = `${Math.max(8, Math.min(top + 5, innerHeight - box.height - 8))}px`;
    }
    function select(state, index) {
        state.selected = index;
        while (state.rendered <= index && state.rendered < state.results.length) appendRows(state);
        for (const row of state.list.children) row.classList.toggle('selected', Number(row.dataset.index) === index);
        const row = state.list.children[index];
        if (row) { state.area.setAttribute('aria-activedescendant', row.id); row.scrollIntoView({block: 'nearest'}); }
        preview(state, state.results[index]);
    }
    function preview(state, result) {
        const image = state.image;
        const enabled = settings.previews && result?.preview;
        image.hidden = !enabled;
        if (enabled) {
            image.src = new URL(`asset/${result.preview}`, apiRoot).href;
            image.alt = result.name;
        } else image.removeAttribute('src');
    }
    function appendRows(state) {
        const end = Math.min(state.results.length, state.rendered + settings.batch_size);
        const dark = Boolean(document.querySelector('.dark'));
        for (let index = state.rendered; index < end; index++) {
            const result = state.results[index], row = document.createElement('li');
            row.id = `${state.popup.id}-${index}`;
            row.dataset.index = index; row.role = 'option';
            const label = document.createElement('span'); label.className = 'ftc-label';
            label.textContent = result.matchedAlias ? (settings.only_alias ? result.matchedAlias : `${result.matchedAlias} → ${result.name}`) : result.name;
            if (result.kind === 'chant') label.textContent += ` — ${result.content.slice(0, 100)}`;
            const color = (settings.colors[String(result.category)] || settings.colors.default)?.[dark ? 0 : 1];
            if (color && CSS.supports('color', color)) label.style.color = color;
            if (result.translation) {
                const translation = document.createElement('small'); translation.textContent = result.translation;
                label.appendChild(translation);
            }
            row.appendChild(label);
            const meta = document.createElement('span'); meta.className = 'ftc-meta';
            meta.textContent = `${result.boosted ? '✨ ' : ''}${result.meta || (result.count ? Number(result.count).toLocaleString() : result.kind)}`;
            row.appendChild(meta);
            if (settings.wiki_links && ['tag', 'extra'].includes(result.kind)) {
                const link = document.createElement('a'); link.textContent = '?'; link.target = '_blank'; link.rel = 'noopener noreferrer';
                link.href = result.source?.includes('e621') ? `https://e621.net/wiki_pages/show_or_new?title=${encodeURIComponent(result.name)}` :
                    `https://danbooru.donmai.us/wiki_pages/${encodeURIComponent(result.name)}`;
                link.addEventListener('click', event => event.stopPropagation());
                row.appendChild(link);
            }
            row.addEventListener('pointerdown', event => { if (!event.target.closest('a')) event.preventDefault(); });
            row.addEventListener('click', event => { if (!event.target.closest('a')) choose(state, result).catch(error => status(error.message)); });
            row.addEventListener('pointerenter', () => preview(state, result));
            state.list.appendChild(row);
        }
        state.rendered = end;
    }
    function show(state, results) {
        if (state.filter) results = results.filter(state.filter);
        if (!results.length || document.activeElement !== state.area || !allowed(state)) { hide(state); return; }
        state.results = results; state.selected = -1; state.rendered = 0;
        state.list.replaceChildren(); appendRows(state);
        state.image.hidden = true;
        state.popup.hidden = false;
        state.area.setAttribute('aria-expanded', 'true');
        position(state);
    }
    async function query(state) {
        if (!allowed(state) || state.composing || !ready || document.activeElement !== state.area) { hide(state); return; }
        const ctx = engine.context(state.area.value, state.area.selectionStart, state.area.selectionEnd);
        if (!ctx) { hide(state); return; }
        ctx.role = state.role; ctx.negative = state.negative;
        state.context = ctx; state.value = state.area.value; state.cursor = state.area.selectionStart;
        state.selectionEnd = state.area.selectionEnd;
        if (active !== state) hide(active);
        active = state;
        const id = ++request;
        state.request = id;
        let wildcardValues = [];
        if (ctx.kind === 'wildcard_value') {
            const entry = catalog.wildcards.find(x => x.name === ctx.wildcard) || catalog.wildcards.find(x => x.name.split('/').at(-1) === ctx.wildcard);
            if (entry) wildcardValues = (await asset(entry)).split(/\r?\n/).filter(x => x.trim() && !x.trimStart().startsWith('#'));
        }
        if (request !== id || state.area.value !== state.value || state.area.selectionStart !== state.cursor || state.area.selectionEnd !== state.selectionEnd || state.composing) return;
        worker.postMessage({type: 'query', id, context: ctx, wildcardValues});
    }
    function schedule(state) {
        if (!states.has(state.area)) return;
        clearTimeout(state.timer);
        state.timer = setTimeout(() => query(state).catch(error => status(error.message)), settings?.delay_ms || 0);
    }
    function replace(area, start, end, text, caret) {
        area.focus(); area.setSelectionRange(start, end);
        // insertText preserves the browser's native undo stack, unlike assigning .value.
        if (!document.execCommand('insertText', false, text)) area.setRangeText(text, start, end, 'end');
        area.setSelectionRange(caret, caret);
        area.dispatchEvent(new InputEvent('input', {bubbles: true, inputType: 'insertReplacementText', data: text}));
    }
    async function choose(state, result, tab = false) {
        if (!result || state.value !== state.area.value || state.area.selectionStart !== state.cursor || state.area.selectionEnd !== state.selectionEnd || !state.context || state.composing) return;
        const ctx = state.context, original = state.area.value;
        let text = engine.sanitize(result, settings, ctx), partial = false;
        if (tab && result.kind === 'wildcard' && settings.wildcard_mode !== 'full' && text.includes('/')) {
            if (settings.wildcard_mode === 'next_folder') {
                const next = text.indexOf('/', Math.max(2, original.slice(ctx.start, state.cursor).length));
                if (next >= 0) { text = text.slice(0, next + 1); partial = true; }
            } else {
                const names = state.results.filter(x => x.kind === 'wildcard').map(x => `__${x.name}__`);
                let length = names[0]?.length || 0;
                for (const name of names) { let i = 0; while (i < length && name[i] === names[0][i]) i++; length = i; }
                if (length < text.length) { text = text.slice(0, length); partial = true; }
            }
        }
        const after = original.slice(ctx.end);
        const separator = engine.completionSeparator(result, settings, after, state.appendSeparator);
        let insertion = text + separator, caret = ctx.start + insertion.length;
        let full = original.slice(0, ctx.start) + insertion + after;
        if (result.kind === 'lora' && settings.trigger_words && result.keywords) {
            const keywords = result.keywords.trim();
            if (settings.trigger_location === 'start') { full = keywords + ', ' + full; caret += keywords.length + 2; }
            else if (settings.trigger_location === 'end') { full += (full.trimEnd().endsWith(',') ? ' ' : ', ') + keywords; }
            else { insertion = keywords + ', ' + insertion; full = original.slice(0, ctx.start) + insertion + after; caret += keywords.length + 2; }
        }
        if (result.kind === 'style' && !bridge('tagcomplete_style_bridge', JSON.stringify({name: result.name, nonce: ++sequence}))) throw new Error('Styles bridge is not ready.');
        if (result.kind === 'lora' && settings.trigger_words && result.keywords) replace(state.area, 0, original.length, full, caret);
        else replace(state.area, ctx.start, ctx.end, insertion, caret);
        hide(state);
        state.onCommit?.(state.area.value, result);
        if (settings.frequency && !partial) {
            await api('usage', {method: 'POST', headers: {'Content-Type': 'application/json'}, body: JSON.stringify({name: result.name, kind: result.kind, negative: state.negative})});
            usage = await api('usage'); worker.postMessage({type: 'usage', usage});
        }
        if (result.kind === 'wildcard') schedule(state);
    }
    function keydown(state, event) {
        if (event.isComposing || state.composing || event.keyCode === 229 || state.popup.hidden || !allowed(state)) return;
        const combo = (event.ctrlKey ? 'Ctrl+' : '') + (event.altKey ? 'Alt+' : '') + (event.shiftKey ? 'Shift+' : '') + (event.metaKey ? 'Meta+' : '') + event.key;
        // Fooocus generation and attention-weight shortcuts retain priority.
        if ((event.ctrlKey || event.metaKey) && ['Enter', 'ArrowUp', 'ArrowDown'].includes(event.key)) return;
        const action = Object.entries(settings.keymap).find(([, key]) => key === combo)?.[0];
        if (!action) return;
        if (action === 'choose' && state.selected < 0) { hide(state); return; }
        event.preventDefault(); event.stopPropagation();
        const size = state.results.length;
        if (action === 'close') hide(state);
        else if (action === 'choose' || action === 'choose_first') choose(state, state.results[Math.max(0, state.selected)], event.key === 'Tab').catch(error => status(error.message));
        else {
            const index = {up: (state.selected < 0 ? size : state.selected) - 1,
                down: state.selected + 1, page_up: Math.max(0, state.selected - 5), page_down: Math.min(size - 1, state.selected + 5), first: 0, last: size - 1}[action];
            select(state, (index + size) % size);
        }
    }
    function bind() {
        if (settings && !controlsSynced) {
            syncing = true;
            controlsSynced = bridge('tagcomplete_settings_bridge', JSON.stringify(settings));
        }
        for (const area of document.querySelectorAll('.fooocus-completion textarea')) {
            if (area.closest('.pai-host') || states.has(area)) continue;
            const container = area.closest('.fooocus-completion');
            const role = container.classList.contains('completion-inpaint') ? 'inpaint' :
                container.classList.contains('completion-enhance') ? 'enhance' : container.classList.contains('completion-negative') ? 'negative' : 'positive';
            attach(area, {role, negative: container.classList.contains('completion-negative'), container});
        }
        for (const area of textareas) if (!area.isConnected) detach(area);
    }
    function attach(area, options = {}) {
        if (states.has(area)) return () => detach(area);
        const {role = 'positive', negative = false, container = area.parentElement, onCommit, appendSeparator = true, filter, capture = false} = options;
        const popup = document.createElement('div'); popup.className = 'ftc-popup'; popup.id = `ftc-popup-${++sequence}`; popup.hidden = true;
        const list = document.createElement('ul'); list.role = 'listbox'; list.setAttribute('aria-label', 'Prompt completions');
        const image = document.createElement('img'); image.className = 'ftc-preview'; image.hidden = true;
        image.addEventListener('error', () => { image.hidden = true; });
        popup.append(list, image); document.body.appendChild(popup);
        const abort = new AbortController();
        const state = {area, role, negative, popup, list, image, onCommit, appendSeparator, filter, abort, composing: false, selected: -1, results: []};
        states.set(area, state); textareas.add(area);
        area.setAttribute('aria-autocomplete', 'list'); area.setAttribute('aria-controls', popup.id); area.setAttribute('aria-expanded', 'false');
        const listen = (name, callback) => area.addEventListener(name, callback, {signal: abort.signal});
        listen('input', () => { hide(state); if (!state.composing) schedule(state); updateTranslation(state); });
        listen('focus', () => { if (active !== state) hide(active); active = state; });
        listen('blur', () => hide(state));
        listen('click', () => schedule(state));
        area.addEventListener('keydown', event => {keydown(state, event); if (capture && event.defaultPrevented) event.stopImmediatePropagation();}, {signal: abort.signal, capture});
        listen('keyup', event => { if (['ArrowLeft', 'ArrowRight', 'Home', 'End'].includes(event.key)) schedule(state); });
        listen('compositionstart', () => { state.composing = true; clearTimeout(state.timer); hide(state); });
        listen('compositionend', () => { state.composing = false; schedule(state); });
        listen('scroll', () => position(state));
        popup.addEventListener('scroll', () => { if (popup.scrollTop + popup.clientHeight >= popup.scrollHeight - 60) appendRows(state); });
        const translation = document.createElement('div'); translation.className = 'ftc-translation'; translation.hidden = true;
        container.appendChild(translation); state.translation = translation;
        if (document.activeElement === area) { hide(active); active = state; schedule(state); }
        return () => detach(area);
    }
    function detach(area) {
        const state = states.get(area);
        if (!state) return;
        clearTimeout(state.timer); state.abort.abort(); hide(state);
        if (active === state) { active = undefined; ++request; }
        state.popup.remove(); state.translation.remove();
        for (const name of ['aria-autocomplete', 'aria-controls', 'aria-expanded', 'aria-activedescendant']) area.removeAttribute(name);
        states.delete(area); textareas.delete(area);
    }
    let translationMap = new Map();
    function updateTranslation(state) {
        state.translation.hidden = !settings?.live_translation || !allowed(state);
        if (state.translation.hidden) return;
        const translated = state.area.value.split(/[,\n]/).map(tag => {
            const key = engine.normalize(tag.trim().replace(/^\(/, '').replace(/:[\d.]+\)$/, ''));
            return translationMap.get(key);
        }).filter(Boolean);
        state.translation.textContent = translated.join(' · ');
    }
    async function rebuild() {
        if (!settings || !catalog) return;
        const id = ++loading;
        ready = false; ++request; hide();
        if (!settings.enabled) {
            for (const area of textareas) updateTranslation(states.get(area));
            status('Prompt completion disabled.'); return;
        }
        status('Loading prompt completion data…');
        const datasets = {};
        for (const [kind, key] of [['main', 'tag_file'], ['translation', 'translation_file'], ['extra', 'extra_file'], ['chants', 'chant_file']]) {
            if (settings[key] === 'None') { datasets[kind] = ''; continue; }
            const entry = catalog.datasets.find(x => x.name === settings[key]);
            if (!entry) throw new Error(`Completion dataset not found: ${settings[key]}`);
            datasets[kind] = await asset(entry);
        }
        if (loading !== id) return;
        usage = settings.frequency ? await api('usage') : [];
        if (loading !== id) return;
        translationMap = new Map(engine.parseCSV(datasets.translation || '').map(row => [engine.normalize(row[0]), row[settings.translation_old_format ? 2 : 1] || '']));
        worker.postMessage({type: 'init', id, settings, catalog, datasets, capabilities, usage});
        for (const area of textareas) updateTranslation(states.get(area));
    }
    function rebuildSoon() {
        clearTimeout(rebuildTimer);
        rebuildTimer = setTimeout(() => rebuild().catch(error => status(error.message)), 120);
    }
    function setSetting(key, value) {
        if (!settings || syncing) return;
        if (['keymap', 'colors'].includes(key)) {
            try { value = JSON.parse(value); } catch (_) { status(`Invalid JSON for ${key}.`); return; }
            if (!value || typeof value !== 'object' || Array.isArray(value)) { status(`Invalid ${key}.`); return; }
        }
        try { settings = engine.validateSettings({...settings, [key]: value}, defaults, schema); }
        catch (error) { status(error.message); return; }
        saveLocal(); rebuildSoon();
    }
    function useSettings(config, updateControls = false) {
        settings = config; saveLocal();
        if (updateControls) { syncing = true; bridge('tagcomplete_settings_bridge', JSON.stringify(config)); }
        rebuildSoon();
    }
    async function refresh() {
        catalog = await api('refresh', {method: 'POST'});
        window.dispatchEvent(new Event('fooocus-completion-catalog'));
        assets.clear(); await rebuild();
    }
    async function start() {
        if (started) return;
        started = true;
        try {
            const data = await api('bootstrap');
            catalog = data.catalog; settings = defaults = data.settings; schema = data.schema;
            window.dispatchEvent(new Event('fooocus-completion-catalog'));
            try {
                const local = JSON.parse(localStorage.getItem(storageKey) || 'null');
                if (local && typeof local === 'object' && !Array.isArray(local)) settings = engine.validateSettings({...defaults, ...local}, defaults, schema);
            } catch (_) { settings = defaults; saveLocal(); }
            worker = new Worker(new URL('worker.js', scripts));
            worker.onerror = () => { ready = false; status('Prompt completion worker failed to load.'); };
            worker.onmessage = ({data}) => {
                if (data.type === 'ready' && data.id === loading) {
                    ready = true;
                    for (const entry of data.translations) translationMap.set(...entry);
                    for (const area of textareas) updateTranslation(states.get(area));
                    status(`Prompt completion ready: ${data.count.toLocaleString()} entries.`);
                    if (active) schedule(active);
                }
                else if (data.type === 'results' && ready && active?.request === data.id && request === data.id && !active.composing && active.area.value === active.value && active.area.selectionStart === active.cursor && active.area.selectionEnd === active.selectionEnd) show(active, data.results);
                else if (data.type === 'error') { ready = false; hide(); status(data.message); }
            };
            syncing = true; bind();
            await rebuild();
        } catch (error) { status(error.message); started = false; }
    }
    window.FooocusTagComplete = {
        start, setSetting, useSettings, refresh, attach, detach, hide,
        translation(text) { return translationMap.get(engine.normalize(text)) || ''; },
        get catalog() { return catalog; }, get capabilities() { return capabilities; },
        finishSettings(config) { syncing = false; useSettings(config); },
        setCapabilities(value) { capabilities = value; window.dispatchEvent(new CustomEvent('fooocus-model-capabilities', {detail: value})); ++request; worker?.postMessage({type: 'capabilities', capabilities}); hide(); if (active) schedule(active); },
        async clearUsage() { await api('usage', {method: 'DELETE'}); usage = []; worker?.postMessage({type: 'usage', usage}); status('Completion usage cleared for this login.'); },
        get settings() { return settings; }, get ready() { return ready; },
    };
    onUiLoaded(start); onAfterUiUpdate(bind);
    // Gradio mounts hidden panels lazily. Bind before their first input, even
    // when focus arrives before the host's debounced UI update callback.
    document.addEventListener('focusin', event => {
        if (event.target.matches('.fooocus-completion textarea') && !states.has(event.target)) bind();
    });
    window.addEventListener('resize', () => { if (active) position(active); });
    window.addEventListener('scroll', () => { if (active) position(active); }, true);
    document.addEventListener('pointerdown', event => { if (active && !active.popup.contains(event.target) && event.target !== active.area) hide(); });
})();
