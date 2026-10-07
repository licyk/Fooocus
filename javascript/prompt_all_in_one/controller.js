// Fooocus host bridge for Physton's MIT Prompt All-in-One; heavy UI loads on demand.
(function () {
    'use strict';
    const assets = new URL('./', document.currentScript.src);
    const version = new URL(document.currentScript.src).search;
    const root = new URL('./prompt-all-in-one/v1/', window.location.href);
    const hosts = new Map();
    let availableProviders = [];
    let settings, defaults, storageKey, bridged = false, timer, bundle, session = 0, starting, preferenceRevision = 0;
    const controllers = new Set();
    function status(text) {
        const node = document.getElementById('prompt_all_in_one_status');
        if (node) node.textContent = text;
    }
    function validate(values) {
        const result = {...defaults};
        for (const [key, value] of Object.entries(values || {})) {
            if (!Object.hasOwn(defaults, key) || typeof value !== typeof defaults[key]) continue;
            if (key === 'weight_step' && (!Number.isFinite(value) || value < .01 || value > 1)) continue;
            if (key === 'history_limit' && (!Number.isInteger(value) || value < 10 || value > 1000)) continue;
            if (key.endsWith('_color') && !/^#[0-9a-f]{6}$/i.test(value)) continue;
            if (key === 'source_language' && !['auto','zh','en','ja','ko','de','fr','es','it','pt','ru'].includes(value)) continue;
            const choices = {language: ['zh_CN','zh_TW','en_US','ja_JP','ko_KR','de_DE','fr_FR','es_ES','it_IT','pt_PT','ru_RU','zh_HK'], target_language: ['zh','en','ja','ko','de','fr','es','it','pt','ru'], provider: availableProviders, separator: ['comma','newline'], theme: ['auto','light','dark']};
            if (choices[key] && !choices[key].includes(value)) continue;
            if (typeof value === 'string' && value.length > 100000) continue;
            result[key] = value;
        }
        try { if (!Array.isArray(JSON.parse(result.custom_groups))) result.custom_groups = '[]'; } catch (_) { result.custom_groups = '[]'; }
        return result;
    }
    async function request(path, options = {}, activeOnly = true) {
        const generation = session, controller = new AbortController(); controllers.add(controller);
        try {
            const response = await fetch(new URL(path, root), {credentials: 'same-origin', ...options, signal: controller.signal});
            if (!response.ok) {
                let detail; try { detail = (await response.json()).detail; } catch (_) { /* Non-JSON error. */ }
                throw new Error(typeof detail === 'string' ? detail : `Prompt editor: HTTP ${response.status}`);
            }
            const result = await response.json();
            if (activeOnly && (generation !== session || !settings?.enabled)) throw new DOMException('Editor disabled', 'AbortError');
            return result;
        } finally { controllers.delete(controller); }
    }
    function post(path, data, method = 'POST') {
        return request(path, {method, headers: {'Content-Type': 'application/json'}, body: JSON.stringify(data)});
    }
    function save() { try { localStorage.setItem(storageKey, JSON.stringify(settings)); } catch (_) { /* Private mode. */ } }
    function bridge(force = false) {
        if (!settings || (bridged && !force)) return;
        const area = document.querySelector('#prompt_all_in_one_settings_bridge textarea');
        if (area) {
            bridged = true; area.value = JSON.stringify({settings, revision: ++preferenceRevision});
            area.dispatchEvent(new Event('input', {bubbles: true}));
        }
    }
    function loadBundle() {
        if (bundle) return bundle;
        const sheet = document.createElement('link'); sheet.rel = 'stylesheet'; sheet.href = new URL('editor.css' + version, assets).href;
        document.head.appendChild(sheet);
        bundle = new Promise((resolve, reject) => {
            const script = document.createElement('script'); script.src = new URL('editor.js' + version, assets).href;
            script.onload = () => {
                if (window.FooocusPromptEditor?.mount) resolve(window.FooocusPromptEditor);
                else { bundle = null; script.remove(); sheet.remove(); reject(new Error('Prompt editor bundle failed to initialize')); }
            };
            script.onerror = () => { script.remove(); sheet.remove(); bundle = null; reject(new Error('Cannot load Prompt All-in-One editor')); };
            document.head.appendChild(script);
        });
        return bundle;
    }
    function scopeFor(area) {
        const container = area.closest('.fooocus-completion');
        if (!container || area.closest('.pai-host')) return null;
        const id = container.id;
        if (id === 'positive_prompt') return {key: 'positive', role: 'positive', negative: false};
        if (id === 'negative_prompt') return {key: 'negative', role: 'negative', negative: true};
        if (id === 'inpaint_additional_prompt') return {key: 'inpaint', role: 'inpaint', negative: false};
        const enhance = /^enhance_(negative_)?prompt_(\d+)$/.exec(id);
        if (enhance) return {key: `enhance_${enhance[1] || ''}${enhance[2]}`, role: 'enhance', negative: Boolean(enhance[1])};
        return null;
    }
    function unmount(area, entry) {
        entry.editor.flush(); entry.editor.destroy();
        entry.container.classList.remove('pai-native-hidden'); entry.node.remove(); hosts.delete(area);
    }
    async function reconcile() {
        bridge();
        if (!settings?.enabled) return;
        const generation = session;
        try {
            const ui = await loadBundle();
            if (generation !== session || !settings.enabled) return;
            for (const [area, entry] of hosts) {
                if (!area.isConnected || !settings[entry.scope.role]) { unmount(area, entry); continue; }
                entry.container.classList.toggle('pai-native-hidden', settings.hide_native);
                entry.editor.sync(); entry.editor.configure(settings);
            }
            for (const area of document.querySelectorAll('.fooocus-completion textarea')) {
                const scope = scopeFor(area);
                if (!scope || !settings[scope.role] || hosts.has(area)) continue;
                const container = area.closest('.fooocus-completion'), node = document.createElement('div');
                node.className = 'pai-host'; node.dataset.scope = scope.key; container.appendChild(node);
                const write = text => {
                    if (area.value === text) return;
                    area.value = text; area.dispatchEvent(new InputEvent('input', {bubbles: true, inputType: 'insertReplacementText'}));
                };
                const editor = ui.mount(node, {area, scope, settings, request, post, write, setSetting});
                hosts.set(area, {editor, scope, container, node});
                container.classList.toggle('pai-native-hidden', settings.hide_native);
            }
            status(`Prompt All-in-One enabled (${hosts.size} prompts).`);
        } catch (error) { if (error.name !== 'AbortError') status(error.message); }
    }
    function apply() {
        if (!settings.enabled) {
            ++session; clearInterval(timer); timer = undefined;
            for (const [area, entry] of hosts) unmount(area, entry);
            for (const controller of controllers) controller.abort();
            status('Prompt All-in-One disabled.');
        } else {
            if (!timer) timer = setInterval(reconcile, 250);
            reconcile();
        }
    }
    function setSetting(key, value) {
        if (!settings) return;
        settings = validate({...settings, [key]: value}); save(); apply(); bridge(true);
    }
    function useSettings(value) { settings = validate(value); save(); apply(); bridge(true); }
    async function start() {
        if (starting) return starting;
        starting = (async () => {
            try {
                const data = await request('bootstrap', {}, false);
                availableProviders = data.providers; defaults = data.settings; storageKey = 'fooocus.prompt-all-in-one.v1.' + data.user;
                let local; try { local = JSON.parse(localStorage.getItem(storageKey) || 'null'); } catch (_) { /* Corrupt preference. */ }
                settings = validate(local || defaults); bridge(); apply();
            } catch (error) { status(error.message); starting = undefined; }
        })();
        return starting;
    }
    window.FooocusPromptAllInOne = {
        start, setSetting, useSettings,
        finishSettings(value) {
            if (value._bridge_revision !== preferenceRevision) return;
            settings = validate(value); save(); apply();
        },
        flush() { for (const entry of hosts.values()) entry.editor.flush(); },
        get settings() { return settings; }, get mounted() { return hosts.size; },
    };
    onUiLoaded(start); onAfterUiUpdate(() => { bridge(); if (settings?.enabled) reconcile(); });
    document.addEventListener('pointerdown', event => {
        if (event.target.closest('#generate_button')) window.FooocusPromptAllInOne.flush();
    }, true);
    document.addEventListener('keydown', event => {
        if ((event.ctrlKey || event.metaKey) && event.key === 'Enter') window.FooocusPromptAllInOne.flush();
    }, true);
})();
