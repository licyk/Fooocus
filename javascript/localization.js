var re_num = /^[.\d]+$/;

var original_lines = {};
var translated_lines = {};
var localizedTextNodes = new WeakMap();
var localizedAttributes = new WeakMap();
var localizationAttributes = ['title', 'placeholder', 'aria-label', 'aria-description', 'alt'];
var localizationExcluded = 'script, style, pre, code, [data-no-localize], ' +
    '[contenteditable]:not([contenteditable="false"]), .pai-host, ' +
    '#physton-prompt-all-in-one, [id^="physton-prompt-extension-"], .ftc-label, .ftc-translation';
var localizationTemplateDictionary;
var localizationTemplates = [];

function hasLocalization() {
    return window.localization && Object.keys(window.localization).length > 0;
}

function textNodesUnder(el) {
    var n, a = [], walk = document.createTreeWalker(el, NodeFilter.SHOW_TEXT, null, false);
    while ((n = walk.nextNode())) a.push(n);
    return a;
}

function canBeTranslated(node, text) {
    if (!text) return false;
    if (!node.parentElement) return false;
    if (node.parentElement.nodeName === 'TEXTAREA') return false;
    if (node.parentElement.closest(localizationExcluded)) return false;
    if (re_num.test(text)) return false;
    return true;
}

function getNumericTemplateTranslation(text, dictionary) {
    if (localizationTemplateDictionary !== dictionary) {
        localizationTemplateDictionary = dictionary;
        localizationTemplates = [];
        for (var [source, translated] of Object.entries(dictionary)) {
            if (typeof translated !== 'string') continue;
            var parts = source.split(/(\{(?:count|status)\})/g);
            if (parts.length === 1 || /\{[a-zA-Z_][a-zA-Z0-9_]*\}/.test(
                source.replace(/\{(?:count|status)\}/g, ''))) continue;
            var names = [];
            var pattern = parts.map(part => {
                if (part === '{count}' || part === '{status}') {
                    names.push(part.slice(1, -1));
                    return '(\\p{Nd}[\\p{Nd}., \\u00a0\\u202f]*)';
                }
                return part.replace(/[.*+?^${}()|[\]\\]/g, '\\$&');
            }).join('');
            localizationTemplates.push({source, translated, names, pattern: new RegExp('^' + pattern + '$', 'u')});
        }
    }
    for (var template of localizationTemplates) {
        var matches = template.pattern.exec(text);
        if (!matches) continue;
        var params = Object.fromEntries(template.names.map((name, index) => [name, matches[index + 1]]));
        return {source: template.source, translated: template.translated.replace(
            /\{([a-zA-Z_][a-zA-Z0-9_]*)\}/g,
            (match, key) => Object.hasOwn(params, key) ? params[key] : match)};
    }
}

function getTranslation(text) {
    if (!text) return undefined;
    var dictionary = window.localization || {};
    var tl = Object.hasOwn(dictionary, text) ? dictionary[text] : undefined;
    if (tl === undefined) {
        // Third-party DOM labels cannot call t(). Limit template matching to
        // declared numeric counters/status codes; user data is excluded above.
        var template = getNumericTemplateTranslation(text, dictionary);
        if (template) { text = template.source; tl = template.translated; }
    }
    if (translated_lines[text] === undefined) original_lines[text] = 1;
    if (typeof tl !== 'string') return undefined;
    if (tl !== undefined) {
        translated_lines[tl] = 1;
    }

    return tl;
}

function translateUI(text, params = {}) {
    var translated = getTranslation(text);
    var result = (translated ?? text).replace(/\{([a-zA-Z_][a-zA-Z0-9_]*)\}/g, (match, key) =>
        Object.hasOwn(params, key) ? String(params[key]) : match);
    if (translated !== undefined) translated_lines[result] = 1;
    return result;
}

window.FooocusI18n = Object.freeze({t: translateUI});

function processTextNode(node) {
    var raw = node.nodeValue;
    if (localizedTextNodes.get(node) === raw) return;
    var text = raw.trim();

    if (!canBeTranslated(node, text)) return;

    var tl = getTranslation(text);
    if (tl !== undefined && tl !== text) {
        var replacement = raw.match(/^\s*/)[0] + tl + raw.match(/\s*$/)[0];
        localizedTextNodes.set(node, replacement);
        node.nodeValue = replacement;
        if (text && node.parentElement) {
          node.parentElement.setAttribute("data-original-text", text);
        }
    }
}

function processLocalizationAttributes(node) {
    if (node.closest(localizationExcluded)) return;
    var previous = localizedAttributes.get(node) || {};
    for (var name of localizationAttributes) {
        var value = node.getAttribute(name);
        if (!value || previous[name] === value) continue;
        var translated = getTranslation(value);
        if (translated !== undefined && translated !== value) {
            previous[name] = translated;
            node.setAttribute(name, translated);
        }
    }
    localizedAttributes.set(node, previous);
}

function processNode(node) {
    if (!node) return;
    if (node.nodeType == 3) {
        processTextNode(node);
        return;
    }

    if (node.nodeType === 1) {
        if (node.closest(localizationExcluded)) return;
        processLocalizationAttributes(node);
    }
    if (!node.querySelectorAll) return;
    node.querySelectorAll(localizationAttributes.map(name => '[' + name + ']').join(','))
        .forEach(processLocalizationAttributes);

    textNodesUnder(node).forEach(function(node) {
        processTextNode(node);
    });
}

function refresh_style_localization() {
    processNode(document.querySelector('.style_selections'));
}

function refresh_aspect_ratios_label(value) {
    const label = document.querySelector('#aspect_ratios_accordion .label-wrap span');
    if (!label) return;
    let translation = getTranslation("Aspect Ratios");
    if (typeof translation == "undefined") {
        translation = "Aspect Ratios";
    }
    label.textContent = translation + " " + htmlDecode(value);
}

function localizeWholePage() {
    processNode(gradioApp());
}

onUiLoaded(function() {
    if (!hasLocalization()) {
        return;
    }

    // Keep text/attribute observation separate from the shared UI callbacks.
    // A frame batches overlapping updates; caches and equality checks stop our
    // own translations from recursively writing to the DOM.
    var pending = new Set();
    var frame;
    var observer = new MutationObserver(function(mutations) {
        for (var mutation of mutations) {
            if (mutation.type === 'childList') {
                mutation.addedNodes.forEach(node => pending.add(node));
            } else {
                pending.add(mutation.target);
            }
        }
        if (!pending.size || frame !== undefined) return;
        frame = requestAnimationFrame(function() {
            frame = undefined;
            var nodes = Array.from(pending);
            pending.clear();
            var roots = new Set(nodes);
            for (var node of nodes) {
                if (!node.isConnected) continue;
                var parent = node.parentNode;
                while (parent && !roots.has(parent)) parent = parent.parentNode;
                if (!parent) processNode(node);
            }
        });
    });
    observer.observe(gradioApp(), {
        childList: true, subtree: true, characterData: true,
        attributes: true, attributeFilter: localizationAttributes,
    });

    localizeWholePage();

    if (localization.rtl) { // if the language is from right to left,
        (new MutationObserver((mutations, observer) => { // wait for the style to load
            mutations.forEach(mutation => {
                mutation.addedNodes.forEach(node => {
                    if (node.tagName === 'STYLE') {
                        observer.disconnect();

                        for (const x of node.sheet.rules) { // find all rtl media rules
                            if (Array.from(x.media || []).includes('rtl')) {
                                x.media.appendMedium('all'); // enable them
                            }
                        }
                    }
                });
            });
        })).observe(gradioApp(), {childList: true});
    }
});
