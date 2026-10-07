<!-- Native Fooocus adaptation of Physton's Prompt All-in-One tag editing workflow (MIT). -->
<template>
  <section class="pai-editor" :class="['pai-theme-' + settings.theme, {'pai-negative': scope.negative}]" :style="{'--pai-lora': settings.lora_color, '--pai-embedding': settings.embedding_color, '--pai-wildcard': settings.wildcard_color}" :aria-label="'Prompt All-in-One ' + scope.key" @keydown="shortcut">
    <header class="pai-toolbar">
      <strong>{{ scope.negative ? t('negative_prompt') : t('prompt') }}</strong>
      <button type="button" :aria-pressed="view === 'text'" @click="setView('text')">{{ t('text_view', 'Text') }}</button>
      <button type="button" :aria-pressed="view === 'tags'" @click="setView('tags')">{{ t('tag_view', 'Tags') }}</button>
      <button type="button" @click="copy">{{ t('copy_to_clipboard') }}</button>
      <button type="button" @click="saveFavorite()">{{ t('add_to_favorite') }}</button>
      <button type="button" :aria-pressed="panel === 'history'" @click="openPanel('history')">{{ t('history') }}</button>
      <button type="button" :aria-pressed="panel === 'favorites'" @click="openPanel('favorites')">{{ t('favorite') }}</button>
      <button type="button" @click="openPanel('groups')">{{ t('group_tags', 'Tag groups') }}</button>
      <button type="button" @click="openPanel('models')">{{ t('extra_networks', 'Models') }}</button>
      <button type="button" @click="translateTags(false)" :disabled="busy">{{ t('translate_keywords_to_local_language') }}</button>
      <button type="button" @click="translateTags(true)" :disabled="busy">{{ t('translate_keyword_to_english') }}</button>
      <button type="button" @click="openPanel('provider')">{{ t('translate_api') }}</button>
      <button type="button" @click="openPanel('generate')">{{ t('chatgpt', 'AI prompt') }}</button>
    </header>
    <textarea v-show="view === 'text'" ref="textArea" class="pai-text" :value="draft" rows="3" :aria-label="'Prompt text ' + scope.key" @input="onText" @compositionstart="composing = true" @compositionend="compositionEnd"></textarea>
    <div v-show="view === 'tags'">
      <div class="pai-batch">
        <button type="button" @click="selectAll">{{ t('select_all', 'Select all') }}</button>
        <span>{{ selected.size }}</span>
        <button type="button" :disabled="!selected.size" @click="batch('increase')">+ {{ t('weight', 'Weight') }}</button>
        <button type="button" :disabled="!selected.size" @click="batch('decrease')">− {{ t('weight', 'Weight') }}</button>
        <button type="button" :disabled="!selected.size" @click="batch('disable')">{{ t('disable_keyword') }}</button>
        <button type="button" :disabled="!selected.size" @click="batch('enable')">{{ t('enable_keyword') }}</button>
        <button type="button" :disabled="!selected.size" @click="batch('remove')">{{ t('delete', 'Delete') }}</button>
        <button type="button" :disabled="!selected.size" @click="saveFavorite(true)">{{ t('add_to_favorite') }}</button>
      </div>
      <div ref="tagList" class="pai-tags">
        <div v-for="tag in tags" :key="tag.id" :data-id="tag.id" class="pai-tag" :class="{'pai-disabled': tag.disabled, 'pai-selected': selected.has(tag.id), 'pai-lora': tag.raw.trim().startsWith('<lora:'), 'pai-embedding': tag.raw.trim().startsWith('embedding:'), 'pai-wildcard': tag.raw.trim().startsWith('__')}" :title="tag.raw" @click.self="select(tag.id)">
          <button class="pai-drag" type="button" :aria-label="t('drag', 'Drag tag')">⠿</button>
          <input type="checkbox" :checked="selected.has(tag.id)" :aria-label="t('select', 'Select tag')" @change="select(tag.id)">
          <textarea class="pai-tag-text" :value="tag.raw" rows="1" :aria-label="'Edit tag ' + tag.id" @input="editTag(tag, $event)" @focus="attachTag($event.target, tag)"></textarea>
          <small v-if="settings.bilingual && translation(tag)">{{ translation(tag) }}</small>
          <NumberInput v-if="!tag.metadata && tag.raw.trim()" :model-value="weight(tag.raw).value" :step="settings.weight_step" :min="-10" :max="10" inline :attrs="{'aria-label': t('weight', 'Weight')}" @update:model-value="value => changeWeight(tag, value)" />
          <button type="button" :aria-label="tag.disabled ? t('enable_keyword') : t('disable_keyword')" @click="toggle(tag)">{{ tag.disabled ? '○' : '●' }}</button>
          <button type="button" :aria-label="t('delete', 'Delete tag')" @click="remove(tag)">×</button>
          <button type="button" :aria-label="t('blacklist', 'Blacklist tag')" @click="blacklistTag(tag)">⊘</button>
        </div>
      </div>
      <div class="pai-append">
        <textarea ref="appendArea" v-model="addition" rows="1" :placeholder="t('please_enter_new_keyword')" :aria-label="'Add tag ' + scope.key" @keydown.enter="appendEnter"></textarea>
        <button type="button" @click="addText">{{ t('add', 'Add') }}</button>
      </div>
    </div>
    <div v-if="panel" class="pai-panel">
      <button class="pai-close" type="button" :aria-label="t('close', 'Close')" @click="panel = ''">×</button>
      <template v-if="panel === 'history' || panel === 'favorites'">
        <h4>{{ panel === 'history' ? t('history') : t('favorite') }}</h4>
        <input v-model="filter" :aria-label="t('search', 'Search')" :placeholder="t('search', 'Search')">
        <button type="button" @click="loadEntries">{{ t('refresh', 'Refresh') }}</button>
        <button type="button" @click="clearEntries">{{ t('clear', 'Clear list') }}</button>
        <ul class="pai-entries">
          <li v-for="(entry, index) in filteredEntries" :key="entry.id">
            <input :value="entry.name" :aria-label="t('name', 'Name')" :placeholder="new Date(entry.time * 1000).toLocaleString()" @change="rename(entry, $event.target.value)">
            <pre>{{ entry.prompt }}</pre>
            <button type="button" @click="useEntry(entry)">{{ t('use') }}</button>
            <button v-if="panel === 'history'" type="button" @click="favoriteEntry(entry)">{{ t('add_to_favorite') }}</button>
            <button v-if="panel === 'favorites' && index > 0" type="button" @click="moveEntry(entry, index)">↑</button>
            <button v-if="panel === 'favorites' && index < filteredEntries.length - 1" type="button" @click="moveEntry(entry, index, 1)">↓</button>
            <button type="button" @click="deleteEntry(entry)">{{ t('delete', 'Delete') }}</button>
          </li>
        </ul>
      </template>
      <template v-else-if="panel === 'groups'">
        <h4>{{ t('group_tags', 'Tag groups') }}</h4>
        <input v-model="filter" :aria-label="t('search', 'Search')" :placeholder="t('search', 'Search')">
        <details v-for="(group, index) in filteredGroups" :key="index" :open="Boolean(filter)">
          <summary>{{ group.name }}</summary>
          <div class="pai-group-tags"><button v-for="(label, tag) in group.tags" :key="tag" type="button" @click="add(tag)">{{ tag }}<small v-if="settings.bilingual && label">{{ label }}</small></button></div>
        </details>
      </template>
      <template v-else-if="panel === 'models'">
        <h4>{{ t('extra_networks', 'Models and wildcards') }}</h4>
        <input v-model="filter" :aria-label="t('search', 'Search')" :placeholder="t('search', 'Search')">
        <div class="pai-models"><button v-for="model in models" :key="model.kind + model.name" type="button" @click="addModel(model)"><img v-if="model.preview" :src="previewUrl(model.preview)" alt="" loading="lazy"><span>{{ model.kind }}: {{ model.name }}</span></button></div>
      </template>
      <template v-else-if="panel === 'provider'">
        <h4>{{ t('translate_api') }}</h4>
        <label>{{ t('translate_api') }} <select :value="settings.provider" @change="setSetting('provider', $event.target.value)"><option v-for="name in providers" :key="name">{{ name }}</option></select></label>
        <label>{{ t('local_language') }} <select :value="settings.target_language" @change="setSetting('target_language', $event.target.value)"><option v-for="name in languages" :key="name">{{ name }}</option></select></label>
        <label>{{ t('source_language', 'Source language') }} <select v-model="source"><option>auto</option><option v-for="name in languages" :key="name">{{ name }}</option></select></label>
        <template v-if="settings.provider !== 'dictionary'">
          <label v-for="field in providerConfig.fields || []" :key="field.key">{{ field.key }} <input v-model="providerConfig.options[field.key]" :type="field.secret ? 'password' : 'text'" autocomplete="new-password" :placeholder="providerConfig.configured_fields?.includes(field.key) ? t('configured', 'Configured; leave blank to keep') : ''"></label>
          <template v-if="!providerConfig.fields?.length">
          <label v-if="settings.provider !== 'mbart50'">Endpoint <input v-model="providerConfig.endpoint" placeholder="https://…/v1"></label>
          <label v-if="settings.provider !== 'mbart50'">API key <input v-model="providerConfig.key" type="password" autocomplete="new-password" :placeholder="providerConfig.configured ? t('configured', 'Configured; leave blank to keep') : ''"></label>
          <label v-if="settings.provider === 'openai'">Model <input v-model="providerConfig.model" placeholder="OpenAI-compatible model ID"></label>
          <label v-if="settings.provider === 'microsoft'">Region <input v-model="providerConfig.region"></label>
          <label v-if="settings.provider === 'mbart50'">Local MBart50 directory <input v-model="providerConfig.model_path"></label>
          </template>
          <label v-if="settings.provider !== 'mbart50'"><input v-model="providerConfig.clear_key" type="checkbox">{{ t('clear_key', 'Clear saved API key') }}</label>
          <button type="button" @click="saveProvider">{{ t('save', 'Save') }}</button>
        </template>
        <p>{{ t('provider_notice', 'External translation sends selected text to the configured service. Dictionary translation stays local. MBart50 uses an existing local model on CPU and does not download weights.') }}</p>
      </template>
      <template v-else-if="panel === 'generate'">
        <h4>{{ t('chatgpt', 'AI prompt') }}</h4>
        <textarea v-model="description" rows="3" :aria-label="t('describe', 'Image description')"></textarea>
        <button type="button" :disabled="busy || !description.trim()" @click="generate">{{ t('generate', 'Generate prompt') }}</button>
        <textarea v-if="generated" v-model="generated" rows="4" :aria-label="t('result', 'Generated prompt')"></textarea>
        <button v-if="generated" type="button" @click="add(generated)">{{ t('add', 'Add') }}</button>
        <p>{{ t('ai_notice', 'Configure the OpenAI-compatible provider first. Review the generated prompt before adding it.') }}</p>
      </template>
    </div>
    <footer><small :title="t('token_notice', 'Text only: excludes resolved wildcards, styles and loaded embedding vectors.')">{{ tokenLabel }}</small><span role="status">{{ message }}</span></footer>
  </section>
</template>

<script setup>
import {computed, ref, shallowRef, onMounted, onBeforeUnmount, watch, nextTick} from 'vue';
import Sortable from 'sortablejs';
import NumberInput from './NumberInput.vue';
import i18n from '../i18n.json';
import {parse, edit, serialize, snapshot, restore, weight, setWeight, translationText, applyTranslation, append} from '../parser.js';
const props = defineProps(['area', 'scope', 'configuration', 'request', 'post', 'write', 'setSetting']);
const settings = computed(() => props.configuration.settings);
const scope = props.scope;
const {request, post, setSetting} = props;
const tags = ref(parse(props.area.value)), draft = ref(props.area.value), addition = ref(''), view = ref('text');
const textArea = ref(), appendArea = ref(), tagList = ref(), selected = ref(new Set()), panel = ref(''), filter = ref(''), entries = ref([]), groups = shallowRef([]);
const message = ref(''), busy = ref(false), providerConfig = ref({}), description = ref(''), generated = ref(''), tokens = ref(null), caps = ref({});
const source = computed({get: () => settings.value.source_language, set: value => setSetting('source_language', value)});
const translations = ref(new Map());
const providers = ['dictionary', 'openai', 'deepl', 'libretranslate', 'microsoft', 'mbart50', 'google', 'baidu', 'amazon', 'alibaba', 'yandex', 'youdao', 'tencent', 'mymemory', 'niutrans', 'caiyun', 'volcengine', 'iflytekV1', 'iflytekV2'];
const languages = ['zh', 'en', 'ja', 'ko', 'de', 'fr', 'es', 'it', 'pt', 'ru'];
let alive = true, revision = 0, composing = false, lastWritten = props.area.value, historyTimer, tokenTimer, sort, lastTokenKey, groupsGeneration = 0;
const attachments = new Map();
function t(key, fallback = key) {
  return i18n.languages.find(lang => lang.code === settings.value.language)?.lang[key] || i18n.languages.find(lang => lang.code === 'en_US')?.lang[key] || fallback;
}
function report(error) { if (alive && error.name !== 'AbortError') message.value = error.message; }
function allowed(text) { return !settings.value.blacklist.split('\n').map(tag => tag.trim().replaceAll('_', ' ').toLowerCase()).filter(Boolean).includes(text.trim().replaceAll('_', ' ').toLowerCase()); }
function payload(favorite = false, onlySelected = false) {
  const current = onlySelected ? tags.value.filter(tag => selected.value.has(tag.id)) : tags.value;
  return {scope: scope.key, prompt: serialize(current), tags: snapshot(current), favorite, limit: settings.value.history_limit};
}
function record() {
  clearTimeout(historyTimer);
  if (!settings.value.history || !settings.value.enabled || !draft.value.trim()) return;
  historyTimer = setTimeout(() => { if (alive && settings.value.history && settings.value.enabled) post('entries', payload()).catch(report); }, 1000);
}
function scheduleTokens() {
  clearTimeout(tokenTimer);
  tokenTimer = setTimeout(async () => {
    const currentCaps = window.FooocusTagComplete?.capabilities || {}, text = serialize(tags.value);
    const key = JSON.stringify([text, currentCaps.base, currentCaps.refiner]);
    if (key === lastTokenKey || !alive) return;
    const currentRevision = revision;
    try {
      const result = await post('tokens', {text, base: currentCaps.base || '', refiner: currentCaps.refiner || 'None'});
      if (alive && currentRevision === revision) { tokens.value = result; lastTokenKey = key; }
    } catch (error) { if (alive) tokens.value = null; report(error); }
  }, 600);
}
function commit() {
  if (!alive) return;
  revision++; draft.value = serialize(tags.value); lastWritten = draft.value; props.write(draft.value);
  record(); scheduleTokens();
  nextTick(cleanAttachments);
}
function onText(event) {
  draft.value = event.target.value;
  if (composing) { lastWritten = draft.value; props.write(draft.value); return; }
  tags.value = edit(draft.value, tags.value); selected.value.clear(); commit();
}
function compositionEnd(event) { composing = false; onText(event); }
function editTag(tag, event) { tag.raw = event.target.value; translations.value.delete(tag.id); commit(); }
function sync() {
  if (!alive || composing || props.area.value === lastWritten) return;
  lastWritten = props.area.value; draft.value = lastWritten; tags.value = parse(lastWritten); selected.value.clear(); revision++;
  record(); scheduleTokens(); nextTick(cleanAttachments);
}
function flush() { if (alive && props.area.value !== lastWritten) sync(); }
function select(id) { selected.value.has(id) ? selected.value.delete(id) : selected.value.add(id); }
function selectAll() { selected.value = selected.value.size === tags.value.length ? new Set() : new Set(tags.value.map(tag => tag.id)); }
function changeWeight(tag, value) { tag.raw = setWeight(tag.raw, value); commit(); }
function toggle(tag) { tag.disabled = !tag.disabled; commit(); }
function remove(tag) { tags.value = tags.value.filter(item => item.id !== tag.id); selected.value.delete(tag.id); commit(); }
function blacklistTag(tag) { setSetting('blacklist', [settings.value.blacklist, translationText(tag.raw) || tag.raw].filter(Boolean).join('\n')); remove(tag); }
function batch(action) {
  for (const tag of tags.value) if (selected.value.has(tag.id)) {
    if (action === 'increase' || action === 'decrease') tag.raw = setWeight(tag.raw, weight(tag.raw).value + (action === 'increase' ? 1 : -1) * settings.value.weight_step);
    if (action === 'disable' || action === 'enable') tag.disabled = action === 'disable';
  }
  if (action === 'remove') { tags.value = tags.value.filter(tag => !selected.value.has(tag.id)); selected.value.clear(); }
  commit();
}
async function setView(value) { view.value = value; await nextTick(); if (value === 'text') textArea.value?.focus(); }
function appendEnter(event) { if (event.isComposing || event.keyCode === 229 || event.shiftKey || event.ctrlKey || event.metaKey) return; event.preventDefault(); addText(); }
function addText() { const text = addition.value; if (text.trim()) { addition.value = ''; add(text); } }
async function add(text) {
  if (tags.value.some(tag => tag.metadata)) { message.value = t('metadata_notice', 'Import generation metadata before editing tags.'); return; }
  const extra = parse(text).filter(tag => !tag.raw.trim() || allowed(translationText(tag.raw) || tag.raw));
  if (!extra.some(tag => tag.raw.trim())) { message.value = t('blacklisted', 'Tag is blacklisted.'); return; }
  const existing = new Set(tags.value.map(tag => tag.id));
  append(tags.value, serialize(extra), settings.value.separator === 'newline' ? '\n' : ', '); commit();
  if (settings.value.auto_translate) await translateTags(true, tags.value.filter(tag => !existing.has(tag.id)).map(tag => tag.id));
}
function attach(area, onCommit, appendSeparator = true) {
  if (!area || attachments.has(area)) return;
  const detach = window.FooocusTagComplete?.attach(area, {role: scope.role, negative: scope.negative, onCommit, appendSeparator, filter: result => allowed(result.name)});
  if (detach) attachments.set(area, detach);
}
function attachTag(area, tag) { attach(area, value => { tag.raw = value; commit(); }, false); }
function cleanAttachments() { for (const [area, detach] of attachments) if (!area.isConnected) { detach(); attachments.delete(area); } }
function translation(tag) {
  const text = translationText(tag.raw);
  return translations.value.get(tag.id) || window.FooocusTagComplete?.translation(text) || groupTranslations.value.get(text.replaceAll('_', ' ').toLowerCase()) || '';
}
const groupTranslations = computed(() => new Map(allGroups.value.flatMap(group => Object.entries(group.tags)).map(([tag, label]) => [tag.replaceAll('_', ' ').toLowerCase(), label])));
async function loadGroups() {
  const generation = ++groupsGeneration;
  try {
    const categories = await request('groups?language=' + encodeURIComponent(settings.value.language));
    if (alive && generation === groupsGeneration) groups.value = categories.flatMap(category => category.groups.map(group => ({...group, name: category.name + ' / ' + group.name})));
  } catch (error) { report(error); }
}
const allGroups = computed(() => {
  let custom = []; try { custom = JSON.parse(settings.value.custom_groups); } catch (_) { /* Validated in settings. */ }
  return [...custom.filter(group => typeof group?.name === 'string' && group.tags && typeof group.tags === 'object'), ...groups.value];
});
const normalize = value => String(value).replaceAll('_', ' ').toLowerCase();
const filteredGroups = computed(() => allGroups.value.map(group => ({...group, tags: Object.fromEntries(Object.entries(group.tags).filter(([tag, label]) => allowed(tag) && normalize(tag + ' ' + (label || '')).includes(normalize(filter.value))))})).filter(group => Object.keys(group.tags).length));
const filteredEntries = computed(() => entries.value.filter(entry => (entry.name + ' ' + entry.prompt).toLowerCase().includes(filter.value.toLowerCase())));
const models = computed(() => {
  // capabilities ref makes computed refresh on model-context changes.
  const capabilities = caps.value, catalog = window.FooocusTagComplete?.catalog || {};
  return [['lora', 'loras'], ['embedding', 'embeddings'], ['wildcard', 'wildcards']].flatMap(([kind, key]) => kind === 'embedding' && capabilities.embeddings === false ? [] : (catalog[key] || []).map(model => ({...model, kind}))).filter(model => model.name.toLowerCase().includes(filter.value.toLowerCase()));
});
function previewUrl(id) { return new URL('./tagcomplete/v1/asset/' + encodeURIComponent(id), window.location.href).href; }
function addModel(model) {
  if (model.kind === 'embedding' && window.FooocusTagComplete?.capabilities?.embeddings === false) return;
  add(model.kind === 'lora' ? `<lora:${model.name}:${model.weight ?? 1}>` : model.kind === 'embedding' ? `embedding:${model.name}` : `__${model.name}__`);
}
async function translateTags(toEnglish, ids) {
  if (busy.value) return;
  const subset = tags.value.filter(tag => !tag.disabled && translationText(tag.raw) && (ids ? ids.includes(tag.id) : !selected.value.size || selected.value.has(tag.id)));
  if (!subset.length) return;
  const currentRevision = revision; busy.value = true; message.value = '';
  try {
    const values = [];
    for (let start = 0; start < subset.length; start += 100) {
      const result = await post('translate', {provider: settings.value.provider, texts: subset.slice(start, start + 100).map(tag => translationText(tag.raw)), source: source.value, target: toEnglish ? 'en' : settings.value.target_language, language: settings.value.language});
      if (!alive || revision !== currentRevision) return;
      values.push(...result.texts);
    }
    subset.forEach((tag, index) => { if (toEnglish) tag.raw = applyTranslation(tag.raw, values[index]); else translations.value.set(tag.id, values[index]); });
    if (toEnglish) commit();
  } catch (error) { report(error); } finally { if (alive) busy.value = false; }
}
async function copy() { try { await navigator.clipboard.writeText(draft.value); message.value = t('success'); } catch (error) { report(error); } }
async function saveFavorite(onlySelected = false) { try { await post('entries', payload(true, onlySelected)); message.value = t('success'); if (panel.value === 'favorites') loadEntries(); } catch (error) { report(error); } }
async function loadEntries() { try { const current = panel.value; const result = await request(`entries?scope=${scope.key}&favorite=${current === 'favorites'}`); if (alive && panel.value === current) entries.value = result; } catch (error) { report(error); } }
function useEntry(entry) { tags.value = entry.tags.length ? restore(entry.tags) : parse(entry.prompt); selected.value.clear(); commit(); }
async function favoriteEntry(entry) { try { await post('entries', {scope: scope.key, prompt: entry.prompt, tags: entry.tags, favorite: true, name: entry.name}); message.value = t('success'); } catch (error) { report(error); } }
async function rename(entry, name) { try { await post('entries/' + entry.id, {name}, 'PATCH'); entry.name = name; } catch (error) { report(error); } }
async function moveEntry(entry, index, direction = -1) {
  const items = filteredEntries.value, target = index + direction;
  const neighbor = items[target]?.position;
  if (neighbor === undefined) return;
  const boundary = items[target + direction]?.position ?? Math.max(0, neighbor - direction * 2);
  try { await post('entries/' + entry.id, {position: (neighbor + boundary) / 2}, 'PATCH'); await loadEntries(); } catch (error) { report(error); }
}
async function deleteEntry(entry) { try { await request(`entries?scope=${scope.key}&favorite=${panel.value === 'favorites'}&identifier=${entry.id}`, {method: 'DELETE'}); await loadEntries(); } catch (error) { report(error); } }
async function clearEntries() { if (!window.confirm(t('clear_history_confirm', 'Clear this list?'))) return; try { await request(`entries?scope=${scope.key}&favorite=${panel.value === 'favorites'}`, {method: 'DELETE'}); await loadEntries(); } catch (error) { report(error); } }
async function loadProvider() { try { const provider = settings.value.provider; const config = await request('providers/' + provider); if (alive && settings.value.provider === provider) providerConfig.value = {...config, key: '', clear_key: false}; } catch (error) { report(error); } }
async function saveProvider() { try { const {configured, configured_fields, fields, ...config} = providerConfig.value; const result = await post('providers/' + settings.value.provider, config, 'PUT'); providerConfig.value = {...result, key: '', clear_key: false}; message.value = t('success'); } catch (error) { report(error); } }
async function generate() { if (busy.value) return; busy.value = true; try { const result = await post('generate', {text: description.value}); if (alive) generated.value = result.prompt; } catch (error) { report(error); } finally { if (alive) busy.value = false; } }
function openPanel(name) { panel.value = panel.value === name ? '' : name; filter.value = ''; if (name === 'history' || name === 'favorites') loadEntries(); if (name === 'provider') loadProvider(); if (name === 'models') caps.value = {...window.FooocusTagComplete?.capabilities}; }
function shortcut(event) {
  if (!settings.value.hotkeys || event.isComposing || event.keyCode === 229) return;
  if ((event.ctrlKey || event.metaKey) && event.shiftKey && event.key.toLowerCase() === 'f') { event.preventDefault(); saveFavorite(); }
  if (event.altKey && ['ArrowUp', 'ArrowDown'].includes(event.key) && selected.value.size) { event.preventDefault(); batch(event.key === 'ArrowUp' ? 'increase' : 'decrease'); }
  if (event.key === 'Escape' && panel.value) { panel.value = ''; }
}
const tokenLabel = computed(() => tokens.value ? Object.entries(tokens.value).map(([role, counts]) => `${role}: ` + (counts.kind === 'anima' ? `Qwen ${counts.qwen} · T5 ${counts.t5}` : `CLIP ${counts.clip}/${counts.chunk_size}`)).join(' | ') : '');
function modelChanged(event) { caps.value = {...event.detail}; lastTokenKey = null; scheduleTokens(); }
function catalogChanged() { caps.value = {...window.FooocusTagComplete?.capabilities}; }
watch(() => settings.value.history, value => { if (!value) clearTimeout(historyTimer); });
watch(() => settings.value.language, loadGroups);
watch(() => settings.value.target_language, () => translations.value.clear());
watch(() => settings.value.provider, () => { if (panel.value === 'provider') loadProvider(); });
onMounted(() => {
  caps.value = {...window.FooocusTagComplete?.capabilities};
  attach(textArea.value, value => { tags.value = edit(value, tags.value); commit(); });
  attach(appendArea.value, value => { addition.value = value; });
  props.area.addEventListener('input', sync);
  window.addEventListener('fooocus-model-capabilities', modelChanged);
  window.addEventListener('fooocus-completion-catalog', catalogChanged);
  sort = Sortable.create(tagList.value, {animation: 120, handle: '.pai-drag', onEnd() {
    const order = Array.from(tagList.value.children, node => Number(node.dataset.id));
    const map = new Map(tags.value.map(tag => [tag.id, tag]));
    tags.value = order.map(id => map.get(id)).filter(Boolean);
    for (let i = 0; i < tags.value.length - 1; i++) if (!tags.value[i].separator) tags.value[i].separator = settings.value.separator === 'newline' ? '\n' : ', ';
    if (tags.value.length) tags.value.at(-1).separator = '';
    commit();
  }});
  loadGroups(); scheduleTokens();
});
onBeforeUnmount(() => {
  alive = false; clearTimeout(historyTimer); clearTimeout(tokenTimer); sort?.destroy();
  props.area.removeEventListener('input', sync); window.removeEventListener('fooocus-model-capabilities', modelChanged);
  window.removeEventListener('fooocus-completion-catalog', catalogChanged);
  for (const detach of attachments.values()) detach(); attachments.clear();
});
defineExpose({sync, flush});
</script>
