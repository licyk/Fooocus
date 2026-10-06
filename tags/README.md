# Completion data

Bundled files are copied from `a1111-sd-webui-tagcomplete` at commit
`4170882f90b47be130a0ff9314f663c230b9153d`:

- `danbooru.csv`: Danbooru tag names, categories, counts and aliases.
- `extra-quality-tags.csv`: extra quality tags.
- `demo-chants.json`: example prompt snippets.
- `TAGCOMPLETE_LICENSE`: the upstream MIT license, Copyright (c) 2022 Dominik Reh.
- `CARET_LICENSE`: the MIT license from `component/textarea-caret-position`,
  Copyright (c) 2015 Jonathan Ong.

The port's CSV/search/insertion behavior is based on the same extension.
`javascript/tagcomplete/caret.js` adapts its `javascript/_caretPosition.js`,
which attributes the MIT-licensed `textarea-caret-position` project.

See [the prompt assistance guide](../tagcomplete.md) for configuration,
translation CSV formats and custom Chants. Completion reads its datasets locally
and performs no downloads on startup.
