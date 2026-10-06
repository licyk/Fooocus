// Tag matching stays off the UI thread; no server call is made for tag keystrokes.
importScripts(new URL('engine.js', self.location.href).href);
let index = [], settings = {}, capabilities = {}, usage = [];
self.onmessage = ({data}) => {
    try {
        if (data.type === 'init') {
            settings = data.settings; capabilities = data.capabilities; usage = data.usage;
            index = FooocusTagEngine.buildIndex(data.datasets, data.catalog, settings);
            self.postMessage({type: 'ready', id: data.id, count: index.length,
                translations: index.filter(x => x.translation).map(x => [x.normalized, x.translation])});
        } else if (data.type === 'query') {
            const results = FooocusTagEngine.search(index, data.context, settings, capabilities, usage, data.wildcardValues);
            self.postMessage({type: 'results', id: data.id, results});
        } else if (data.type === 'usage') usage = data.usage;
        else if (data.type === 'capabilities') capabilities = data.capabilities;
    } catch (error) {
        self.postMessage({type: 'error', id: data.id, message: error.message});
    }
};
