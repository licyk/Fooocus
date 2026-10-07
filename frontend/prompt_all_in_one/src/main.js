import {createApp, reactive} from 'vue';
import PromptEditor from './components/phystonPrompt.vue';
import './style.css';
export function mount(node, options) {
    const configuration = reactive({settings: options.settings});
    const {settings, ...host} = options;
    const app = createApp(PromptEditor, {...host, configuration});
    const editor = app.mount(node);
    return {
        sync() { editor?.sync(); },
        configure(settings) { if (configuration.settings !== settings) configuration.settings = settings; },
        flush() { editor?.flush(); },
        destroy() { app.unmount(); },
    };
}
