import {defineConfig} from 'vite';
import vue from '@vitejs/plugin-vue';
import {fileURLToPath, URL} from 'node:url';
export default defineConfig({
    plugins: [vue()],
    resolve: {alias: {'@': fileURLToPath(new URL('./src', import.meta.url))}},
    define: {'process.env.NODE_ENV': JSON.stringify('production')},
    build: {
        outDir: '../../javascript/prompt_all_in_one', emptyOutDir: false,
        lib: {entry: 'src/main.js', name: 'FooocusPromptEditor', formats: ['iife'], fileName: () => 'editor.js'},
        cssCodeSplit: false,
        rollupOptions: {output: {assetFileNames: 'editor.[ext]'}},
    },
});
