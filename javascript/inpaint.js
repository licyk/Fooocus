// Gradio's Pixi editor renders background and painted layers in one canvas.
// Change layer alpha only during screen rendering; restore it before exporting
// masks, drawing into textures, or serializing the generation inputs.
(() => {
    const apps = new Set();
    let transparency = 50;
    let lastApp = window.__PIXI_APP__;

    function install(app) {
        if (!app?.renderer || app.destroyed || app.__fooocusMaskTransparency) return;
        const canvas = app.canvas;
        if (!canvas?.closest('#inpaint_canvas, #inpaint_mask_canvas')) return;
        const render = app.renderer.render;
        app.renderer.render = function(options, ...args) {
            const target = options?.container || options;
            if (target !== app.stage || options?.target) {
                return render.call(this, options, ...args);
            }
            const image = app.stage.children.find(child => child.sortableChildren);
            const layers = (image?.children || []).filter(child => child.zIndex >= 0);
            const alpha = layers.map(layer => layer.alpha);
            try {
                layers.forEach(layer => {layer.alpha *= 1 - transparency / 100;});
                return render.call(this, options, ...args);
            } finally {
                layers.forEach((layer, index) => {layer.alpha = alpha[index];});
            }
        };
        app.__fooocusMaskTransparency = true;
        canvas.dataset.fooocusMaskTransparency = String(transparency);
    }

    // Gradio exposes each normal editor application here when initializing it.
    // Keep the original behavior for other components and capture all editors.
    const descriptor = Object.getOwnPropertyDescriptor(window, '__PIXI_APP__');
    if (!descriptor || descriptor.configurable) {
        Object.defineProperty(window, '__PIXI_APP__', {
            configurable: true,
            get() {return descriptor?.get ? descriptor.get.call(window) : lastApp;},
            set(app) {
                lastApp = app;
                descriptor?.set?.call(window, app);
                apps.add(app);
            }
        });
    }
    if (lastApp) apps.add(lastApp);

    function refresh() {
        const input = document.querySelector('#inpaint_mask_transparency input[type="number"]');
        if (input && Number.isFinite(Number(input.value))) {
            transparency = Math.min(100, Math.max(0, Number(input.value)));
        }
        for (const app of apps) {
            if (app.destroyed || !app.stage) {apps.delete(app); continue;}
            if (!app.renderer) continue;
            install(app);
            if (app.__fooocusMaskTransparency) {
                if (app.canvas.dataset.fooocusMaskTransparency !== String(transparency)) {
                    app.canvas.dataset.fooocusMaskTransparency = String(transparency);
                }
                app.render();
            }
        }
    }
    window.setInpaintMaskTransparency = value => {
        transparency = Math.min(100, Math.max(0, Number(value)));
        refresh();
    };
    onUiLoaded(refresh);
    onAfterUiUpdate(refresh);
})();
