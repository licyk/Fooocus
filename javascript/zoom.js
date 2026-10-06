// Use the native ImageEditor controls so zooming and drawing share coordinates.
onUiLoaded(() => {
    let activeEditor = null;
    const editorSelector = "#inpaint_canvas, #inpaint_mask_canvas";
    document.addEventListener("pointerover", (event) => {
        const editor = event.target.closest(editorSelector);
        if (editor) activeEditor = editor;
    });

    function clickControl(label) {
        if (!activeEditor || !uiElementIsVisible(activeEditor)) return false;
        const button = activeEditor.querySelector(`button[aria-label="${label}"]`);
        if (!button || button.disabled) return false;
        button.click();
        return true;
    }

    document.addEventListener("wheel", (event) => {
        const editor = event.target.closest(editorSelector);
        if (!editor || !event.shiftKey) return;
        activeEditor = editor;
        if (clickControl(event.deltaY < 0 ? "Zoom in" : "Zoom out")) {
            event.preventDefault();
            event.stopPropagation();
        }
    }, {passive: false, capture: true});

    document.addEventListener("keydown", (event) => {
        if (event.target.matches("input, textarea, [contenteditable=true]")) return;
        let handled = false;
        if ((event.ctrlKey || event.metaKey) && event.code === "KeyZ") {
            handled = clickControl(event.shiftKey ? "Redo" : "Undo");
        } else if ((event.ctrlKey || event.metaKey) && event.code === "KeyY") {
            handled = clickControl("Redo");
        } else if (!event.ctrlKey && !event.metaKey && !event.altKey) {
            if (event.code === "KeyF" && !event.repeat) handled = clickControl("Pan");
            if (event.code === "KeyS" && activeEditor) {
                if (document.fullscreenElement) document.exitFullscreen();
                else activeEditor.requestFullscreen();
                handled = true;
            }
        }
        if (handled) {
            event.preventDefault();
            event.stopImmediatePropagation();
        }
    }, {capture: true});
    document.addEventListener("keyup", (event) => {
        if (event.code === "KeyF" && !event.target.matches("input, textarea")) clickControl("Brush");
    });
});
