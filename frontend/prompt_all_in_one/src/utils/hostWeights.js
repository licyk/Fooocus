// Rendering adapter only: tag splitting and editing still use upstream source.
// Fooocus recognizes parentheses; preserve upstream bracket controls in the UI.
export default function hostWeights(text, novelAi = false) {
    const stack = [{open: '', text: ''}]
    const pairs = {'(': ')', '[': ']', '{': '}'}
    for (let index = 0; index < text.length; index++) {
        const char = text[index]
        const frame = stack[stack.length - 1]
        if (char === '\\' && index + 1 < text.length) {
            frame.text += char + text[++index]
        } else if (char === '(' || char === '[' || (novelAi && char === '{')) {
            stack.push({open: char, text: ''})
        } else if (stack.length > 1 && char === pairs[frame.open]) {
            stack.pop()
            let rendered = frame.open + frame.text + char
            if (frame.open !== '(' && !frame.text.includes('|') && (frame.text.match(/:/g) || []).length < 2) {
                let inner = frame.text
                let weight = frame.open === '[' ? 1 / 1.1 : 1.05
                // Fooocus's explicit numeric weight overrides the outer weight.
                // Collapse simple nested weights instead of losing a bracket layer.
                const numeric = inner.match(/^\(([^()]*):([+-]?(?:\d+\.?\d*|\.\d+))\)$/) || inner.match(/^([^()]*):([+-]?(?:\d+\.?\d*|\.\d+))$/)
                if (numeric) {inner = numeric[1]; weight *= Number(numeric[2])}
                rendered = '(' + inner + ':' + Number(weight.toFixed(6)) + ')'
            }
            stack[stack.length - 1].text += rendered
        } else {
            frame.text += char
        }
    }
    while (stack.length > 1) {
        const frame = stack.pop()
        stack[stack.length - 1].text += frame.open + frame.text
    }
    return stack[0].text
}
