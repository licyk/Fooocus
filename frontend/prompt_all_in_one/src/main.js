import '../styles/host.css'
import '../styles/main.less'
import '../styles/animate.min.css'
import 'toastr/build/toastr.min.css'
import 'tippy.js/dist/tippy.css'
import 'tippy.js/themes/light.css'
import 'vue3-colorpicker/style.css'
import './style.css'
import * as Vue from 'vue'
import App from './App.vue'
import toastr from 'toastr'
import VueClipboard from 'vue-clipboard3'
const {toClipboard} = VueClipboard()
import CommonMixin from "@/mixins/commonMixin"
import common from "@/utils/common"
import tippy from "tippy.js"

export function mount(div, host) {
    host = Vue.shallowReactive({...host, prompts: host.prompts.map(item => Vue.shallowReactive(item))})
    const app = Vue.createApp(App, {host})

    app.config.globalProperties.$appMode   = typeof APP_MODE !== 'undefined' && APP_MODE
    app.config.globalProperties.$toastr = Object.fromEntries(['error','success','info','warning'].map(name => [name, (...args) => {if (!host.abort.signal.aborted) return toastr[name](...args)}]))
    app.config.globalProperties.$copyText  = toClipboard
    app.config.globalProperties.$tippyList = []
    app.config.globalProperties.$fooocusHost = host
    app.mixin(CommonMixin)
    app.directive('animate', {
        mounted(el, binding) {
            el.classList.add('animate__animated', `animate__${binding.value}`, 'animate__faster')
        },
        updated(el, binding) {
            el.classList.remove(`animate__${binding.oldValue}`)
            el.classList.add(`animate__${binding.value}`)
        }
    })
    app.directive('tooltip', {
        mounted(el, binding) {
            // data-tippy-content
            el.setAttribute('data-tippy-content', binding.value)
            // 如果有 unaffected 属性，则不受 localStorage 控制
            let unaffected = el.getAttribute('unaffected')
            unaffected     = unaffected === null ? false : unaffected === 'true'
            const instance = tippy(el, {
                placement: 'bottom',
                theme: 'light',
                allowHTML: true,
                onCreate(instance, partialProps) {
                    if (!binding.value) return
                    if (unaffected) return
                    instance.enable()
                    const enable = host.tooltip !== false
                    if (!enable) {
                        instance.disable()
                    }
                },
            })
            el.$tippyInstance = instance
            if (!unaffected) app.config.globalProperties.$tippyList.push(instance)
        },
        unmounted(el) {el.$tippyInstance?.destroy()},
        updated(el, binding) {
            if (!binding.value) {
                el.$tippyInstance.disable()
            } else {
                el.setAttribute('data-tippy-content', binding.value)
                el.$tippyInstance.setContent(binding.value)
            }
        }
    })

    const vm = app.mount(div)
    return {vm,
        async setPrompts(prompts) {
            const added = prompts.filter(item => !host.prompts.some(old => old.id === item.id))
            host.prompts = prompts.map(item => host.prompts.find(old => old.id === item.id) || Vue.shallowReactive(item))
            await Vue.nextTick()
            for (const entry of added) {
                const item = host.prompts.find(item => item.id === entry.id)
                const fields = ['hideDefaultInput','autoLoadWebuiPrompt','hidePanel','hideGroupTags']
                const values = await vm.gradioAPI.getDatas(fields.map(key => item[key + 'Key']))
                if (host.abort.signal.aborted) return
                fields.forEach(key => {if (values[item[key + 'Key']] !== null) item[key] = values[item[key + 'Key']]})
                item.$prompt.classList.toggle('pai-native-hidden', item.hideDefaultInput)
                item.mount.classList.toggle('dark', vm.theme === 'dark'); item.mount.classList.toggle('light', vm.theme === 'light')
            }
            for (const [name, key] of [['history','histories'], ['favorite','favorites']]) {
                const panel = vm.$refs[name]
                panel[key] = host.prompts.map(item => panel[key].find(old => old.key === item.historyKey) || {name: item.historyKey, type: item.neg ? 'negative_prompt' : 'prompt', key: item.historyKey, list: []})
            }
        },
        setSetting(key, value) {
            const fields = {language:'languageCode', provider:'translateApi', auto_translate:'autoTranslateToEnglish'}
            if (fields[key]) vm[fields[key]] = value === 'mymemory' ? 'myMemory' : value
            if (key === 'hide_native') host.prompts.forEach(item => {item.hideDefaultInput = value})
        },
        destroy() {app.unmount(); app.config.globalProperties.$tippyList.forEach(t => t.destroy())},
        configure(settings) {host.settings = settings; vm.theme = settings.theme === 'auto' ? vm.theme : settings.theme},
        flush() {Object.values(vm.$refs).flat().forEach(item => {
            if (!item?.textarea || !Array.isArray(item.tags)) return
            for (const tag of item.tags) {
                if (item.editing[tag.id]) {
                    const input = item.$refs['promptTagEdit-' + tag.id]?.[0]
                    if (input) item._changeTagValue(tag, input.value)
                }
            }
        })}}
}
