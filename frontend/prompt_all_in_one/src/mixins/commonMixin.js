import common from "@/utils/common";
import GradioAPI from "@/utils/gradioAPI";

export default {
    data() {
        return {
            /**
             * @type {GradioAPI}
             */
            gradioAPI: null,
        }
    },
    beforeMount() {
        this.gradioAPI = new GradioAPI(this.$fooocusHost)
        this._fooocusTimers = new Set()
        this._fooocusIntervals = new Set()
        this._fooocusListeners = []
    },
    beforeUnmount() {
        this._fooocusDisposed = true
        this._fooocusTimers.forEach(clearTimeout)
        this._fooocusIntervals.forEach(clearInterval)
        this._fooocusListeners.forEach(([target, name, handler, options]) => target.removeEventListener(name, handler, options))
    },
    methods: {
        $clearTimeout(id) {clearTimeout(id); this._fooocusTimers.delete(id)},
        $clearInterval(id) {clearInterval(id); this._fooocusIntervals.delete(id)},
        $setTimeout(callback, delay, ...args) {
            const id = setTimeout(() => {this._fooocusTimers.delete(id); if (!this._fooocusDisposed) callback(...args)}, delay)
            this._fooocusTimers.add(id); return id
        },
        $setInterval(callback, delay, ...args) {
            const id = setInterval(() => {if (!this._fooocusDisposed) callback(...args)}, delay)
            this._fooocusIntervals.add(id); return id
        },
        $listen(target, name, handler, options) {
            target.addEventListener(name, handler, options)
            this._fooocusListeners.push([target, name, handler, options])
        },
        loadExtraNetworks() {return this._loadExtraNetworks()},
        _loadExtraNetworks() {
            this.gradioAPI.getExtraNetworks().then(res => {
                if (!res) return
                this.extraNetworks = res
                res.forEach(extraNetwork => {
                    if (extraNetwork.name === 'textual inversion') {
                        let list = {}
                        extraNetwork.items.forEach(item => {
                            list[item.name.toLowerCase()] = item.name
                        })
                        this.embeddings = list
                    } else if (extraNetwork.name === 'lora' || extraNetwork.name === 'lycoris') {
                        let list = {}
                        extraNetwork.items.forEach(item => {
                            list[item.name.toLowerCase()] = item.name
                            if (item.output_name) {
                                list[item.output_name.toLowerCase()] = item.name
                            }
                        })
                        if (extraNetwork.name === 'lora') {
                            this.loras = list
                        } else {
                            this.lycos = list
                        }
                    }
                })
            })
        },
        getExtraNetworkFullName(name, type = 'lora') {
            if (typeof this.extraNetworks !== 'object') return name
            for (let extraNetwork of this.extraNetworks) {
                if (extraNetwork.name !== type) continue
                const nameLowerCase = name.toLowerCase()
                for (let item of extraNetwork.items) {
                    if (item.name.toLowerCase() === nameLowerCase || item.output_name?.toLowerCase() === nameLowerCase) {
                        if (!item.civitai_info?.name) return name
                        if (item.civitai_info.model?.name && item.civitai_info.model.name !== item.civitai_info.name) {
                            return '[' + item.civitai_info.name + '] ' + item.civitai_info.model.name
                        } else {
                            return item.civitai_info.name
                        }
                    }
                }
            }
            return name
        },
        loraExists(name) {
            if (typeof this.loras !== 'object') return name
            return this.loras[name.toLowerCase()] ?? false
        },
        lycoExists(name) {
            if (typeof this.lycos !== 'object') return name
            return this.lycos[name.toLowerCase()] ?? false
        },
        embeddingExists(name) {
            if (typeof this.embeddings !== 'object') return name
            return this.embeddings[name.replace(/^embedding:/i, '').toLowerCase()] ?? false
        },
    }
}
