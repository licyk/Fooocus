<template>
    <div v-show="svgCode" :class="['icon-svg', 'icon-svg-' + name]" data-name="name" v-html="svgCode"></div>
</template>
<script>
const icons = import.meta.glob('../../styles/icons/*.svg', {query: '?raw', import: 'default', eager: true})
export default {
    name: 'IconSvg',
    props: {
        name: {
            type: String,
            required: true
        },
        svgName: {
            type: String,
            default: ''
        },
    },
    data() {
        return {
            svgCode: ''
        }
    },
    watch: {
        name: {
            handler: function (val, oldVal) {
                this.getSvg()
            },
            immediate: false
        }
    },
    mounted() {
        this.getSvg()
    },
    methods: {
        getSvg() {
            this.svgCode = icons['../../styles/icons/' + (this.svgName || this.name) + '.svg'] || ''
        },
    }
}
</script>
