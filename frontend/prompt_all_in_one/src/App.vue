<template>
    <div :class="theme">
        <template v-for="(item) in prompts" :key="item.name">
            <Teleport :to="item.mount">
            <physton-prompt v-if="item.$textarea && initialized" :id="item.id" :ref="item.id" :name="item.name"
                            :neg="item.neg" :textarea="item.$textarea" :steps="item.$steps"
                            v-model:language-code="languageCode"
                            :translate-apis="translateApis" :languages="languages"
                            :history-key="item.historyKey"
                            @click:show-history="onShowHistory(item.id, $event)"
                            :favorite-key="item.favoriteKey"
                            @refresh-favorites="onRefreshFavorites"
                            @click:show-favorite="onShowFavorite(item.id, $event)"
                            v-model:can-one-translate="canOneTranslate"
                            v-model:auto-translate="autoTranslate"
                            v-model:auto-translate-to-english="autoTranslateToEnglish"
                            v-model:auto-translate-to-local="autoTranslateToLocal"
                            v-model:auto-remove-space="autoRemoveSpace"
                            v-model:auto-remove-last-comma="autoRemoveLastComma"
                            v-model:auto-keep-weight-zero="autoKeepWeightZero"
                            v-model:auto-keep-weight-one="autoKeepWeightOne"
                            v-model:auto-break-before-wrap="autoBreakBeforeWrap"
                            v-model:auto-break-after-wrap="autoBreakAfterWrap"
                            v-model:auto-remove-lora-before-comma="autoRemoveLoraBeforeComma"
                            v-model:auto-remove-lora-after-comma="autoRemoveLoraAfterComma"
                            v-model:use-novel-ai-weight-symbol="useNovelAiWeightSymbol"
                            v-model:auto-remove-before-line-comma="autoRemoveBeforeLineComma"
                            :hide-default-input="item.hideDefaultInput"
                            @update:hide-default-input="onUpdateHideDefaultInput(item.id, $event)"
                            :auto-load-webui-prompt="item.autoLoadWebuiPrompt"
                            @update:auto-load-webui-prompt="onUpdateAutoLoadWebuiPrompt(item.id, $event)"
                            :hide-panel="item.hidePanel"
                            @update:hide-panel="onUpdateHidePanel(item.id, $event)"
                            v-model:enable-tooltip="enableTooltip"
                            v-model:translate-api="translateApi"
                            :translate-api-config="translateApiConfig"
                            @click:translate-api="onTranslateApiClick"
                            @click:prompt-format="onPromptFormatClick"
                            @click:blacklist="onBlacklistClick"
                            @click:hotkey="onHotkeyClick"
                            v-model:tag-complete-file="tagCompleteFile"
                            v-model:only-csv-on-auto="onlyCsvOnAuto"
                            v-model:group-tags-translate="groupTagsTranslate"
                            @click:select-language="onSelectLanguageClick"
                            @click:select-theme="onSelectThemeClick"
                            @click:show-chatgpt="onShowChatgpt(item.id, $event)"
                            :extra-networks="extraNetworks"
                            :loras="loras"
                            :lycos="lycos"
                            :embeddings="embeddings"
                            :version="version"
                            :latest-version="latestVersion"
                            :is-latest-version="isLatestVersion"
                            @click:show-about="onShowAbout"
                            :theme="theme"
                            @click:switch-theme="onSwitchTheme"
                            :group-tags="groupTags"
                            :hide-group-tags="item.hideGroupTags"
                            v-model:group-tags-color="groupTagsColor"
                            :group-tags-color-key-cache="groupTagsColorKeyCache"
                            @update:hide-group-tags="onUpdateHideGroupTags(item.id, $event)"
                            :group-tags-translate-cache="groupTagsTranslateCache"
                            v-model:extra-networks-width="extraNetworksWidth"
                            v-model:extra-networks-height="extraNetworksHeight"
                            :blacklist="blacklist"
                            :cancel-blacklist-confirm="cancelBlacklistConfirm"
                            @update:blacklist="onUpdateBlacklist"
                            :hotkey="hotkey"
                            @show-extra-networks="onShowExtraNetworks"
                            @hide-extra-networks="onHideExtraNetworks"
                            @refresh-extra-networks="onRefreshExtraNetworks"
            ></physton-prompt>
            </Teleport>
        </template>
        <translate-setting ref="translateSetting" v-model:language-code="languageCode"
                           :translate-apis="translateApis" :languages="languages"
                           @forceUpdate:translateApi="updateTranslateApiConfig"
                           v-model:tag-complete-file="tagCompleteFile"
                           v-model:only-csv-on-auto="onlyCsvOnAuto"
                           v-model:group-tags-translate="groupTagsTranslate"
                           v-model:translate-api="translateApi"></translate-setting>
        <select-language ref="selectLanguage" v-model:language-code="languageCode"
                         :translate-apis="translateApis"
                         :languages="languages"
                         v-model:translate-api="translateApi"
                         v-model:tag-complete-file="tagCompleteFile"
                         v-model:only-csv-on-auto="onlyCsvOnAuto"></select-language>
        <prompt-format ref="promptFormat" v-model:language-code="languageCode"
                       :translate-apis="translateApis"
                       :languages="languages"
                       v-model:auto-remove-space="autoRemoveSpace"
                       v-model:auto-remove-last-comma="autoRemoveLastComma"
                       v-model:auto-keep-weight-zero="autoKeepWeightZero"
                       v-model:auto-keep-weight-one="autoKeepWeightOne"
                       v-model:auto-break-before-wrap="autoBreakBeforeWrap"
                       v-model:auto-break-after-wrap="autoBreakAfterWrap"
                       v-model:auto-remove-lora-before-comma="autoRemoveLoraBeforeComma"
                       v-model:auto-remove-lora-after-comma="autoRemoveLoraAfterComma"
                       v-model:use-novel-ai-weight-symbol="useNovelAiWeightSymbol"
                       v-model:auto-remove-before-line-comma="autoRemoveBeforeLineComma"
        ></prompt-format>
        <blacklist ref="blacklist" v-model:language-code="languageCode"
                   :translate-apis="translateApis"
                   :languages="languages"
                   @update:blacklist="onUpdateBlacklist"></blacklist>
        <hotkey ref="hotkey" v-model:language-code="languageCode"
                :translate-apis="translateApis"
                :languages="languages"
                :default-hotkey="hotkey"
                @update:hotkey="onUpdateHotkey"
        ></hotkey>
        <history ref="history" v-model:language-code="languageCode"
                 :translate-apis="translateApis" :languages="languages"
                 v-model:tag-complete-file="tagCompleteFile"
                 v-model:only-csv-on-auto="onlyCsvOnAuto"
                 @refresh-favorites="onRefreshFavorites"
                 @use="onUseHistory"/>
        <favorite ref="favorite" v-model:language-code="languageCode"
                  :translate-apis="translateApis" :languages="languages"
                  v-model:tag-complete-file="tagCompleteFile"
                  v-model:only-csv-on-auto="onlyCsvOnAuto"
                  @use="onUseFavorite"></favorite>
        <extension-css ref="extensionCss" v-model:language-code="languageCode"
                       :translate-apis="translateApis" :languages="languages"/>
        <packages-state ref="packagesState" v-model:language-code="languageCode"
                        :translate-apis="translateApis" :languages="languages"
                        @click:select-language="onSelectLanguageClick"
                        :packages-state="packagesState" :python="python"/>
        <chatgpt-prompt ref="chatgptPrompt" v-model:language-code="languageCode"
                        :translate-apis="translateApis" :languages="languages"
                        @use="onUseChatgpt" />
        <about ref="about" v-model:language-code="languageCode"
               :translate-apis="translateApis" :languages="languages" />

        <extra-networks-popup ref="extraNetworksPopup"
                              v-model:language-code="languageCode"
                              :translate-apis="translateApis" :languages="languages"
                              :extra-networks="extraNetworks" />

        <div class="physton-paste-popup" v-if="showPastePopup" @click="closePastePopup">
            <div class="paste-popup-main" @click.stop>
                <div class="paste-popup-close" @click="closePastePopup">
                    <icon-svg name="close"/>
                </div>
                <div class="paste-popup-title">{{ pasteTitle }}</div>
                <div class="paste-popup-body">
                    <textarea class="paste-content" v-model="pasteContent"
                              :placeholder="getLang('please_enter_the_content_here')"></textarea>
                    <div v-if="!pasteLoading" class="paste-submit" @click="onClickPasteSubmit">Submit</div>
                    <div v-else class="paste-submit">
                        <icon-svg name="loading"/>
                    </div>
                </div>
            </div>
        </div>
    </div>
</template>

<script>
import PhystonPrompt from "./components/phystonPrompt.vue"
import TranslateSetting from "@/components/translateSetting.vue";
import common from "@/utils/common";
import SelectLanguage from "@/components/selectLanguage.vue";
import Favorite from "@/components/favorite.vue";
import History from "@/components/history.vue";
import IconSvg from "@/components/iconSvg.vue";
import ExtensionCss from "@/components/extensionCss.vue";
import PromptFormat from "@/components/promptFormat.vue";
import Blacklist from "@/components/blacklist.vue";
import PackagesState from "@/components/packagesState.vue";
import ChatgptPrompt from "@/components/chatgptPrompt.vue";
import About from "@/components/about.vue";
import globals from "../globals";
import jsYaml from "js-yaml";
import {ref} from "vue";
import Hotkey from "@/components/hotkey.vue";
import ExtraNetworksPopup from "@/components/extraNetworksPopup.vue";
import waitTick from "@/utils/waitTick";

export default {
    name: 'App',
    components: {
        Hotkey,
        About,
        ChatgptPrompt,
        PackagesState,
        PromptFormat,
        Blacklist,
        ExtensionCss,
        IconSvg,
        History,
        Favorite,
        SelectLanguage,
        TranslateSetting,
        PhystonPrompt,
        ExtraNetworksPopup
    },
    mixins: [],
    props: {host: {type: Object, required: true}},
    computed: {prompts() {return this.host.prompts}},
    data() {
        return {
            initialized: false,
            languageCode: '',
            languages: {},
            translateApis: [],
            translateApi: '',
            translateApiConfig: {},
            canOneTranslate: false,
            autoTranslate: false,
            autoTranslateToEnglish: false,
            autoTranslateToLocal: false,
            autoRemoveSpace: true,
            autoRemoveLastComma: false,
            autoKeepWeightZero: false,
            autoKeepWeightOne: false,
            autoBreakBeforeWrap: false,
            autoBreakAfterWrap: false,
            autoRemoveLoraBeforeComma: false,
            autoRemoveLoraAfterComma: false,
            useNovelAiWeightSymbol: false,
            autoRemoveBeforeLineComma: false,
            // hideDefaultInput: false,
            enableTooltip: true,
            tagCompleteFile: '',
            onlyCsvOnAuto: true,

            startWatchSave: false,

            pasteBtn: null,
            showPastePopup: false,
            pasteTitle: '',
            pasteContent: '',
            pasteLoading: false,

            historyCurrentPrompt: '',
            favoriteCurrentPrompt: '',
            chatgptCurrentPrompt: '',

            extraNetworks: [],
            loras: [],
            lycos: [],
            embeddings: [],

            python: '',
            packagesState: [],

            version: '',
            latestVersion: '',
            isLatestVersion: true,

            theme: 'dark',

            groupTags: [],
            groupTagsColor: {},
            groupTagsColorKeyCache: {},
            groupTagsTranslate: true,
            groupTagsTranslateCache: {
                toEn: new Map(),
                toLocal: new Map()
            },
            extraNetworksWidth: 100,
            extraNetworksHeight: 120,

            blacklist: {},
            cancelBlacklistConfirm: false,

            hotkey: {
                click: 'edit', // edit, disable, extend
                dblClick: 'disable', // edit, disable, extend
                rightClick: '', // edit, disable, extend
                hover: 'extend', // extend
            }
        }
    },
    watch: {
        theme(value) {this.prompts.forEach(item => {item.mount.classList.toggle("dark", value === "dark"); item.mount.classList.toggle("light", value === "light")})},
        languageCode: {
            handler: function (val, oldVal) {
                if (!this.startWatchSave) return
                console.log('onLanguageCodeChange', val)
                this.canOneTranslate = common.canOneTranslate(this.languageCode)
                this.gradioAPI.setData('languageCode', val).then(data => {
                }).catch(err => {
                })
                waitTick.addWaitTick(() => this.loadGroupTags())
            },
            immediate: false,
        },
        autoTranslateToEnglish: {
            handler: function (val, oldVal) {
                if (!this.startWatchSave) return
                console.log('onAutoTranslateToEnglishChange', val)
                this.gradioAPI.setData('autoTranslateToEnglish', val).then(data => {
                }).catch(err => {
                })
            },
            immediate: false,
        },
        autoTranslateToLocal: {
            handler: function (val, oldVal) {
                if (!this.startWatchSave) return
                console.log('onAutoTranslateToLocalChange', val)
                this.gradioAPI.setData('autoTranslateToLocal', val).then(data => {
                }).catch(err => {
                })
            },
            immediate: false,
        },
        autoTranslate: {
            handler: function (val, oldVal) {
                if (!this.startWatchSave) return
                this.autoTranslateToEnglish = this.autoTranslate
                this.autoTranslateToLocal = this.autoTranslate
                console.log('onAutoTranslateChange', val)
                this.gradioAPI.setData('autoTranslate', val).then(data => {
                }).catch(err => {
                })
            },
            immediate: false,
        },
        autoRemoveSpace: {
            handler: function (val, oldVal) {
                if (!this.startWatchSave) return
                console.log('onAutoRemoveSpaceChange', val)
                this.gradioAPI.setData('autoRemoveSpace', val).then(data => {
                    this.prompts.forEach(item => {
                        this.$refs[item.id]?.[0]?.updatePrompt()
                    })
                }).catch(err => {
                })
            },
            immediate: false,
        },
        autoRemoveLastComma: {
            handler: function (val, oldVal) {
                if (!this.startWatchSave) return
                console.log('onAutoRemoveLastCommaChange', val)
                this.gradioAPI.setData('autoRemoveLastComma', val).then(data => {
                    this.prompts.forEach(item => {
                        this.$refs[item.id]?.[0]?.updatePrompt()
                    })
                }).catch(err => {
                })
            },
            immediate: false,
        },
        autoKeepWeightZero: {
            handler: function (val, oldVal) {
                if (!this.startWatchSave) return
                console.log('onAutoKeepWeightZeroChange', val)
                this.gradioAPI.setData('autoKeepWeightZero', val).then(data => {
                }).catch(err => {
                })
            },
            immediate: false,
        },
        autoKeepWeightOne: {
            handler: function (val, oldVal) {
                if (!this.startWatchSave) return
                console.log('onAutoKeepWeightOneChange', val)
                this.gradioAPI.setData('autoKeepWeightOne', val).then(data => {
                }).catch(err => {
                })
            },
            immediate: false,
        },
        autoBreakBeforeWrap: {
            handler: function (val, oldVal) {
                if (!this.startWatchSave) return
                console.log('onAutoBreakBeforeWrap', val)
                this.gradioAPI.setData('autoBreakBeforeWrap', val).then(data => {
                    this.prompts.forEach(item => {
                        this.$refs[item.id]?.[0]?.updatePrompt()
                    })
                }).catch(err => {
                })
            },
            immediate: false,
        },
        autoBreakAfterWrap: {
            handler: function (val, oldVal) {
                if (!this.startWatchSave) return
                console.log('onAutoBreakAfterWrap', val)
                this.gradioAPI.setData('autoBreakAfterWrap', val).then(data => {
                    this.prompts.forEach(item => {
                        this.$refs[item.id]?.[0]?.updatePrompt()
                    })
                }).catch(err => {
                })
            },
            immediate: false,
        },
        autoRemoveLoraBeforeComma: {
            handler: function (val, oldVal) {
                if (!this.startWatchSave) return
                console.log('onAutoRemoveLoraBeforeCommaChange', val)
                this.gradioAPI.setData('autoRemoveLoraBeforeComma', val).then(data => {
                }).catch(err => {
                })
            },
            immediate: false,
        },
        autoRemoveLoraAfterComma: {
            handler: function (val, oldVal) {
                if (!this.startWatchSave) return
                console.log('onAutoRemoveLoraAfterCommaChange', val)
                this.gradioAPI.setData('autoRemoveLoraAfterComma', val).then(data => {
                }).catch(err => {
                })
            },
            immediate: false,
        },
        useNovelAiWeightSymbol: {
            handler: function (val, oldVal) {
                if (!this.startWatchSave) return
                console.log('onUseNovelAiWeightSymbolChange', val)
                this.gradioAPI.setData('useNovelAiWeightSymbol', val).then(data => {
                }).catch(err => {
                })
            },
            immediate: false,
        },
        autoRemoveBeforeLineComma: {
            handler: function (val, oldVal) {
                if (!this.startWatchSave) return
                console.log('onAutoRemoveBeforeLineCommaChange', val)
                this.gradioAPI.setData('autoRemoveBeforeLineComma', val).then(data => {
                }).catch(err => {
                })
            }
        },
        /*hideDefaultInput: {
            handler: function (val, oldVal) {
                if (!this.startWatchSave) return
                console.log('onHideDefaultInputChange', val)
                this.prompts.forEach(item => {
                    item.$prompt.parentElement.parentElement.style.display = val ? 'none' : 'flex'
                })
                this.gradioAPI.setData('hideDefaultInput', val).then(data => {
                }).catch(err => {
                })
            },
            immediate: false,
        },*/
        enableTooltip: {
            handler: function (val, oldVal) {
                if (!this.startWatchSave) return
                this.host.tooltip = val
                this.updateTippyState()
                this.gradioAPI.setData('enableTooltip', val).then(data => {
                }).catch(err => {
                })
            },
            immediate: false,
        },
        translateApi: {
            handler: function (val, oldVal) {
                if (!this.startWatchSave) return
                console.log('onTranslateApiChange', val, oldVal)
                this.updateTranslateApiConfig()
                this.host.provider = val
                this.loadPackagesState()
                this.gradioAPI.setData('translateApi', val).then(data => {
                }).catch(err => {
                })
            },
            immediate: false,
        },
        tagCompleteFile: {
            handler: function (val, oldVal) {
                if (!this.startWatchSave) return
                console.log('onTagCompleteFileChange', val, oldVal)
                this.gradioAPI.setData('tagCompleteFile', val).then(data => {
                }).catch(err => {
                })
            },
            immediate: false,
        },
        groupTagsColor: {
            handler: function (val, oldVal) {
                if (!this.startWatchSave) return
                console.log('onGroupTagsColorChange', val, oldVal)
                this.gradioAPI.setData('groupTagsColor', val).then(data => {
                }).catch(err => {
                })
            },
            deep: true,
            immediate: false,
        },
        groupTagsTranslate: {
            handler: function (val, oldVal) {
                if (!this.startWatchSave) return
                console.log('onGroupTagsTranslateChange', val, oldVal)
                this.gradioAPI.setData('groupTagsTranslate', val).then(data => {
                }).catch(err => {
                })
            },
            immediate: false,
        },
        onlyCsvOnAuto() {
            if (!this.startWatchSave) return
            console.log('onOnlyCsvOnAutoChange', this.onlyCsvOnAuto)
            this.gradioAPI.setData('onlyCsvOnAuto', this.onlyCsvOnAuto).then(data => {
            }).catch(err => {
            })
        },
        extraNetworksWidth() {
            if (!this.startWatchSave) return
            if (this.extraNetworksWidthTimer) clearTimeout(this.extraNetworksWidthTimer)
            this.extraNetworksWidthTimer = this.$setTimeout(() => {
                console.log('onExtraNetworksWidthChange', this.extraNetworksWidth)
                this.gradioAPI.setData('extraNetworksWidth', this.extraNetworksWidth).then(data => {
                }).catch(err => {
                })
            }, 500)
        },
        extraNetworksHeight() {
            if (!this.startWatchSave) return
            if (this.extraNetworksHeightTimer) clearTimeout(this.extraNetworksHeightTimer)
            this.extraNetworksHeightTimer = this.$setTimeout(() => {
                console.log('onExtraNetworksHeightChange', this.extraNetworksHeight)
                this.gradioAPI.setData('extraNetworksHeight', this.extraNetworksHeight).then(data => {
                }).catch(err => {
                })
            }, 500)
        },
    },
    mounted() {
        this.$listen(window, 'fooocus-model-capabilities', () => this._loadExtraNetworks())
        this.$listen(window, 'fooocus-completion-catalog', () => this._loadExtraNetworks())
        this.theme = this.host.settings.theme === 'auto' ? (document.documentElement.classList.contains('dark') || document.body.classList.contains('dark') ? 'dark' : 'light') : this.host.settings.theme
        this.gradioAPI.getConfigWithRetry().then(res => {
            this.languageCode = res.i18n.default
            this.translateApi = res.translate_apis.default
            this.translateApis = res.translate_apis.apis
            this.python = res.python
            this.languages = Object.fromEntries(res.i18n.languages.map(lang => [lang.code, lang]))
            this.init()
            this.loadPackagesState()
        }).catch(err => {if (!this._fooocusDisposed) this.$toastr.error(err.message)})
    },
    methods: {
        loadPackagesState() {
            this.gradioAPI.getPackagesState().then(packagesState => {
                this.packagesState = packagesState
            }).catch(err => {
                console.warn('Failed to load package state.', err)
            })
        },
        getLang(key) {
            return common.getLang(key, this.languageCode, this.languages)
        },
        init() {
            this.loadExtraNetworks()
            let dataListsKeys = ['languageCode', 'autoTranslate', 'autoTranslateToEnglish', 'autoTranslateToLocal', 'autoRemoveSpace', 'autoRemoveLastComma', 'autoKeepWeightZero', 'autoKeepWeightOne', 'autoBreakBeforeWrap', 'autoBreakAfterWrap', 'autoRemoveLoraBeforeComma', 'autoRemoveLoraAfterComma', 'useNovelAiWeightSymbol', 'autoRemoveBeforeLineComma', /*'hideDefaultInput', */'translateApi', 'enableTooltip', 'tagCompleteFile', 'onlyCsvOnAuto', 'extensionSelect.minimalist', 'groupTagsColor', 'groupTagsTranslate', 'blacklist', 'cancelBlacklistConfirm', 'hotkey', 'extraNetworksWidth', 'extraNetworksHeight']
            this.prompts.forEach(item => {
                dataListsKeys.push(item.hideDefaultInputKey)
                dataListsKeys.push(item.autoLoadWebuiPromptKey)
                dataListsKeys.push(item.hidePanelKey)
                dataListsKeys.push(item.hideGroupTagsKey)
            })

            this.gradioAPI.getDatas(dataListsKeys).then(data => {
                if (data.languageCode !== null) {
                    let findLang = false
                    for (let key in this.languages) {
                        if (this.languages[key].code === data.languageCode) {
                            findLang = true
                            break
                        }
                    }
                    if (findLang) {
                        this.languageCode = data.languageCode
                        this.$forceUpdate()
                        this.gradioAPI.setData('languageCode', this.languageCode)
                    }
                } else {
                    let browserLang = navigator.language || navigator.userLanguage || ''
                    if (browserLang) {
                        for (let key in this.languages) {
                            if (common.isSameLang(this.languages[key].code, browserLang)) {
                                this.languageCode = this.languages[key].code
                                this.$forceUpdate()
                                this.gradioAPI.setData('languageCode', this.languageCode)
                                break
                            }
                        }
                    }
                }
                this.canOneTranslate = common.canOneTranslate(this.languageCode)
                if (data.autoTranslateToEnglish !== null) {
                    this.autoTranslateToEnglish = data.autoTranslateToEnglish
                }
                if (data.autoTranslateToLocal !== null) {
                    this.autoTranslateToLocal = data.autoTranslateToLocal
                }
                if (data.autoTranslate !== null) {
                    if (this.canOneTranslate) {
                        this.autoTranslate = data.autoTranslate
                        this.autoTranslateToEnglish = this.autoTranslate
                        this.autoTranslateToLocal = this.autoTranslate
                    } else {
                        this.autoTranslate = false
                    }
                } else {
                    if (this.canOneTranslate) {
                        this.autoTranslate = this.autoTranslateToEnglish || this.autoTranslateToLocal
                        this.autoTranslateToEnglish = this.autoTranslate
                        this.autoTranslateToLocal = this.autoTranslate
                    } else {
                        this.autoTranslate = false
                    }
                }
                if (data.autoRemoveSpace !== null) {
                    this.autoRemoveSpace = data.autoRemoveSpace
                }
                if (data.autoRemoveLastComma !== null) {
                    this.autoRemoveLastComma = data.autoRemoveLastComma
                }
                if (data.autoKeepWeightZero !== null) {
                    this.autoKeepWeightZero = data.autoKeepWeightZero
                }
                if (data.autoKeepWeightOne !== null) {
                    this.autoKeepWeightOne = data.autoKeepWeightOne
                }
                if (data.autoBreakBeforeWrap !== null) {
                    this.autoBreakBeforeWrap = data.autoBreakBeforeWrap
                }
                if (data.autoBreakAfterWrap !== null) {
                    this.autoBreakAfterWrap = data.autoBreakAfterWrap
                }
                if (data.autoRemoveLoraBeforeComma !== null) {
                    this.autoRemoveLoraBeforeComma = data.autoRemoveLoraBeforeComma
                }
                if (data.autoRemoveLoraAfterComma !== null) {
                    this.autoRemoveLoraAfterComma = data.autoRemoveLoraAfterComma
                }
                if (data.useNovelAiWeightSymbol !== null) {
                    this.useNovelAiWeightSymbol = data.useNovelAiWeightSymbol
                }
                if (data.autoRemoveBeforeLineComma !== null) {
                    this.autoRemoveBeforeLineComma = data.autoRemoveBeforeLineComma
                }
                /*if (data.hideDefaultInput !== null) {
                    this.hideDefaultInput = data.hideDefaultInput
                }*/
                if (data.enableTooltip !== null) {
                    this.enableTooltip = data.enableTooltip
                }
                this.host.tooltip = this.enableTooltip
                this.updateTippyState()
                if (data.translateApi !== null) {
                    this.translateApi = data.translateApi
                    /*if (data.translateApi === 'alibaba_free') {
                        this.gradioAPI.setData('translateApi', this.translateApi)
                    } else {
                        this.translateApi = data.translateApi
                    }*/
                }
                if (data.tagCompleteFile !== null) {
                    this.tagCompleteFile = data.tagCompleteFile
                    waitTick.addWaitTick(() => {
                        this.$refs.translateSetting.getCSV(this.tagCompleteFile)
                    })
                } else {
                    /*if (typeof TAC_CFG === 'object' && typeof QUEUE_FILE_LOAD === 'object') {
                        QUEUE_FILE_LOAD.push(() => {
                            if (typeof TAC_CFG.translation !== 'object' || typeof TAC_CFG.translation.translationFile !== 'string') return
                            if (!TAC_CFG.translation.translationFile) return
                            this.tagCompleteFile = '\\extensions\\a1111-sd-webui-tagcomplete\\tags\\' + TAC_CFG.translation.translationFile
                            this.$refs.translateSetting.getCSV(this.tagCompleteFile)
                        })
                    }*/
                }
                if (data.onlyCsvOnAuto !== null) {
                    this.onlyCsvOnAuto = data.onlyCsvOnAuto
                }

                if (data['extensionSelect.minimalist'] === null) {
                    this.gradioAPI.setData('extensionSelect.minimalist', true)
                }

                if (data.groupTagsColor !== null) {
                    if (typeof data.groupTagsColor === 'object') {
                        this.groupTagsColor = {}
                        for (let key in data.groupTagsColor) {
                            let color = data.groupTagsColor[key]
                            this.groupTagsColor[key] = ref(common.fitterInputColor(color))
                        }
                    }
                }

                if (data.groupTagsTranslate !== null) {
                    this.groupTagsTranslate = data.groupTagsTranslate
                }

                if (data.blacklist !== null) {
                    this.blacklist = this._handleBlacklist(data.blacklist)
                }

                if (data.cancelBlacklistConfirm !== null) {
                    this.cancelBlacklistConfirm = data.cancelBlacklistConfirm
                }

                if (data.hotkey !== null) {
                    this.hotkey = data.hotkey
                }

                if (data.extraNetworksWidth !== null) {
                    this.extraNetworksWidth = data.extraNetworksWidth
                }

                if (data.extraNetworksHeight !== null) {
                    this.extraNetworksHeight = data.extraNetworksHeight
                }

                this.updateTranslateApiConfig()
                this.$refs.extensionCss.init()

                this.prompts.forEach(item => {
                    if (data[item.hideDefaultInputKey] !== null) {
                        item.hideDefaultInput = data[item.hideDefaultInputKey]
                    }
                    if (data[item.autoLoadWebuiPromptKey] !== null) {
                        item.autoLoadWebuiPrompt = data[item.autoLoadWebuiPromptKey]
                    }
                    if (data[item.hidePanelKey] !== null) {
                        item.hidePanel = data[item.hidePanelKey]
                    }
                    if (data[item.hideGroupTagsKey] !== null) {
                        item.hideGroupTags = data[item.hideGroupTagsKey]
                    }
                })
                this.prompts.forEach(item => {item.mount.classList.toggle("dark", this.theme === "dark"); item.mount.classList.toggle("light", this.theme === "light")})
                this.prompts.forEach(item => item.$prompt.classList.toggle('pai-native-hidden', item.hideDefaultInput))
                this.initialized = true
                this.$nextTick(() => { this.startWatchSave = true })

                // Fooocus native JSON/preset import stays on its original Gradio controls.

                waitTick.addWaitTick(() => this.loadGroupTags())

                /*this.gradioAPI.getVersion().then(res => {
                    this.version = res.version
                    this.latestVersion = res.latest_version
                    this.isLatestVersion = res.version === res.latest_version
                })*/

                // todo: test
                // this.$refs.about.open()
                // this.$refs.chatgptPrompt.open()
                // this.$refs.promptFormat.open()
                // this.$refs.blacklist.open()
                // this.$refs.hotkey.open()
                // this.$refs.translateSetting.open(this.translateApi)
                /*this.$refs.extraNetworksPopup.show({
                    getBoundingClientRect: () => {
                        return {
                            top: 0,
                            left: 0,
                            width: 0,
                            height: 0,
                            bottom: 0,
                            right: 0,
                        }
                    },
                    offsetHeight: 26,
                }, 'FGOTiamatv2')*/
                /*this.onShowFavorite('phystonPrompt_txt2img_prompt', {
                    clientY: 150,
                    clientX: 283,
                })*/
            })
        },
        loadGroupTags() {
            return this.gradioAPI.getGroupTags(this.languageCode).then(data => {
                if (!data || data === '') {
                    this.groupTags = []
                } else {
                    try {
                        this.groupTags = jsYaml.load(data)
                        if (!Array.isArray(this.groupTags)) {
                            this.groupTags = []
                        }
                    } catch (e) {
                        console.log(e)
                        this.groupTags = []
                    }
                }
                this._handleGroupTags()
            })
        },
        _handleGroupTags() {
            let data = {toEn: new Map(), toLocal: new Map()}
            let setData = (en, local) => {
                const texts = [
                    en,
                    en.replace(/\_/g, ' '),
                    en.replace(/\-/g, ' '),
                ]
                texts.forEach((t) => {
                    if (data.toLocal.has(t)) {
                        let oldLocal = data.toLocal.get(t)
                        if (!oldLocal.includes(local)) {
                            oldLocal.push(local)
                        }
                    } else {
                        data.toLocal.set(t, [local])
                    }
                })
                data.toEn.set(local, en)
                // console.log('setData:groupTags', local, key, en)
            }
            this.groupTags.forEach((item, index) => {
                item.type = item.type || ''
                item.tabKey = 'groupTags-' + index
                item.groups.forEach((group, subIndex) => {
                    group.type = group.type || ''
                    group.tabKey = 'subGroupTags-' + index + '-' + subIndex
                    if (group.type == 'wrap') return
                    let key = common.getTagsColorKey(item.name, group.name)
                    if (!this.groupTagsColor[key]) {
                        this.groupTagsColor[key] = ref(common.fitterInputColor(group.color))
                    }
                    for (let en in group.tags) {
                        if (!en) continue
                        this.groupTagsColorKeyCache[en] = key

                        let local = group.tags[en]
                        if (!local || en == local) continue
                        setData(en, local)
                    }
                })
            })

            this.groupTagsTranslateCache = data
        },
        updateTippyState() {
            for (const $tippy of this.$tippyList) {
                if (this.enableTooltip) {
                    $tippy.enable()
                } else {
                    $tippy.disable()
                }
            }
        },
        updateTranslateApiConfig() {
            this.gradioAPI.getData('translate_api.' + this.translateApi).then(res => {
                let config = {}
                const apiItem = common.getTranslateApiItem(this.translateApis, this.translateApi)
                if (apiItem.config) {
                    for (const item of apiItem.config) {
                        if (apiItem.type === 'translators' && item.key === 'region' && !res['region']) {
                            config[item.key] = this.languageCode === 'zh_CN' || this.languageCode === 'zh_HK' || this.languageCode === 'zh_TW' ? 'China' : 'EN'
                        } else {
                            if (res) {
                                config[item.key] = res[item.key]
                            } else {
                                config[item.key] = item.default || ''
                            }
                        }
                    }
                    config['concurrent'] = apiItem.concurrent || 0
                }
                this.translateApiConfig = config
            })
        },
        onPromptFormatClick(e) {
            this.$refs.promptFormat.open(e)
        },
        onBlacklistClick(e) {
            this.$refs.blacklist.open(e)
        },
        onHotkeyClick(e) {
            this.$refs.hotkey.open(e)
        },
        onSelectLanguageClick(e) {
            this.$refs.selectLanguage.open(e)
        },
        onTranslateApiClick() {
            this.$refs.translateSetting.open(this.translateApi)
        },
        onSelectThemeClick() {
            this.$refs.extensionCss.open()
        },
        handlePaste() {},
        openPastePopup() {
            this.pasteContent = ''
            this.pasteLoading = false
            this.showPastePopup = true
        },
        closePastePopup() {
            this.showPastePopup = false
        },
        onClickPasteSubmit() {
            const item = this.prompts.find(item => !item.neg && item.historyKey === 'positive')
            if (!item) return
            item.$textarea.value = this.pasteContent
            item.$textarea.dispatchEvent(new Event('input', {bubbles: true}))
            this.closePastePopup()
        },
        onUpdateHideDefaultInput(id, value) {
            const item = this.prompts.find(item => item.id == id)
            if (!item) return
            item.hideDefaultInput = value
            this.gradioAPI.setData(item.hideDefaultInputKey, item.hideDefaultInput)
            item.$prompt.classList.toggle('pai-native-hidden', item.hideDefaultInput)
        },
        onUpdateAutoLoadWebuiPrompt(id, value) {
            const item = this.prompts.find(item => item.id == id)
            if (!item) return
            item.autoLoadWebuiPrompt = value
            this.gradioAPI.setData(item.autoLoadWebuiPromptKey, item.autoLoadWebuiPrompt)
        },
        onUpdateHidePanel(id, value) {
            const item = this.prompts.find(item => item.id == id)
            if (!item) return
            item.hidePanel = value
            this.gradioAPI.setData(item.hidePanelKey, item.hidePanel)
        },
        onUpdateHideGroupTags(id, value) {
            const item = this.prompts.find(item => item.id == id)
            if (!item) return
            item.hideGroupTags = value
            this.gradioAPI.setData(item.hideGroupTagsKey, item.hideGroupTags)
        },
        onShowHistory(id, e) {
            this.$refs.favorite.hide()
            this.historyCurrentPrompt = id
            const item = this.prompts.find(item => item.id == id)
            if (!item) return
            this.$refs.history.show(item.historyKey, e)
        },
        onUseHistory(history) {
            if (!this.historyCurrentPrompt) return
            const item = this.prompts.find(item => item.id == this.historyCurrentPrompt)
            if (!item) return
            this.$refs[item.id]?.[0]?.useFavorite(history)
        },
        onShowFavorite(id, e) {
            this.$refs.history.hide()
            this.favoriteCurrentPrompt = id
            const item = this.prompts.find(item => item.id == id)
            if (!item) return
            this.$refs.favorite.show(item.favoriteKey, e)
        },
        onUseFavorite(favorite) {
            if (!this.favoriteCurrentPrompt) return
            const item = this.prompts.find(item => item.id == this.favoriteCurrentPrompt)
            if (!item) return
            this.$refs[item.id]?.[0]?.useFavorite(favorite)
        },
        onRefreshFavorites(key) {
            this.$refs.favorite.getFavorites(key)
        },
        onShowChatgpt(id, e) {
            this.chatgptCurrentPrompt = id
            const item = this.prompts.find(item => item.id == id)
            if (!item) return
            this.$refs.chatgptPrompt.open()
        },
        onUseChatgpt(prompt) {
            if (!this.chatgptCurrentPrompt) return
            const item = this.prompts.find(item => item.id == this.chatgptCurrentPrompt)
            if (!item) return
            this.$refs[item.id]?.[0]?.useChatgpt(prompt)
        },
        onShowAbout() {
            this.$refs.about.open()
        },
        onSwitchTheme() {
            this.theme = this.theme === 'dark' ? 'light' : 'dark'
            this.host.setSetting('theme', this.theme)
        },
        _handleBlacklist(blacklist) {
            blacklist = {...blacklist}
            blacklist.prompt = blacklist.prompt?.slice().map(item => item.toLowerCase())
            blacklist.negative_prompt = blacklist.negative_prompt?.slice().map(item => item.toLowerCase())
            blacklist.lora = blacklist.lora?.slice().map(item => item.toLowerCase())
            blacklist.lycoris = blacklist.lycoris?.slice().map(item => item.toLowerCase())
            blacklist.embedding = blacklist.embedding?.slice().map(item => item.toLowerCase())
            blacklist.translate = blacklist.translate?.slice().map(item => item.toLowerCase())
            return blacklist
        },
        onUpdateBlacklist(data, cancelBlacklistConfirm) {
            this.blacklist = this._handleBlacklist(data)
            if (typeof cancelBlacklistConfirm === "boolean") this.cancelBlacklistConfirm = cancelBlacklistConfirm
        },
        onUpdateHotkey(data) {
            this.hotkey = data
        },
        onShowExtraNetworks(e, name, useCallback, showCheckpoints, from) {
            this.$refs.extraNetworksPopup.show(e, name, useCallback, showCheckpoints, from)
        },
        onHideExtraNetworks() {
            this.$refs.extraNetworksPopup.hide()
        },
        onRefreshExtraNetworks() {
            this._loadExtraNetworks()
        },
    },
}
</script>
