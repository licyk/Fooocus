# Prompt All-in-One

Prompt All-in-One 默认启用，可在 **Advanced → Prompt Assistance → Prompt All-in-One** 调整 **Enable Prompt All-in-One**。主正向、主反向、Inpaint、Enhance 可独立启停。Tag Autocomplete 保留自己的开关。已保存的服务器默认值和浏览器偏好仍按原有优先级生效。

此版本直接复制 `~/code_workspace/sd-webui-prompt-all-in-one` 的源码，在源码基础上修改 Fooocus 宿主接入。原版 `App.vue`、全部 16 个 Vue 组件、标签/拖动/分类词库 mixin、分词与权重工具、历史收藏操作、108 种界面语言、图标和主题扩展均保留；上一版手写的简化组件与解析器已移除。[源码记录](frontend/prompt_all_in_one/UPSTREAM.json) 保存参考提交、来源路径及原始 SHA-256，便于以后与上游对比。

## 界面与操作

编辑器采用原扩展的布局：提示词标题、token 计数、原版工具栏、输入新关键词、标签列表、分类标签页和模型卡片。添加关键词按 Enter；单击标签编辑、双击禁用、拖动调整顺序，鼠标行为可在原版快捷键弹窗修改。悬停标签后使用数字权重、括号层数、换行、翻译和删除按钮。原版框选、多标签拖动与批量操作也保留。

工具栏中的设置图标展开原版菜单：翻译接口、提示词格式、黑名单、鼠标快捷键、主题扩展、明暗主题、关于、自动翻译、隐藏原始输入、自动同步、提示说明等。历史、收藏、语言选择和智能提示词使用各自的原版弹窗。格式设置仍提供空格/末尾逗号、BREAK 换行、LoRA 两侧逗号、权重等选项。分类词库保留上游颜色与布局节点，可以在分类栏调整颜色；词库文件位于 `modules/prompt_all_in_one/group_tags/`。

原版减权方括号及可选 NovelAI 花括号在编辑器内保留，写入生成输入时转换成 Fooocus 支持的数字圆括号权重。未执行标签操作时保持原始文本；A1111 专属的提示词调度等生成语法仍由 Fooocus 本身决定是否支持。

启用功能及接收原始文本、预设、JSON 图片参数回填时不主动改写原始提示词。执行原版标签编辑操作后，按照原版格式设置生成文本。Fooocus 最终从原始 Gradio 提示词组件读取生成参数；Generate 按钮不会随“隐藏原始输入”隐藏。关闭功能会提交尚在编辑的标签、卸载 Vue 界面、取消请求与计时器、清理补全监听和主题扩展样式，当前提示词保留在原始输入中。

模型卡片复用 Fooocus 的实际模型目录和现有标签补全目录数据。基础模型卡片通过 Gradio 切换模型，LoRA/embedding/通配符插入 Fooocus 原生语法。Anima 基础模型或 Refiner 会隐藏 embedding；token 计数按实际基础模型/Refiner 分别显示 CLIP 或 Qwen/T5，仅加载 tokenizer。

## 翻译和智能提示词

默认 **Local dictionary / 本地词库** 使用本地分类词库；TagComplete CSV 翻译增强仍通过原版设置弹窗选择。原版免费聚合服务、签名 API 服务、MBart50 均保留在原接口列表，另保留 LibreTranslate。免费服务会访问对应网站，可能需要可选依赖且受服务端可用性限制。

TagComplete 本地翻译读取与提示词补全相同的数据目录：默认 `tags/`，可通过 `config.txt` 中的 `path_tagcomplete` 修改。弹窗显示当前目录和 CSV 文件名，点击刷新会重新扫描新增或删除的文件。请选择 UTF-8 的两列翻译文件（`tag,translation`）；JSON 词库不会出现在此列表中。选择保存后会保留当前文件，不启用选项可关闭 CSV 翻译增强。

在原版翻译设置弹窗配置接口并测试，API 密钥使用密码输入；凭证按登录用户保存在服务器，不返回浏览器，空凭证保留已保存值。智能提示词保留原版 API 配置、可编辑预设、图片描述、生成结果与“使用”流程，查看生成结果后才能写入提示词。

MBart50 使用指定的已有本地模型目录，主动初始化或翻译时才在 CPU 加载，不自动下载权重。可选翻译 SDK 只在选中并调用相关服务时使用；缺失依赖显示原版依赖弹窗和安装命令，“安装”按钮仅安装列表中允许的可选包，不在启动时自动安装。源码里的免费聚合服务后端来自 UlionTse/translators，保留 GPL-3.0 声明及 [许可](modules/prompt_all_in_one/providers/LICENSE.translators)。

外部翻译和智能提示词会向所选服务发送对应内容；需要用户配置服务和网络。界面语言支持不代表每个翻译服务都支持对应语言，以原版接口的语言支持列表为准。

## 数据与开发

全局开关、字段范围、历史开关/上限的设置仍可在 Advanced 中调整；“Save as default / Load server defaults / Restore built-in defaults”操作针对这些宿主设置。编辑器内部的原版设置按登录用户保存在 SQLite。默认路径：配置文件旁 `prompt_all_in_one_settings.json` 和 `userdata/prompt_all_in_one/editor.sqlite3`。开启 Gradio 登录后使用同一登录机制隔离数据；本地无登录模式使用共享用户 `local`。上一版历史/收藏首次访问自动迁移，禁用标签状态保留，旧数据不删除。

前端源代码位于 `frontend/prompt_all_in_one/`，已编译的 JS/CSS 随项目提供，运行 Fooocus 不需要 Node.js。修改源码后执行：

```sh
cd frontend/prompt_all_in_one
npm ci
npm test
npm run build
```

移植参考 [licyk 源码提交 5179da83](https://github.com/licyk/sd-webui-prompt-all-in-one/tree/5179da83c7a6ba688f924f0c81cb775d5695a414)，原项目为 [Physton/sd-webui-prompt-all-in-one](https://github.com/Physton/sd-webui-prompt-all-in-one)。宿主差异集中在挂载与生命周期、API 路由、认证存储、实际模型目录、Gradio 回填和 token 计数；原版 MIT 声明保留在 [LICENSE.upstream](frontend/prompt_all_in_one/LICENSE.upstream)。
