# Prompt All-in-One

在 **Advanced → Prompt Assistance → Prompt All-in-One** 勾选 **Enable Prompt All-in-One**。默认关闭，可单独开启主正向、主反向、Inpaint 和 Enhance 编辑器。开关与 Tag Autocomplete 独立；启用两者时共用已有的补全 Worker、候选窗口、词库、模型预览和使用频率。

## 使用

- **Text / Tags**：自由文本与标签视图。切换视图不会转换标点、空白、换行、转义或嵌套权重。自然语言可直接输入；粘贴的 JSON 参数仍由 Fooocus 原生导入功能处理。
- 标签可修改、拖动排序、调整权重、删除、禁用和批量操作。禁用标签保留在编辑状态与历史/收藏中，不进入生成提示词。
- 历史按提示词字段记录，默认保留最近 100 条，可设为 10–1000 条；收藏独立保留，支持命名、排序、搜索和选中标签收藏。
- 分类词库复用上游 12 种语言的数据。可在设置中的 **Custom tag groups** 添加自己的词组，例如 `[{"name":"My tags","tags":{"cat":"猫","dog":"狗"}}]`。黑名单每行一个词，影响词库、编辑器补全和添加操作，不会删除原始提示词中已有的文字。
- 模型栏插入原生 `<lora:relative/name:weight>`、`embedding:name` 和 `__folder/wildcard__`。LoRA 使用本地元数据中的推荐权重和注册预览。Anima 基础模型或 Refiner 会隐藏 embedding 候选。
- 可调整主题、模型标签颜色、新增词分隔符、权重步长、双语显示、翻译和历史开关。开启快捷键后，`Ctrl/Cmd+Shift+F` 收藏，`Alt+↑/↓` 调整选中词权重；Fooocus 的 `Ctrl+Enter` 生成与原生权重快捷键继续有效。
- **Hide original prompt** 只隐藏对应的原始输入，不隐藏 Generate 或其父级布局。关闭功能时编辑器卸载并取消前端请求、计时器和补全监听；当前提示词仍保留在原始输入中。

所有生成参数继续从原始 Gradio 提示词组件读取。编辑器与原始输入双向同步，也接收预设、图片参数和历史回填。浏览器设置按登录用户名分别保存；**Save as default** 明确保存服务端默认值，**Load server defaults / Restore built-in defaults** 调整当前浏览器。

## 翻译与智能提示词

默认 **dictionary** 仅使用本地分类词库与现有 CSV 翻译，不访问外部服务。输入词在本地词库缺失时保留原文。所选浏览器的 Tag Autocomplete 翻译数据优先用于双语显示；服务端字典翻译使用其保存的默认 CSV。

在编辑器的 **翻译接口** 中选择服务并保存配置。支持 OpenAI 兼容接口、DeepL、LibreTranslate、Microsoft Translator，以及从原扩展移植的 Google Cloud、百度、Amazon、阿里云、Yandex、有道、腾讯、小牛、彩云、火山引擎、讯飞 V1/V2 和 MyMemory。原扩展的网页爬取式免费聚合翻译器不移植；这些接口不作为默认服务。

API 密钥只存储在服务端，不返回浏览器。空密钥保留已保存值，勾选 **Clear saved API key** 清除当前服务凭证。上游签名适配器和 MBart50 需选择明确的源语言，支持范围由各服务决定。翻译修改词时保留权重及模型语法；请求期间编辑提示词会丢弃过期返回，失败保留原文。

外部翻译和智能提示词会将所选内容发送到用户配置的服务。智能提示词使用保存的 OpenAI 兼容 endpoint、key 和 model，生成后先展示结果，点击 Add 才写入提示词。外部服务须有可用网络与账号；未配置时显示错误，不会自动切换服务。

**MBart50** 可指定已有本地模型目录，第一次主动调用才在 CPU 上加载，使用本地文件且不自动下载。Amazon 需要可选 `boto3`；阿里云需要可选 `aliyun-python-sdk-core` 与 `aliyun-python-sdk-alimt`；MBart50 需要与现有 Transformers 兼容的模型/tokenizer文件及 `sentencepiece`。应用不会安装这些可选 SDK，也不会改变核心依赖版本。

Token 数是输入文本的预览，尚未解析通配符/样式，也未展开 embedding 向量。SDXL 显示 CLIP 数与 75-token 分块容量；Anima 分别显示 Qwen 与 T5 数量，混合 Refiner 时分别显示基础/Refiner 计数。计算仅加载项目自带 tokenizer，不加载图像或文本编码器权重。

## 数据与开发

默认设置：配置文件旁的 `prompt_all_in_one_settings.json`。历史、收藏、服务凭证：同目录 `userdata/prompt_all_in_one/editor.sqlite3`。这些文件不提交 Git。启用 Gradio 登录时，各用户的数据通过同一登录机制隔离；未启用登录时使用共享本地用户 `local`。

静态资源已随项目提供，运行 Fooocus 无需 Node.js。修改前端后执行：

```sh
cd frontend/prompt_all_in_one
npm ci
npm test
npm run build
```

源码基于 [Physton/sd-webui-prompt-all-in-one](https://github.com/Physton/sd-webui-prompt-all-in-one) 的 MIT 版本，移植参考 [licyk 分支提交 5179da83](https://github.com/licyk/sd-webui-prompt-all-in-one/tree/5179da83c7a6ba688f924f0c81cb775d5695a414)。Fooocus 的 Vue 编辑器重写了宿主绑定、无损提示词解析、生命周期和后端存储；沿用上游分类词库、多语言文案、签名翻译适配器和权重输入组件（原作者 fengyuanchen）。MIT 许可保留在 [LICENSE.upstream](frontend/prompt_all_in_one/LICENSE.upstream)。
