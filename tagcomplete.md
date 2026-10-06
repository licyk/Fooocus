# 提示词辅助

本分支内置了提示词补全，无需安装扩展。实现参考 [a1111-sd-webui-tagcomplete](https://github.com/DominikDoom/a1111-sd-webui-tagcomplete)，并适配 Fooocus 的提示词语法和 Gradio 界面。

打开 **Advanced → Prompt Assistance** 调整设置。正向、负向、Inpaint additional prompt 和每个 Enhance 输入框均支持补全，支持分别关闭。标签查询在浏览器的 Web Worker 中运行。

## 使用方法

| 输入 | 行为 |
| --- | --- |
| `blue_h` | 搜索标签及别名；默认插入 `blue hair, ` |
| `@artist` | 仅搜索画师标签；默认保留 `@`，可用于 Anima |
| `<l:名称` 或 `<lora:名称` | 插入 `<lora:相对路径:权重>` |
| `<e:名称` 或 `embedding:名称` | 插入 Fooocus 的 `embedding:名称` |
| `__名称` | 补全通配符文件名，包括子目录 |
| `__hair/color__` | 查看文件内容，选择一行后替换整个占位符 |
| `<c:名称` 或 `<chant:名称` | 插入 Chants 提示词片段 |
| `$名称` | 在主正向提示词中搜索并勾选原生 Fooocus Styles，移除输入的标记 |
| `*片段` | 执行包含匹配 |

默认按 **↑ / ↓** 选择候选，**Enter** 确认已选候选，**Tab** 确认第一项，**Esc** 关闭。PageUp / PageDown 每次移动五项，Home / End 移到首尾。通配符 Tab 支持逐级目录、公共前缀或完整路径三种模式；直接点击候选会插入完整路径。在已补全的通配符后输入空格可以保留占位符。

中文输入法组词期间不会打开补全。确认后才搜索。替换只影响光标所在的提示词片段，保留其权重和后续提示词。Fooocus 的 Ctrl+Enter 生成快捷键和 Ctrl+↑ / ↓ 权重快捷键保持优先。

Anima 主模型或 Refiner 使用 Qwen/T5，不支持 textual inversion，因此选择任一 Anima 模型时会关闭 embedding 候选。LoRA 候选来自本地文件列表，是否适用于当前模型仍取决于 LoRA 自身。

## 设置与保存

- 设置修改立即作用于当前浏览器，保存至该站点的 `localStorage`。使用另一浏览器或另一站点地址时有独立的偏好。
- **Save as default** 显式保存服务器默认值，文件是主配置所在目录下的 `tagcomplete_settings.json`。已有浏览器保留各自的偏好。
- **Load server defaults** 将服务器默认值应用到当前浏览器；**Restore built-in defaults** 恢复内置值。二者均不会自动修改服务器文件。
- **Refresh completion datasets** 重新扫描标签、片段、模型和通配符，更新数据集下拉选项。Models 页的 **Refresh All Files** 也会刷新补全目录。
- **Clear my completion usage** 删除当前登录用户的补全使用记录。

设置包括数据集与翻译、别名搜索、分类过滤、匹配方式、模型名称黑白名单、候选数量与批量加载、延迟、预览图、Wiki 链接、实时翻译、下划线和括号处理、分隔符、画师前缀、LoRA 权重与触发词、通配符排序与目录排除、使用频率排序、键盘映射和分类颜色。

频率记录默认关闭。打开后仅记录实际确认的候选，正向与负向提示词分开统计，并按 Fooocus 登录用户隔离；未启用登录时使用共享的 `local` 身份。支持次数门槛、过期天数、提升数量上限和别名匹配策略。此缓存不保存完整提示词。

高级设置的键盘映射是 JSON，对应 `KeyboardEvent.key`，修饰键按 `Ctrl+Alt+Shift+Meta+键名` 顺序书写。所有动作键必须保留；空字符串可禁用动作，非空组合不能重复。分类颜色是 `{ "类别": ["暗色主题颜色", "亮色主题颜色"] }`，支持 `default`。

## 本地数据

默认数据目录为 `tags/`，缓存目录为 `cache/tagcomplete/`。可在 Fooocus `config.txt` 中配置 `path_tagcomplete`、`path_tagcomplete_cache`。标签和片段文件放在数据目录顶层，然后刷新并选择。

标签 CSV 格式（无表头，UTF-8，兼容带 BOM、引号和 CRLF）：

```csv
blue_hair,0,855605,"blue_tresses,blue-haired",蓝发
```

列依次为 **标签、分类、使用次数、逗号分隔的别名、可选翻译**。附加标签的第三列可填写显示文本；`Extra tag priority` 决定附加标签排在常规标签之前还是之后。常见分类为 `0` 普通、`1` 画师、`3` 作品、`4` 角色、`5` 元数据。

翻译 CSV 格式：

```csv
blue_hair,蓝发
```

原扩展的旧格式为 `tag,unused,translation`，使用时打开 **Legacy three-column translations**。中文可参与搜索，也可显示在候选及提示词下方。支持为别名配置翻译。

Chants JSON 格式：

```json
[
  {"name": "Quality", "terms": "quality,best,质量", "content": "masterpiece, best quality", "color": 5}
]
```

LoRA、embedding 和通配符自动使用 Fooocus 已配置的目录，保留相对路径以区分同名文件。通配符支持子目录、空格、点和括号，忽略空行及以 `#` 开头的注释。只有符合 Fooocus 提示词解析规则的文件名才会出现为候选，例如 embedding 名称不能包含空格。

模型预览依次寻找同名 `.preview.png`、`.png`、`.jpg`、`.jpeg` 或 `.webp`。LoRA 同名 JSON 支持本地元数据：

```json
{"preferred weight": 0.7, "activation text": "cat_style, detailed fur"}
```

也支持 `trigger_words` 字符串或数组。需打开 **Insert local LoRA trigger words** 才会插入触发词；可选择提示词开头、末尾或 LoRA 前。已有 LoRA 提示词中的权重优先于本地元数据和默认权重。

本实现不提供 Fooocus 没有的 Hypernetwork、UMI 和外部动态提示词语法。支持用户放入其他兼容标签 CSV；内置只带 Danbooru、质量附加标签和示例 Chants。

## 来源与许可

移植参考本地扩展版本 `4170882f90b47be130a0ff9314f663c230b9153d`。数据、示例及光标定位代码来源见 [tags/README.md](tags/README.md)，原扩展 MIT 许可保留在 [tags/TAGCOMPLETE_LICENSE](tags/TAGCOMPLETE_LICENSE)。
