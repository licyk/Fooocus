# 图片反推模型与参数

在“图生图 → 图像提示词反推”中选择“画作 / 动漫图片”，即可看到“反推模型”下拉框。
照片模式继续使用原有 BLIP；同时选择两个内容类型时，两种结果会合并到提示词。
“反推参数”中的设置也用于已启用的自动图片反推。

## 模型

移植 `sd-webui-wd14-tagger` 的全部 17 个注册项，保留原仓库、文件名和版本分支：

| 模型选项 | 仓库 | 分支 |
| --- | --- | --- |
| cl_tagger_1_01 | cella110n/cl_tagger | main，cl_tagger_1_01 子目录 |
| wd-eva02-large-tagger-v3 | SmilingWolf/wd-eva02-large-tagger-v3 | main |
| wd-vit-large-tagger-v3 | SmilingWolf/wd-vit-large-tagger-v3 | main |
| wd-vit-v3 | SmilingWolf/wd-vit-tagger-v3 | main |
| wd-swinv2-v3 | SmilingWolf/wd-swinv2-tagger-v3 | main |
| wd-convnext-v3 | SmilingWolf/wd-convnext-tagger-v3 | main |
| wd14-convnextv2-v2 | SmilingWolf/wd-v1-4-convnextv2-tagger-v2 | v2.0 |
| wd14-moat-v2 | SmilingWolf/wd-v1-4-moat-tagger-v2 | v2.0 |
| wd14-vit-v2 | SmilingWolf/wd-v1-4-vit-tagger-v2 | v2.0 |
| wd14-convnext-v2 | SmilingWolf/wd-v1-4-convnext-tagger-v2 | v2.0 |
| wd14-swinv2-v2 | SmilingWolf/wd-v1-4-swinv2-tagger-v2 | v2.0 |
| wd14-convnextv2-v2-git | SmilingWolf/wd-v1-4-convnextv2-tagger-v2 | main |
| wd14-vit-v2-git | SmilingWolf/wd-v1-4-vit-tagger-v2 | main |
| wd14-convnext-v2-git | SmilingWolf/wd-v1-4-convnext-tagger-v2 | main |
| wd14-swinv2-v2-git | SmilingWolf/wd-v1-4-swinv2-tagger-v2 | main |
| wd14-vit | SmilingWolf/wd-v1-4-vit-tagger | main |
| wd14-convnext | SmilingWolf/wd-v1-4-convnext-tagger | main |

默认保持 MOAT v2。可通过 `config.txt` 中的 `default_describe_tagger_model` 修改默认模型。
首次使用才下载选中的 ONNX 与标签文件，支持项目已有的 `HF_MIRROR`。
模型缓存位于 `path_clip_vision/taggers/<模型选项>/`，不同版本的缓存相互独立。
原有 `wd-v1-4-moat-tagger-v2.onnx` 与 `.csv` 完整存在时继续复用。

## 参数

| 参数 | 默认值 | 作用 |
| --- | --- | --- |
| 通用标签阈值 | 0.35 | 筛选通用标签及 CL 的其他非角色类别；值越高标签越少 |
| 角色标签阈值 | 0.85 | 独立控制角色识别 |
| 标签类别 | 全部 | WD 通常提供通用和角色标签；CL 还提供作品、画师、元信息、质量和模型标签，以模型标签表为准 |
| 附加标签 | 空 | 用逗号分隔，加入结果并赋予 1.0 的置信度 |
| 排除标签 | 空 | 用原始标签名匹配，在下划线替换、转义前排除；也能排除附加标签 |
| 按字母顺序排序 | 关闭 | 默认按置信度降序；开启后按原始标签名称排序 |
| 将标签置信度作为提示词权重 | 关闭 | 输出 `(标签:置信度)` 格式 |
| 将下划线替换为空格 | 开启 | 将 booru 标签转为可读提示词 |
| 保留下划线的标签 | 原扩展的颜文字列表 | 这些标签保留下划线 |
| 转义括号和反斜杠 | 开启 | 避免标签内容被当成提示词权重语法 |
| 使用 CPU 进行反推 | 关闭 | 默认使用已安装 ONNX Runtime 支持的设备；开启后仅用 CPU；启动参数 `--always-cpu` 仍优先 |
| 反推后卸载模型 | 关闭 | 每次反推后释放 ONNX 会话 |
| 显示分级与标签置信度 | 关闭 | 显示内容分级与最终筛选标签的置信度；分级不会写入提示词 |

“卸载反推模型”按钮可手动释放会话，磁盘缓存保留。
模型或运行设备切换时自动释放旧会话，内存中最多保留一个反推模型。
阈值按标签类别判断，不依赖标签表中各类别连续排列；没有角色标签的模型也可使用。

## 源码移植

来源：本地 `sd-webui-wd14-tagger` 提交 `ded550b7f6b8873a589ab258377836e49a85ccba`。

- `tagger/utils.py` 的全部模型注册项移至 `extras/wd14_tagger/models.py`。
- `tagger/cl.py` 的标签映射、图像预处理、sigmoid 与分类逻辑复制至 `extras/wd14_tagger/cl.py`。
- `tagger/dbimutils.py` 复制至同名文件，WD 继续使用原扩展的 BGR、白边填充及 OpenCV 缩放逻辑。
- `tagger/interrogator.py` 的标签格式处理复制至 `extras/wd14_tagger/interrogator.py`。
- 模型下载、会话生命周期和 WebUI 依赖改为 Fooocus 适配，沿用项目现有 ONNX Runtime，不在运行时安装依赖。

上游 README 声明其代码为 public domain，但借用部分（例如 `dbimutils.py`）除外；保留对应源码与来源说明。
模型权重与标签的许可由各模型仓库提供。
