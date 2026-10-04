# 文档维护与验收

本书采用单份 Markdown、真实源码嵌入和可执行原理实验。
组织方法参考《动手学深度学习》和 mini-verl；不复制第三方内容或宣称达到它们的覆盖规模。

## 更新一章

1. 保留“本章问题”和前置条件，避免从 CLI 跳进所有分支。
2. 解释输入/输出 shape、梯度归属、mask、reduction 和状态生命周期。
3. 使用源码嵌入标记指向真实函数，不粘贴另一份会过期的实现。
4. 修改对应 `lesson_XX` 小实验，加入能够区分对错的断言。
5. 写明预期观察、小结、练习与答案提示，以及不能得出的结论。

源码嵌入标记的格式，在 Markdown 原文件中查看本段的注释示例：

```text
<!-- source: laptop_llm/model.py::RMSNorm.forward -->
```

构建使用 Python AST 定位符号，读取原文件，不 import 或执行模型模块。
代码块保持原始行号；源文件链接固定到本次 commit。改名后未修文档会使构建失败。
代码省略、生产 kernel 差异和独立实验都要在正文说明。

## 本地验收

```bash
python -m pip install -e ".[dev]"
python -m pip install -r requirements-docs.txt
python scripts/lesson_examples.py all
python -m pytest
python -m mkdocs build --strict
python scripts/check_docs.py --site site
```

`check_docs.py` 检查章节结构、AST 符号、README 和实际 HTML 中未渲染的双星号，以及全站内部
链接、片段锚点和资源。源码页的 `**` 运算符不误报为 Markdown 错误。
GitHub 与 Python-Markdown 是不同渲染器，README 还应在 GitHub 的实际页面验收，
不能只检查源文件中星号成对出现。

## 截图验收

启动本地真实 checkpoint，浏览器验证发送与 SSE 结束后，截完整页面。
桌面与手机分别检查横向溢出、控件可见性、代码块滚动和目录操作。
图片不拼接虚构回答，不把短训随机输出替换成精心编造的强模型回复。
截图中的模型规模/设备不能与实际 checkpoint 不符。

## 发布

`Documentation` 工作流先 strict build 和链接审计，再用 GitHub Pages 发布静态产物。
PR 只构建不发布。站点不托管推理、不收集对话，不请求付费 API；模型聊天仍是本地服务。
公开后检查真实 URL、章节源码链接、搜索结果与默认分支状态。

页面主体使用系统字体，不加载第三方字体；搜索索引与高亮资源随站点部署。
