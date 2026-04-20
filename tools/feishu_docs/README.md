# Feishu Docs Tools

这个目录是**完全独立**于 `src/steporlm_stage1` 主训练链的飞书辅助工具目录。

- 删除整个 `tools/feishu_docs/` 不会影响数据生成、SFT、DPO、评测或模型对比代码。
- 这里的脚本只负责两件事：生成项目报告、把现成 Markdown 上传到飞书文档。

## 文件说明

- `build_orllm_project_report.py`
  - 扫描当前仓库里的 `configs/`、`runs/`、`outputs/`，生成一份面向项目展示的 Markdown 报告。
- `ORLLM_STAGE1_PROJECT_REPORT.md`
  - 自动生成的项目报告，带图表引用、结果解读、局限说明和云端路线图。
- `feishu_doc_tools.py`
  - 飞书文档最小客户端和 Markdown 解析逻辑。
- `publish_markdown.py`
  - 使用应用身份或应用凭证，把现成 Markdown 直接发布到飞书。
- `oauth_publish.py`
  - 走 `user_access_token` OAuth，把 Markdown 发布到用户可见文件夹。

## 常用命令

在仓库根目录执行：

```powershell
python tools/feishu_docs/build_orllm_project_report.py
python tools/feishu_docs/publish_markdown.py --source-file tools/feishu_docs/ORLLM_STAGE1_PROJECT_REPORT.md --dry-run
python tools/feishu_docs/oauth_publish.py --source-file tools/feishu_docs/ORLLM_STAGE1_PROJECT_REPORT.md --folder-token OhuWfcOJSlejUIdZ2wZcDqumnvg --open-browser
```

## 说明

- 报告里的本地图片会在 Markdown 阅读器里正常渲染。
- 当前飞书上传脚本会把 Markdown 图片语法转换成“图片说明 + 本地路径”的引用文本，不会自动创建飞书图片块。
- 如果后续需要把本地 PNG 自动上传成飞书图片块，可以在这个目录里继续扩展，不会影响训练主链。
