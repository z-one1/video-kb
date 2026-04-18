# video-kb — 自动视频知识库 RAG 系统

> [English](./README.md) · 中文

把 **YouTube 链接**或**本地视频**,一条命令变成可检索的知识库:带时间戳的字幕、视觉理解的关键帧、LLM 结构化笔记、向量检索 + 带引用的问答。

**核心约束**:**不花一分 API 钱**。用 Claude Max 订阅 + Gemini 免费层 + 本地模型(Whisper / bge-m3),跑自己的视频学习库。

---

## 三条命令跑通

```bash
# 1. 安装
bash scripts/install.sh

# 2. ingest 一条视频
kb ingest ~/Videos/lecture.mp4

# 3. 用自然语言问(或直接在 Claude Code 里问 — 见 MCP 章节)
kb ask "这个视频讲了什么关于流动性的?"
```

输出长这样:

```
流动性 (liquidity) 是指市场中可执行交易的订单池 [ep.3 @ 02:15]。
老师把它分成两类:
- 买方流动性:挂在前高上方的止损和买单 [ep.3 @ 04:42]
- 卖方流动性:挂在前低下方的止损和卖单 [ep.3 @ 05:18]
...
📎 参考关键帧:ep.3 @ 02:15, 04:42, 05:18
```

每条事实都带 `[ep.N @ mm:ss]` 引用,跳回原视频时间点可直接验证。

---

## 为什么做这个

开源已经有一堆 "聊视频" 工具 — NotebookLM、各种 YouTube summarizer。但它们要么按 token 付费、要么只能用一次、要么丢上下文。**video-kb 的差异化**:

1. **零成本** — Whisper 本地转录、bge-m3 本地嵌入、Gemini 免费层看图、Claude Max 订阅综合答案。1-2 小时视频/天的强度,`$0/月`。
2. **带引用的答案,不是概括** — 每个事实都回指 `[ep.N @ mm:ss]`,永远可以跳回去验证。
3. **原生接 Claude Code / Cowork / Desktop** — 通过 MCP Server 暴露 4 个工具,在 Claude 里直接问"CRT 模型是什么",Claude 自己会调检索。
4. **跨视频综合** — 问"流动性在哪几集被展开",自动跨集检索 + 综合,不用你一集集翻。

---

## 功能速览

| 能力 | 命令 / 方式 |
|---|---|
| 单视频处理(本地 / YouTube) | `kb ingest <path-or-url>` |
| 纯向量检索 Top-K 原始片段 | `kb query "..."` |
| 带 LLM 综合 + 引用的问答 | `kb ask "..."` |
| 列出已入库视频 | `kb list-videos` |
| 向量库总览 | `kb stats` |
| 导出 Claude Project 上传包 | `kb export <video_id>` |
| **Claude Code 原生集成**(MCP) | 直接在 Claude Code 里用自然语言问 |

---

## Claude Code / Cowork / Desktop 集成(MCP)

项目内置了一个 MCP Server,把 4 个工具暴露成 Claude 的原生工具:

- `kb_ask` — 带引用答案
- `kb_query` — 原始检索片段(JSON)
- `kb_list_videos` — 视频清单
- `kb_stats` — 向量库总览

仓库里已有 `.mcp.json`,只需:

```bash
cd video-kb
claude              # 启动 Claude Code,会自动发现 .mcp.json 并提示授权
```

授权后在对话里:

```
从视频知识库查:CRT 模型是什么
```

Claude 会自动调 `kb_ask`,返回带 `[ep.N @ mm:ss]` 的综合答案。

**如果要给 Claude Desktop 配**,参考 `src/kb/mcp/server.py` 顶部的 docstring。

---

## 输出结构

```
video-kb/
├── kb/
│   ├── videos/
│   │   └── <video_id>/
│   │       ├── video.mp4                  (软链接或下载的原始文件)
│   │       ├── audio.wav                  (16kHz 单声道)
│   │       ├── meta.yaml                  (视频元数据 + 处理状态)
│   │       ├── transcript.json/.srt/.md   (带时间戳字幕)
│   │       ├── frames/                    (关键帧 JPG)
│   │       ├── visuals.json               (视觉理解描述)
│   │       ├── enriched.json/.md          (字幕 + 视觉对齐后)
│   │       ├── notes.json/.md             (LLM 结构化笔记)
│   │       └── chunks.jsonl               (分块后 for 向量化)
│   └── kb_db/                             (ChromaDB 向量库)
└── claude_upload/
    └── <video_id>/                        (可上传到 Claude Project)
```

> 这些目录都在 `.gitignore` 里,不会被提交。

---

## 架构

```
┌──────────────────────────────────────────────────────────┐
│ 输入: YouTube URL / 本地视频文件                         │
└────────────────────────┬─────────────────────────────────┘
                         ▼
       ┌─────────────────────────────────────┐
       │ [1] 获取                            │
       │   • yt-dlp (YouTube)                │
       │   • register_local (本地 symlink)   │
       │   • ffmpeg 抽音频                   │
       └──┬──────────────────────┬───────────┘
          ▼                      ▼
┌─────────────────┐    ┌────────────────────┐
│ [2] STT         │    │ [3] 关键帧         │
│   faster-whisper│    │   PySceneDetect    │
│   (本地 int8)   │    │   + 间隔兜底       │
└────────┬────────┘    └──────┬─────────────┘
         │                    ▼
         │           ┌─────────────────────┐
         │           │ [4] 视觉理解        │
         │           │   Claude Code CLI   │
         │           │   / Gemini Flash    │
         │           └──────────┬──────────┘
         │                      │
         └──────────┬───────────┘
                    ▼
       ┌─────────────────────────────┐
       │ [5] 时间戳对齐              │
       │   字幕段 ↔ 视觉描述        │
       └─────────────┬───────────────┘
                     ▼
       ┌─────────────────────────────┐
       │ [6] LLM 结构化              │
       │   Claude Code CLI (主)      │
       │   Gemini (备)               │
       └─────────────┬───────────────┘
                     ▼
       ┌─────────────────────────────┐
       │ [7] 分块 + 嵌入             │
       │   bge-m3 (本地)             │
       └─────────────┬───────────────┘
                     ▼
       ┌─────────────────────────────┐
       │ [8] 向量库 + RAG            │
       │   ChromaDB + kb ask (MCP)   │
       └─────────────────────────────┘
```

---

## 配置

`configs/default.yaml` 里所有参数可调:

- Whisper 模型大小 / device / compute_type
- 场景检测阈值 / 最大帧数(保护 Gemini 额度)
- 视觉 provider(`claude_code` / `gemini` / `none`)
- 结构化 provider(`claude_code` / `gemini`)
- 分块大小 / 重叠 / 嵌入模型
- `kb ask` 的 LLM 模型 / 超时

---

## 成本

**1-2 小时视频/天的强度下,$0/月**。

| 环节 | 方案 | 成本 |
|---|---|---|
| STT 转录 | faster-whisper 本地 int8 | 0 |
| 视觉理解 | Gemini 2.5 Flash Lite 免费层 / Claude Max | 0 |
| 结构化笔记 | Claude Code CLI(Max 订阅额度) | 0(按订阅) |
| 嵌入 | bge-m3 本地 | 0 |
| 向量库 | ChromaDB 本地 | 0 |
| 检索答案 | Claude Code CLI | 0(按订阅) |

Gemini 免费层 1500 RPD,对个人够用。

---

## 开发状态

✅ 已稳定:ingest pipeline / 8 阶段 / ChromaDB / `kb ask` / MCP Server 集成

🛠 后续 v0.2:

- [ ] 概念别名词典(同义词映射,解决 "MMXM" vs "CRT" 之类词表错配)
- [ ] 短 chunk 合并(提高检索信号密度)
- [ ] Streamlit 查询 dashboard
- [ ] 目录监听,自动处理新视频
- [ ] pyannote.audio 说话人分离

---

## License

MIT — 见 [LICENSE](./LICENSE)。

## 致谢

- [faster-whisper](https://github.com/SYSTRAN/faster-whisper) — 本地 STT
- [PySceneDetect](https://github.com/Breakthrough/PySceneDetect) — 关键帧检测
- [BAAI/bge-m3](https://huggingface.co/BAAI/bge-m3) — 多语言嵌入
- [ChromaDB](https://github.com/chroma-core/chroma) — 向量库
- [MCP](https://modelcontextprotocol.io/) — Claude 工具协议
