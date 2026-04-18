# video-kb

> Turn any YouTube video or local lecture into a **citation-backed, searchable knowledge base** — for **$0/month**.
>
> English · [中文](./README.zh-CN.md)

[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](./LICENSE)
[![Python 3.10+](https://img.shields.io/badge/python-3.10+-blue.svg)](https://www.python.org/downloads/)
[![MCP](https://img.shields.io/badge/MCP-ready-brightgreen.svg)](https://modelcontextprotocol.io/)

One command turns a video into timestamped transcripts, visually-described key frames, structured notes, and a vector DB. Ask questions in natural language and get answers with `[ep.N @ mm:ss]` citations that jump you back to the exact moment.

Natively plugs into **Claude Code / Cowork / Desktop** via MCP — ask your question right inside Claude, no extra UI needed.

---

## Three commands to try it

```bash
# 1. Install (installs Python deps, writes .env from template)
bash scripts/install.sh

# 2. Ingest a video (local file or YouTube URL)
kb ingest ~/Videos/lecture.mp4

# 3. Ask a question with cited answers
kb ask "What did the lecturer say about liquidity?"
```

Sample answer:

```
Liquidity refers to the pool of executable orders in the market [ep.3 @ 02:15].
The lecturer splits it into two categories:
- Buy-side liquidity: stops and buy orders resting above prior highs [ep.3 @ 04:42]
- Sell-side liquidity: stops and sell orders resting below prior lows [ep.3 @ 05:18]
...
📎 Reference frames: ep.3 @ 02:15, 04:42, 05:18
```

Every fact carries a `[ep.N @ mm:ss]` citation back to the original video timestamp. You can always verify.

---

## Why this exists

There are already plenty of "chat with your video" tools — NotebookLM, various YouTube summarizers, transcript-based Q&A sites. They either charge per token, get cut off at context limits, or produce handwavy summaries without citations.

**What makes video-kb different:**

1. **Zero API cost.** Whisper runs locally for transcription, bge-m3 runs locally for embeddings, Gemini's free tier handles vision, and Claude Max subscription handles synthesis. At 1–2 hours of video per day, your monthly bill is literally `$0`.
2. **Citations, not summaries.** Every claim in the answer traces back to `[ep.N @ mm:ss]`. You can always jump back and audit.
3. **Native Claude integration via MCP.** The bundled MCP server exposes 4 tools (`kb_ask`, `kb_query`, `kb_list_videos`, `kb_stats`) — ask your question directly inside Claude Code / Cowork / Desktop and Claude picks the right tool itself.
4. **Cross-episode synthesis.** Questions like *"which episodes discuss order blocks?"* automatically search across the entire library and synthesize a coherent answer.

---

## What you get

| Capability | Command |
|---|---|
| Ingest local file or YouTube URL | `kb ingest <path-or-url>` |
| Raw Top-K chunk retrieval | `kb query "..."` |
| Answer with LLM synthesis + citations | `kb ask "..."` |
| List ingested videos | `kb list-videos` |
| Vector DB stats | `kb stats` |
| Export Claude Project zip | `kb export <video_id>` |
| **Use directly inside Claude Code / Cowork / Desktop** | via MCP (see below) |

---

## Native Claude integration (MCP)

The project ships with a built-in MCP server that exposes four tools to Claude:

- `kb_ask` — retrieval + synthesis + citations
- `kb_query` — raw Top-K chunks as JSON
- `kb_list_videos` — what's in the library
- `kb_stats` — total chunks, per-video counts

Just drop in and run:

```bash
cd video-kb
claude             # Claude Code auto-detects .mcp.json and asks for approval
```

Then in any Claude Code conversation:

```
Search the video KB: what's the CRT model?
```

Claude will pick `kb_ask` on its own and return a fully-cited answer. Works identically in **Cowork** and **Claude Desktop** (paste the MCP config into their respective config files).

---

## Architecture

```
┌──────────────────────────────────────────────────────────┐
│ Input: YouTube URL / local video file                    │
└────────────────────────┬─────────────────────────────────┘
                         ▼
       ┌─────────────────────────────────────┐
       │ [1] Fetch                           │
       │   • yt-dlp (YouTube)                │
       │   • register_local (symlink)        │
       │   • ffmpeg (extract audio)          │
       └──┬──────────────────────┬───────────┘
          ▼                      ▼
┌─────────────────┐    ┌────────────────────┐
│ [2] STT         │    │ [3] Key frames     │
│   faster-whisper│    │   PySceneDetect    │
│   (local int8)  │    │   + interval fbk   │
└────────┬────────┘    └──────┬─────────────┘
         │                    ▼
         │           ┌─────────────────────┐
         │           │ [4] Vision          │
         │           │   Claude Code CLI   │
         │           │   / Gemini Flash    │
         │           └──────────┬──────────┘
         │                      │
         └──────────┬───────────┘
                    ▼
       ┌─────────────────────────────┐
       │ [5] Timestamp fusion        │
       │   subtitles ↔ visuals      │
       └─────────────┬───────────────┘
                     ▼
       ┌─────────────────────────────┐
       │ [6] LLM structuring         │
       │   Claude Code CLI (primary) │
       │   Gemini (fallback)         │
       └─────────────┬───────────────┘
                     ▼
       ┌─────────────────────────────┐
       │ [7] Chunk + embed           │
       │   bge-m3 (local)            │
       └─────────────┬───────────────┘
                     ▼
       ┌─────────────────────────────┐
       │ [8] RAG                     │
       │   ChromaDB + kb ask + MCP   │
       └─────────────────────────────┘
```

---

## Installation

### System requirements

- **Python 3.10+**
- **ffmpeg** (`brew install ffmpeg` / `apt install ffmpeg`)
- **Claude Code CLI** (optional, for zero-cost synthesis via Max subscription):
  ```bash
  npm install -g @anthropic-ai/claude-code
  claude login
  ```
  If you'd rather use Gemini for everything, set `structuring.provider: gemini` and `ask.provider: gemini` in `configs/default.yaml`.

### Python setup

```bash
git clone https://github.com/z-one1/video-kb.git
cd video-kb

# One-liner:
bash scripts/install.sh

# Or manually:
python3.10 -m venv .venv
source .venv/bin/activate
pip install -e .
```

First ingest will download Whisper (~1.5 GB) and bge-m3 (~2 GB) into `~/.cache/huggingface`. One-time cost.

### API key

```bash
cp .env.example .env
# Edit .env and paste GEMINI_API_KEY (free at https://aistudio.google.com/apikey)
```

### Health check

```bash
bash scripts/doctor.sh
```

All items should be ✅ before your first `kb ingest`.

---

## Configuration

Everything is in `configs/default.yaml`:

- Whisper model size / device / compute type
- Scene detection threshold, max frames per video (caps Gemini usage)
- Vision provider (`claude_code` / `gemini` / `none`)
- Structuring provider (`claude_code` / `gemini`)
- Chunking + embedding model parameters
- `kb ask` LLM model and timeout

Defaults are tuned for **Mac M-series, 16 GB RAM**. Tweak as needed.

---

## Cost breakdown

| Stage | Tool | Cost |
|---|---|---|
| Transcription | faster-whisper (local int8) | Free |
| Vision understanding | Gemini 2.5 Flash Lite (free tier) / Claude Max | Free / subscription |
| LLM structuring | Claude Code CLI (Max subscription) | Free / subscription |
| Embeddings | bge-m3 (local) | Free |
| Vector store | ChromaDB (local) | Free |
| Retrieval + answer | Claude Code CLI | Free / subscription |

Gemini's free tier is 1500 RPD — plenty for personal use at 1–2 hours of video per day.

---

## Output layout

```
video-kb/
├── kb/
│   ├── videos/<video_id>/
│   │   ├── video.mp4           (symlink or download)
│   │   ├── audio.wav           (16 kHz mono)
│   │   ├── meta.yaml
│   │   ├── transcript.{json,srt,md}
│   │   ├── frames/             (keyframe JPGs)
│   │   ├── visuals.json        (per-frame descriptions)
│   │   ├── enriched.{json,md}  (aligned transcript + visuals)
│   │   ├── notes.{json,md}     (LLM-structured notes)
│   │   └── chunks.jsonl        (for embedding)
│   └── kb_db/                  (ChromaDB)
└── claude_upload/<video_id>/   (drag-and-drop into a Claude Project)
```

All output directories are `.gitignore`d — your personal videos never leak into version control.

---

## Roadmap

v0.1 (shipped): ingest pipeline, 8-stage processing, ChromaDB, `kb ask`, MCP server integration.

v0.2 (next):

- [ ] Alias dictionary for concept synonyms (fixes queries using external vocabulary the lecturer doesn't use)
- [ ] Merge short chunks to improve signal density
- [ ] Streamlit query dashboard
- [ ] Watch-folder auto-ingest
- [ ] Speaker diarization via pyannote.audio

---

## Contributing

Issues and PRs welcome. If you're adding a stage or a provider, follow the `src/kb/<stage>/<provider>.py` pattern — each provider is a drop-in module with the same function signature.

---

## License

[MIT](./LICENSE) — free for personal, research, and commercial use.

## Credits

Built on:
- [faster-whisper](https://github.com/SYSTRAN/faster-whisper) — local STT
- [PySceneDetect](https://github.com/Breakthrough/PySceneDetect) — keyframe detection
- [BAAI/bge-m3](https://huggingface.co/BAAI/bge-m3) — multilingual embeddings
- [ChromaDB](https://github.com/chroma-core/chroma) — vector store
- [MCP Python SDK](https://github.com/modelcontextprotocol/python-sdk) — Claude tool protocol
