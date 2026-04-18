# 上手 — 5 分钟跑通第一条视频

## Step 1 — 装系统依赖(约 2 分钟)

```bash
# macOS
brew install ffmpeg

# 可选:Claude Code CLI(用于结构化笔记,吃 Max 订阅配额)
npm install -g @anthropic-ai/claude-code
claude login
```

> 不想装 Claude CLI 的话,把 `configs/default.yaml` 里
> `structuring.provider` 改成 `gemini` 即可。

## Step 2 — 建虚拟环境装 Python 依赖(约 2 分钟)

```bash
cd video_kb   # 项目根目录

# 推荐用 uv(更快)
uv venv -p 3.10
source .venv/bin/activate
uv pip install -e .

# 或者标准 pip
python3.10 -m venv .venv
source .venv/bin/activate
pip install -e .
```

> 首次会下载 Whisper 模型(~1.5GB)和 bge-m3(~2GB),存在
> `~/.cache/huggingface` 下,以后不再重复下载。

## Step 3 — 填 .env

```bash
cp .env.example .env
# 编辑 .env,填入你从 https://aistudio.google.com/apikey 拿到的 key
```

## Step 4 — 体检

```bash
bash scripts/doctor.sh
```

期望看到所有项 ✅。任何 ❌ 先解决再跑下一步。

## Step 5 — 跑第一条视频

把测试视频放在任意位置,比如 `~/Downloads/test.mp4`:

```bash
kb ingest ~/Downloads/test.mp4
```

或 YouTube URL:

```bash
kb ingest "https://www.youtube.com/watch?v=XXXXXXX"
```

跑完后检查:

```bash
kb list-videos
kb stats
```

## Step 6 — 查询

```bash
kb query "这个视频讲了什么关于 XX 的内容"
```

---

## 常见问题

### Whisper 跑得慢?

macOS M3 16GB 建议配置:

```yaml
# configs/default.yaml
stt:
  model_size: "large-v3-turbo"   # 比 large-v3 快 4 倍,质量几乎一样
  device: "cpu"
  compute_type: "int8"           # M3 的 MPS 支持有限,int8 CPU 反而最稳
```

短视频测试时可以先用 `base` / `small`:

```yaml
  model_size: "base"             # 极快,质量够看
```

### Gemini 额度用光?

免费层是 1500 RPD(每日),2 小时视频大约 60-100 帧。如果单日跑了很多长视频爆了,可以:

1. 等第二天自动恢复
2. 临时把 `vision.provider` 改成 `none` 跳过视觉
3. 调 `scenes.max_frames` 降帧数

### Claude CLI 报 "missing command"?

先 `which claude` 验证。没装就跑:

```bash
npm install -g @anthropic-ai/claude-code
```

也可改配置用 Gemini 做结构化。

### 本地视频希望软链接不拷贝?

pipeline 默认就是软链接(Mac / Linux 下)。Windows 回退为 copy。

---

## 目录清理

```bash
# 清空某个视频的所有中间产物(想重跑)
rm -rf kb/videos/<video_id>/
rm -rf claude_upload/<video_id>/

# 清空整个向量库(所有视频的嵌入)
rm -rf kb/kb_db/
```

或用 `--force` 忽略缓存重跑:

```bash
kb ingest input.mp4 --force
```
