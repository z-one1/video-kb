"""统一的 Pydantic 数据模型 — pipeline 各阶段之间传递用。"""
from __future__ import annotations

from pathlib import Path
from typing import Literal, Optional

from pydantic import BaseModel, Field


class VideoMeta(BaseModel):
    """单视频元数据,保存在 kb/videos/<video_id>/meta.yaml"""

    video_id: str = Field(..., description="唯一 ID,本地文件用文件名 hash 或显式指定")
    source: Literal["youtube", "local"] = "local"
    url: Optional[str] = None
    source_path: Optional[str] = None  # 本地原始文件绝对路径
    title: str = ""
    duration_sec: float = 0.0
    language: Optional[str] = None  # Whisper 检测到的语言
    ingested_at: str = ""  # ISO 时间

    # Pipeline 处理状态
    has_transcript: bool = False
    has_frames: bool = False
    has_vision: bool = False
    has_notes: bool = False
    has_embeddings: bool = False


class TranscriptSegment(BaseModel):
    """一段带时间戳的字幕"""

    start_sec: float
    end_sec: float
    text: str
    words: list[dict] = Field(default_factory=list)  # 可选 word timestamps


class Transcript(BaseModel):
    """整段转写结果"""

    video_id: str
    language: str
    duration_sec: float
    segments: list[TranscriptSegment]

    def full_text(self) -> str:
        return "\n".join(s.text.strip() for s in self.segments)


class KeyFrame(BaseModel):
    """一个关键帧的元信息"""

    frame_id: str  # 如 "0015" 表示 15 秒
    t_sec: float
    image_path: str  # 相对 video 目录
    scene_start_sec: Optional[float] = None
    scene_end_sec: Optional[float] = None
    source: Literal["scene_change", "fixed_interval"] = "scene_change"


class VisualDescription(BaseModel):
    """视觉模型对某关键帧的描述"""

    frame_id: str
    t_sec: float
    image_path: str
    description: str
    extracted_text: Optional[str] = None  # 如果截图里有文字
    model: str = ""
    error: Optional[str] = None


class EnrichedSegment(BaseModel):
    """时间戳对齐后的字幕段 — 带对应视觉描述"""

    start_sec: float
    end_sec: float
    text: str
    visual_descriptions: list[VisualDescription] = Field(default_factory=list)


class NoteSection(BaseModel):
    """LLM 结构化产出的一个章节"""

    title: str
    start_sec: float
    end_sec: float
    summary: str
    concepts: list[str] = Field(default_factory=list)
    questions: list[str] = Field(default_factory=list)


class Notes(BaseModel):
    """整视频的结构化笔记"""

    video_id: str
    title: str
    one_liner: str
    sections: list[NoteSection]
    toc_markdown: str = ""
    full_markdown: str = ""


class Chunk(BaseModel):
    """分块后的文本单元 — 准备嵌入用。

    支持多源 (视频 / PDF / 独立图片) — source_type 区分,其他字段按需填:
      - video: start_sec / end_sec / section_title / has_visual
      - pdf:   page_num / source_path(如 'notes.pdf')
      - image: source_path(如 'chart.png'),has_visual=True
    video_id 字段对所有源都复用为 source_id(doc 侧用 'pdf_<hash>' / 'img_<hash>' 前缀)。
    """

    chunk_id: str
    video_id: str  # 实际语义 = source_id,保留字段名以兼容旧数据
    text: str
    start_sec: Optional[float] = None
    end_sec: Optional[float] = None
    section_title: Optional[str] = None
    has_visual: bool = False
    # --- 多源扩展(向后兼容,旧数据 source_type 默认为 'video')---
    source_type: str = "video"  # 'video' | 'pdf' | 'image'
    page_num: Optional[int] = None  # 仅 PDF
    source_path: Optional[str] = None  # 原始文件名,用于 doc 类型的引用显示


class DocMeta(BaseModel):
    """独立文档 (PDF/图片) 元数据,保存在 kb/docs/<doc_id>/meta.yaml"""

    doc_id: str
    source_type: Literal["pdf", "image"]
    source_path: str  # 原始文件绝对路径
    title: str = ""  # 显示用文件名(去扩展名)
    page_count: int = 0  # PDF 的页数;image 固定 1
    ingested_at: str = ""  # ISO 时间
    has_text: bool = False  # PDF 已抽文本 / image 已描述
    has_embeddings: bool = False
