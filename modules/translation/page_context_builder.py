"""
Page-context translation prompt builder.

Builds a full-page prompt that feeds an entire page's text + optional image
to an LLM translator, preserving bubble context and character voice.
Inspired by Comic Translate's page-context translation flow.
"""

from typing import List, Optional, Dict, Any
from dataclasses import dataclass, field

import numpy as np

from utils.textblock import TextBlock


@dataclass
class BlockContext:
    """Context for a single text block within a page."""
    index: int
    text: str
    bubble_type: Optional[str] = None  # "speech", "thought", "caption", "sfx", "narration"
    speaker_hint: Optional[str] = None
    surrounding_context: Optional[str] = None  # brief description of nearby panels


@dataclass
class PageContext:
    """Full context for a page to be translated."""
    source_language: str = "Japanese"
    target_language: str = "English"
    blocks: List[BlockContext] = field(default_factory=list)
    page_description: Optional[str] = None  # e.g. "Chapter 3, page 12"
    previous_page_summary: Optional[str] = None
    style_guide: Optional[str] = None


def build_prompt(
    textblk_lst: List[TextBlock],
    source_language: str = "Japanese",
    target_language: str = "English",
    page_description: Optional[str] = None,
    previous_page_summary: Optional[str] = None,
    style_guide: Optional[str] = None,
    image: Optional[np.ndarray] = None,
    bubble_type_map: Optional[Dict[int, str]] = None,
) -> str:
    """
    Build a page-context translation prompt for LLM translators.

    Args:
        textblk_lst: List of TextBlock objects for the current page.
        source_language: Source language name (e.g. "Japanese", "Korean").
        target_language: Target language name (e.g. "English", "Spanish").
        page_description: Optional page identifier (e.g. "Ch.3 p.12").
        previous_page_summary: Optional summary of previous page for continuity.
        style_guide: Optional project-specific style instructions.
        image: Optional page image (for multimodal models; not embedded here).
        bubble_type_map: Optional dict mapping block index → bubble type.

    Returns:
        A formatted prompt string ready for an LLM.
    """
    lines: List[str] = []

    # Header
    lines.append(f"Translate the following {source_language} manga/comic page into {target_language}.")
    lines.append("")

    if page_description:
        lines.append(f"Page: {page_description}")

    if previous_page_summary:
        lines.append(f"Previous page context: {previous_page_summary}")

    # Style guide
    default_style = (
        "Preserve speech bubble context, character voice, and emotional tone. "
        "Maintain honorifics and formality levels where appropriate. "
        "Keep SFX descriptive if untranslated, or provide onomatopoeia in target language."
    )
    lines.append(f"Style guide: {style_guide or default_style}")
    lines.append("")

    # Output format instructions
    lines.append("Return ONLY a JSON object with this exact shape:")
    lines.append('  {"translations": ["translated block 1", "translated block 2", ...]}')
    lines.append("The array must have exactly one entry per text block, in the same order as listed below.")
    lines.append("")

    # Blocks
    lines.append("Text blocks:")
    for idx, blk in enumerate(textblk_lst):
        text = blk.get_text()
        if not text or not text.strip():
            text = "[empty]"

        bubble_type = ""
        if bubble_type_map and idx in bubble_type_map:
            bubble_type = f"  (bubble type: {bubble_type_map[idx]})"

        lines.append(f"  {idx + 1}. \"{text}\"{bubble_type}")

    lines.append("")
    lines.append(f"Total blocks: {len(textblk_lst)}")

    if image is not None:
        lines.append("")
        lines.append(
            "[Image attached: multimodal models should use visual context to infer "
            "speaker identity, emotion, and scene setting.]"
        )

    return "\n".join(lines)


def build_compact_prompt(
    textblk_lst: List[TextBlock],
    source_language: str = "Japanese",
    target_language: str = "English",
) -> str:
    """
    Compact single-string prompt for translators that accept raw text.

    Format:
        Page context: Japanese manga page
        Text blocks:
        1. [source text 1]
        2. [source text 2]
        Translate each block preserving speech bubble context and character voice.
    """
    lines = [
        f"Page context: {source_language} manga/comic page",
        "",
        "Text blocks:",
    ]
    for idx, blk in enumerate(textblk_lst):
        text = blk.get_text()
        if not text or not text.strip():
            text = "[empty]"
        lines.append(f"{idx + 1}. {text}")

    lines.append("")
    lines.append("Translate each block preserving speech bubble context and character voice.")
    lines.append(f"Return exactly {len(textblk_lst)} lines, one per block.")

    return "\n".join(lines)


def build_chat_messages(
    textblk_lst: List[TextBlock],
    source_language: str = "Japanese",
    target_language: str = "English",
    page_description: Optional[str] = None,
    image: Optional[np.ndarray] = None,
    system_prompt: Optional[str] = None,
) -> List[Dict[str, str]]:
    """
    Build OpenAI-style chat messages for page-context translation.

    Returns a list of message dicts with 'role' and 'content' keys.
    If *image* is provided, content uses the vision format (base64 PNG).
    """
    messages: List[Dict[str, Any]] = []

    default_system = (
        f"You are a professional manga/comic translator from {source_language} to {target_language}. "
        "Preserve character voice, honorifics, and emotional tone. "
        "Return a JSON object with a 'translations' array containing one string per text block."
    )
    messages.append({"role": "system", "content": system_prompt or default_system})

    user_text = build_prompt(
        textblk_lst,
        source_language=source_language,
        target_language=target_language,
        page_description=page_description,
        image=None,  # handled separately below
    )

    if image is not None:
        # Multimodal: embed image as base64
        try:
            import base64
            from io import BytesIO
            from PIL import Image as PILImage

            pil_img = PILImage.fromarray(image)
            buffer = BytesIO()
            pil_img.save(buffer, format="PNG")
            b64 = base64.b64encode(buffer.getvalue()).decode("utf-8")

            messages.append({
                "role": "user",
                "content": [
                    {"type": "text", "text": user_text},
                    {
                        "type": "image_url",
                        "image_url": {"url": f"data:image/png;base64,{b64}"},
                    },
                ],
            })
        except Exception:
            # Fallback to text-only if image encoding fails
            messages.append({"role": "user", "content": user_text})
    else:
        messages.append({"role": "user", "content": user_text})

    return messages
