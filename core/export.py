"""Conversation export module.

Provides functions to export a conversation (with its messages) into
Markdown, JSON, and PDF formats.

Usage:
    from export import export_markdown, export_json, export_pdf

    md_text = export_markdown(conversation)
    json_text = export_json(conversation)
    pdf_bytes = export_pdf(conversation)

Each function expects a ``conversation`` dict shaped like::

    {
        "id": 1,
        "title": "My Chat",
        "created_at": "2025-01-01T00:00:00",
        "messages": [
            {"role": "user", "content": "Hello"},
            {"role": "assistant", "content": "Hi there!"},
        ]
    }
"""
import io
import json
import logging
import re
from datetime import datetime

logger = logging.getLogger("rag.export")


def export_markdown(conversation: dict) -> str:
    """Export a conversation to a Markdown string.

    Args:
        conversation: Dict with ``title``, ``created_at``, and ``messages``.

    Returns:
        Markdown-formatted string.
    """
    title = conversation.get("title", "Untitled Conversation")
    created = conversation.get("created_at", "")
    messages = conversation.get("messages", [])

    lines = [
        f"# {title}",
        "",
        f"*Exported on {datetime.now().strftime('%Y-%m-%d %H:%M')}*",
        "",
    ]

    if created:
        lines.append(f"*Conversation started: {created}*")
        lines.append("")

    lines.append("---")
    lines.append("")

    for msg in messages:
        role = msg.get("role", "unknown")
        content = msg.get("content", "")
        label = "**You**" if role == "user" else "**Assistant**"

        lines.append(f"### {label}")
        lines.append("")
        lines.append(content)
        lines.append("")
        lines.append("---")
        lines.append("")

    return "\n".join(lines)


def export_json(conversation: dict) -> str:
    """Export a conversation to a JSON string.

    Args:
        conversation: Dict with ``title``, ``created_at``, and ``messages``.

    Returns:
        Pretty-printed JSON string.
    """
    export_data = {
        "title": conversation.get("title", "Untitled Conversation"),
        "created_at": conversation.get("created_at", ""),
        "exported_at": datetime.now().isoformat(),
        "message_count": len(conversation.get("messages", [])),
        "messages": [
            {
                "role": msg.get("role", "unknown"),
                "content": msg.get("content", ""),
            }
            for msg in conversation.get("messages", [])
        ],
    }
    return json.dumps(export_data, indent=2, ensure_ascii=False)


def _strip_markdown(text: str) -> str:
    """Remove common markdown formatting for plain-text PDF rendering.

    Handles bold, italic, headers, links, code blocks, and inline code
    so that the text is legible in a PDF rendered with a basic font.
    """
    # Remove code blocks (``` ... ```)
    text = re.sub(r"```[\s\S]*?```", lambda m: m.group(0).strip("`").strip(), text)
    # Remove inline code backticks
    text = re.sub(r"`([^`]+)`", r"\1", text)
    # Bold **text** or __text__
    text = re.sub(r"\*\*(.+?)\*\*", r"\1", text)
    text = re.sub(r"__(.+?)__", r"\1", text)
    # Italic *text* or _text_
    text = re.sub(r"\*(.+?)\*", r"\1", text)
    text = re.sub(r"(?<!\w)_(.+?)_(?!\w)", r"\1", text)
    # Headers: ### Heading -> Heading
    text = re.sub(r"^#{1,6}\s+", "", text, flags=re.MULTILINE)
    # Links [text](url) -> text
    text = re.sub(r"\[([^\]]+)\]\([^\)]+\)", r"\1", text)
    # Horizontal rules
    text = re.sub(r"^---+$", "", text, flags=re.MULTILINE)
    return text


def export_pdf(conversation: dict) -> bytes:
    """Export a conversation to PDF bytes using ReportLab.

    Uses DejaVu Sans (bundled with ReportLab) for full Unicode support
    including Cyrillic, CJK, Arabic, and other scripts.

    Args:
        conversation: Dict with ``title``, ``created_at``, and ``messages``.

    Returns:
        PDF file content as bytes.
    """
    from reportlab.lib.pagesizes import A4
    from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
    from reportlab.lib.units import mm
    from reportlab.lib.enums import TA_LEFT
    from reportlab.platypus import (
        SimpleDocTemplate,
        Paragraph,
        Spacer,
        HRFlowable,
    )
    from reportlab.lib.colors import HexColor
    from reportlab.pdfbase import pdfmetrics
    from reportlab.pdfbase.ttfonts import TTFont

    # Register a Unicode-capable TTF font.
    # Search order: Arial (Windows), DejaVu (Linux), then Helvetica fallback.
    import os
    _FONT_NAME = "Helvetica"
    _FONT_BOLD = "Helvetica-Bold"

    _candidates = [
        # (name, bold_name, regular_path, bold_path)
        ("Arial", "Arial-Bold", "C:/Windows/Fonts/arial.ttf", "C:/Windows/Fonts/arialbd.ttf"),
        ("DejaVuSans", "DejaVuSans-Bold", "/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf", "/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf"),
    ]
    for fname, fbold, fpath, fbpath in _candidates:
        if os.path.exists(fpath) and fname not in pdfmetrics.getRegisteredFontNames():
            pdfmetrics.registerFont(TTFont(fname, fpath))
            pdfmetrics.registerFont(TTFont(fbold, fbpath if os.path.exists(fbpath) else fpath))
            _FONT_NAME = fname
            _FONT_BOLD = fbold
            break

    if _FONT_NAME == "Helvetica":
        logger.warning("No Unicode TTF font found, falling back to Helvetica (Latin-1 only)")

    title = conversation.get("title", "Untitled Conversation")
    created = conversation.get("created_at", "")
    messages = conversation.get("messages", [])

    buffer = io.BytesIO()
    doc = SimpleDocTemplate(
        buffer,
        pagesize=A4,
        leftMargin=20 * mm,
        rightMargin=20 * mm,
        topMargin=20 * mm,
        bottomMargin=20 * mm,
    )

    styles = getSampleStyleSheet()

    # Custom styles using Unicode font
    title_style = ParagraphStyle(
        "ConvTitle",
        parent=styles["Heading1"],
        fontName=_FONT_BOLD,
        fontSize=18,
        spaceAfter=6,
        textColor=HexColor("#0b1220"),
    )
    meta_style = ParagraphStyle(
        "ConvMeta",
        parent=styles["Normal"],
        fontName=_FONT_NAME,
        fontSize=9,
        textColor=HexColor("#6b7280"),
        spaceAfter=12,
    )
    role_style = ParagraphStyle(
        "RoleLabel",
        parent=styles["Heading3"],
        fontName=_FONT_BOLD,
        fontSize=11,
        spaceAfter=4,
        textColor=HexColor("#374151"),
    )
    content_style = ParagraphStyle(
        "MsgContent",
        parent=styles["Normal"],
        fontName=_FONT_NAME,
        fontSize=10,
        leading=14,
        spaceAfter=8,
        alignment=TA_LEFT,
    )

    story = []

    def safe(text: str) -> str:
        """Escape XML-special characters for ReportLab paragraphs."""
        return text.replace("&", "&amp;").replace("<", "&lt;").replace(">", "&gt;")

    story.append(Paragraph(safe(title), title_style))

    meta_parts = [f"Exported on {datetime.now().strftime('%Y-%m-%d %H:%M')}"]
    if created:
        meta_parts.append(f"Started: {created}")
    story.append(Paragraph(safe(" | ".join(meta_parts)), meta_style))
    story.append(HRFlowable(width="100%", thickness=0.5, color=HexColor("#e5e7eb")))
    story.append(Spacer(1, 8))

    for msg in messages:
        role = msg.get("role", "unknown")
        content = msg.get("content", "")
        label = "You" if role == "user" else "Assistant"

        story.append(Paragraph(safe(label), role_style))

        # Strip markdown for cleaner PDF rendering
        plain = _strip_markdown(content)
        # Split into paragraphs and add each
        for para in plain.split("\n\n"):
            para = para.strip()
            if para:
                # Replace single newlines with <br/>
                para_html = safe(para).replace("\n", "<br/>")
                story.append(Paragraph(para_html, content_style))

        story.append(Spacer(1, 4))
        story.append(HRFlowable(width="100%", thickness=0.3, color=HexColor("#e5e7eb")))
        story.append(Spacer(1, 6))

    doc.build(story)
    pdf_bytes = buffer.getvalue()
    buffer.close()

    logger.info(
        "Exported conversation '%s' to PDF (%d bytes, %d messages)",
        title,
        len(pdf_bytes),
        len(messages),
    )
    return pdf_bytes
