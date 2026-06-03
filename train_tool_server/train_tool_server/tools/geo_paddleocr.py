"""
PaddleOCR agent — OCR/text recognition tools via Ray GPU actor.

Tools: text_ocr, table_ocr, formula_ocr, chart_text_ocr,
       text_spotting, seal_ocr, map_text_ocr.

tool_type = "geo_paddleocr"
"""

from .base import register_tool
from .pedia_base import PediaAgentToolBase


@register_tool
class GeoPaddleocrTool(PediaAgentToolBase):
    tool_type = "geo_paddleocr"
    agent_name = "paddleocr"
    enable_tools = [
        "text_ocr", "table_ocr", "formula_ocr", "chart_text_ocr",
        "text_spotting", "seal_ocr", "map_text_ocr",
    ]
