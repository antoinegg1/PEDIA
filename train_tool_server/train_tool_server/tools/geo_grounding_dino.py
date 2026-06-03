"""
Grounding DINO agent — open-vocabulary object detection via Ray GPU actor.

Tools: grounding_dino.

tool_type = "geo_grounding_dino"
"""

from .base import register_tool
from .pedia_base import PediaAgentToolBase


@register_tool
class GeoGroundingDinoTool(PediaAgentToolBase):
    tool_type = "geo_grounding_dino"
    agent_name = "grounding_dino"
    enable_tools = ["grounding_dino"]
