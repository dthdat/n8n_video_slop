"""
Subtitle Generator (ASS Format)
================================
Creates styled .ass subtitles with Be Vietnam Pro font and semi-transparent boxes.
"""

import json
import logging
from pathlib import Path

import pysubs2

logger = logging.getLogger(__name__)


class SubtitleGenerator:
    """Generates styled ASS subtitle files from translation data."""

    def __init__(
        self,
        font_name: str = "Be Vietnam Pro",
        font_size: int = 20,
        primary_color: str = "&H00FFFFFF",    # White
        outline_color: str = "&H00000000",    # Black
        outline_width: int = 3,
        box_color: str = "&H80000000",        # Semi-transparent black
        margin_v: int = 40,
        fade_ms: int = 100,
    ):
        self.font_name = font_name
        self.font_size = font_size
        self.primary_color = primary_color
        self.outline_color = outline_color
        self.outline_width = outline_width
        self.box_color = box_color
        self.margin_v = margin_v
        self.fade_ms = fade_ms

    def generate(
        self,
        timing_data: list,
        output_path: str,
        video_width: int = 1920,
        video_height: int = 1080,
    ) -> str:
        """
        Generate an ASS subtitle file from timing-adjusted chunk data.

        Args:
            timing_data: List of chunk dicts with adjusted timing info
            output_path: Path to save the .ass file
            video_width: Video width for layout
            video_height: Video height for layout

        Returns:
            Path to the generated .ass file
        """
        output_path = Path(output_path)

        subs = pysubs2.SSAFile()
        subs.info["PlayResX"] = str(video_width)
        subs.info["PlayResY"] = str(video_height)
        subs.info["ScaledBorderAndShadow"] = "yes"

        # Define the main subtitle style
        style = pysubs2.SSAStyle()
        style.fontname = self.font_name
        style.fontsize = self.font_size
        style.primarycolor = pysubs2.Color(*self._parse_ass_color(self.primary_color))
        style.outlinecolor = pysubs2.Color(*self._parse_ass_color(self.outline_color))
        style.backcolor = pysubs2.Color(*self._parse_ass_color(self.box_color))
        style.outline = self.outline_width
        style.shadow = 0
        style.alignment = 2  # Bottom center
        style.marginv = self.margin_v
        style.borderstyle = 3  # Opaque box behind text
        style.bold = True

        subs.styles["Default"] = style

        # Create subtitle events from timing data
        for chunk in timing_data:
            text = chunk.get("text", "")
            if not text.strip():
                continue

            # Use adjusted timing (accounts for atempo, bleed, padding)
            start_ms = int(chunk["adjusted_start"] * 1000)
            end_ms = int(chunk["adjusted_end"] * 1000)

            # For padded chunks, end at original end time
            if chunk.get("action") == "pad_silence":
                end_ms = int(chunk["original_end"] * 1000)

            # Ensure minimum display time of 500ms
            if end_ms - start_ms < 500:
                end_ms = start_ms + 500

            # Add fade effect
            fade_tag = f"{{\\fad({self.fade_ms},{self.fade_ms})}}"

            event = pysubs2.SSAEvent(
                start=start_ms,
                end=end_ms,
                text=f"{fade_tag}{text}",
                style="Default",
            )
            subs.events.append(event)

        # Sort by start time
        subs.sort()

        # Save
        output_path.parent.mkdir(parents=True, exist_ok=True)
        subs.save(str(output_path))

        logger.info(
            f"Subtitles saved to {output_path} "
            f"({len(subs.events)} events)"
        )
        return str(output_path)

    def generate_vertical(
        self,
        timing_data: list,
        output_path: str,
        video_width: int = 1080,
        video_height: int = 1920,
    ) -> str:
        """Generate subtitles optimized for 9:16 vertical video."""
        # Adjust font size and margins for vertical
        original_size = self.font_size
        original_margin = self.margin_v

        self.font_size = int(self.font_size * 1.2)  # Slightly larger for mobile
        self.margin_v = int(self.margin_v * 2.5)     # Higher up to avoid UI overlays

        result = self.generate(
            timing_data=timing_data,
            output_path=output_path,
            video_width=video_width,
            video_height=video_height,
        )

        # Restore
        self.font_size = original_size
        self.margin_v = original_margin

        return result

    @staticmethod
    def _parse_ass_color(color_str: str) -> tuple:
        """
        Parse ASS color string like '&H80000000' to (r, g, b, a).
        ASS format: &HAABBGGRR
        """
        color_str = color_str.replace("&H", "").replace("&h", "")
        color_str = color_str.ljust(8, "0")

        a = int(color_str[0:2], 16)
        b = int(color_str[2:4], 16)
        g = int(color_str[4:6], 16)
        r = int(color_str[6:8], 16)

        return (r, g, b, a)
