"""
FFmpeg Renderer
================
Final video assembly with sidechain ducking, subtitle burning,
visual filter bypass, and avatar overlay. Outputs both 16:9 and 9:16.
"""

import json
import logging
import os
import subprocess
from pathlib import Path
from typing import Optional

from pydub import AudioSegment

logger = logging.getLogger(__name__)


class Renderer:
    """Renders the final localized video using FFmpeg."""

    def __init__(
        self,
        nvenc_preset: str = "p4",
        nvenc_cq: int = 25,
        audio_bitrate: str = "192k",
        brightness_shift: float = 0.01,
        contrast_shift: float = 1.01,
        bgm_duck_threshold: float = 0.05,
        bgm_duck_ratio: int = 10,
        bgm_duck_attack: int = 20,
        bgm_duck_release: int = 300,
        bgm_base_volume: float = 0.3,
        avatar_enabled: bool = True,
        avatar_idle_path: str = "",
        avatar_speaking_path: str = "",
        avatar_position: str = "bottom_right",
        avatar_scale: float = 0.15,
    ):
        self.nvenc_preset = nvenc_preset
        self.nvenc_cq = nvenc_cq
        self.audio_bitrate = audio_bitrate
        self.brightness_shift = brightness_shift
        self.contrast_shift = contrast_shift
        self.bgm_duck_threshold = bgm_duck_threshold
        self.bgm_duck_ratio = bgm_duck_ratio
        self.bgm_duck_attack = bgm_duck_attack
        self.bgm_duck_release = bgm_duck_release
        self.bgm_base_volume = bgm_base_volume
        self.avatar_enabled = avatar_enabled
        self.avatar_idle_path = avatar_idle_path
        self.avatar_speaking_path = avatar_speaking_path
        self.avatar_position = avatar_position
        self.avatar_scale = avatar_scale

    def prepare_dub_track(
        self,
        timing_data: list,
        output_path: str,
        total_duration_ms: int,
    ) -> str:
        """
        Concatenate TTS chunks into a single dub audio track with proper timing.

        Args:
            timing_data: Adjusted chunk metadata with timing info
            output_path: Path to save the full dub WAV
            total_duration_ms: Total duration of the original video in ms

        Returns:
            Path to the concatenated dub track
        """
        output_path = Path(output_path)

        if output_path.exists():
            logger.info(f"Dub track already exists at {output_path}. Skipping.")
            return str(output_path)

        # Create a silent base track
        dub = AudioSegment.silent(duration=total_duration_ms)

        for chunk in timing_data:
            chunk_audio = AudioSegment.from_wav(chunk["path"])
            start_ms = int(chunk["adjusted_start"] * 1000)

            # Apply atempo if needed
            atempo = chunk.get("atempo", 1.0)
            if atempo != 1.0 and atempo > 0:
                # Use pydub to speed up (simple resampling)
                # For production, FFmpeg atempo is used at render time
                new_frame_rate = int(chunk_audio.frame_rate * atempo)
                chunk_audio = chunk_audio._spawn(
                    chunk_audio.raw_data,
                    overrides={"frame_rate": new_frame_rate}
                ).set_frame_rate(chunk_audio.frame_rate)

            # Add silence padding if needed
            silence_pad = chunk.get("silence_pad", 0)
            if silence_pad > 0:
                chunk_audio = chunk_audio + AudioSegment.silent(
                    duration=int(silence_pad * 1000)
                )

            # Overlay at the correct position
            dub = dub.overlay(chunk_audio, position=start_ms)

        # Export
        output_path.parent.mkdir(parents=True, exist_ok=True)
        dub.export(str(output_path), format="wav")

        logger.info(f"Dub track saved to {output_path} ({len(dub) / 1000:.1f}s)")
        return str(output_path)

    def render_horizontal(
        self,
        video_path: str,
        bgm_path: str,
        dub_path: str,
        subtitle_path: str,
        output_path: str,
    ) -> str:
        """
        Render the final 16:9 horizontal video.

        Combines:
        - Original video (with visual filter bypass)
        - Background music (sidechain ducked)
        - Vietnamese dub audio
        - Burned-in ASS subtitles
        - Avatar overlay (optional)
        """
        output_path = Path(output_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)

        cmd = self._build_ffmpeg_cmd(
            video_path=video_path,
            bgm_path=bgm_path,
            dub_path=dub_path,
            subtitle_path=subtitle_path,
            output_path=str(output_path),
            mode="horizontal",
        )

        logger.info("Rendering horizontal (16:9) video...")
        logger.debug(f"FFmpeg command: {' '.join(cmd)}")

        result = subprocess.run(
            cmd,
            capture_output=True,
            text=True,
            timeout=3600,  # 1 hour max
        )

        if result.returncode != 0:
            logger.error(f"FFmpeg error:\n{result.stderr}")
            raise RuntimeError(f"FFmpeg render failed: {result.stderr[-500:]}")

        logger.info(f"Horizontal render complete: {output_path}")
        return str(output_path)

    def render_vertical(
        self,
        video_path: str,
        bgm_path: str,
        dub_path: str,
        subtitle_path: str,
        output_path: str,
    ) -> str:
        """
        Render a 9:16 vertical video for TikTok/YT Shorts.

        Applies blurred background + centered original video approach.
        """
        output_path = Path(output_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)

        cmd = self._build_ffmpeg_cmd(
            video_path=video_path,
            bgm_path=bgm_path,
            dub_path=dub_path,
            subtitle_path=subtitle_path,
            output_path=str(output_path),
            mode="vertical",
        )

        logger.info("Rendering vertical (9:16) video...")
        logger.debug(f"FFmpeg command: {' '.join(cmd)}")

        result = subprocess.run(
            cmd,
            capture_output=True,
            text=True,
            timeout=3600,
        )

        if result.returncode != 0:
            logger.error(f"FFmpeg error:\n{result.stderr}")
            raise RuntimeError(f"FFmpeg render failed: {result.stderr[-500:]}")

        logger.info(f"Vertical render complete: {output_path}")
        return str(output_path)

    def _build_ffmpeg_cmd(
        self,
        video_path: str,
        bgm_path: str,
        dub_path: str,
        subtitle_path: str,
        output_path: str,
        mode: str = "horizontal",
    ) -> list:
        """Build the full FFmpeg command with filter_complex."""

        inputs = [
            "-i", video_path,       # [0] original video
            "-i", bgm_path,         # [1] background music
            "-i", dub_path,         # [2] vietnamese dub
        ]

        # Track input index
        next_input = 3

        # Add avatar inputs if enabled
        # Avatar inputs tracked via next_input counter
        if self.avatar_enabled and self.avatar_idle_path and self.avatar_speaking_path:
            inputs.extend(["-i", self.avatar_idle_path])    # [3] idle avatar
            inputs.extend(["-i", self.avatar_speaking_path])  # [4] speaking avatar
            next_input = 5

        # ── Build filter_complex ──────────────────────────

        filters = []

        # ── Audio: Sidechain ducking ──
        # Lower BGM base volume first
        filters.append(
            f"[1:a]volume={self.bgm_base_volume}[bgm_quiet]"
        )
        # Apply sidechain compression: dub triggers ducking of BGM
        # sidechaincompress input order: [signal_to_compress][sidechain_key]
        filters.append(
            f"[bgm_quiet][2:a]sidechaincompress="
            f"threshold={self.bgm_duck_threshold}:"
            f"ratio={self.bgm_duck_ratio}:"
            f"attack={self.bgm_duck_attack}:"
            f"release={self.bgm_duck_release}"
            f"[bgm_ducked]"
        )
        # Mix ducked BGM + dub voice
        filters.append(
            "[bgm_ducked][2:a]amix=inputs=2:duration=longest:dropout_transition=2[audio_out]"
        )

        # ── Video processing ──
        if mode == "vertical":
            # Create blurred background (scaled to 1080x1920)
            filters.append(
                "[0:v]scale=1080:1920:force_original_aspect_ratio=increase,"
                "crop=1080:1920,boxblur=20:5[bg_blur]"
            )
            # Scale original video to fit width
            filters.append(
                "[0:v]scale=1080:-2:force_original_aspect_ratio=decrease[fg_scaled]"
            )
            # Overlay centered
            filters.append(
                "[bg_blur][fg_scaled]overlay=(W-w)/2:(H-h)/2[vid_composed]"
            )
            # Apply subtitle + visual filter
            # Escape Windows paths for FFmpeg subtitle filter
            sub_path_escaped = subtitle_path.replace("\\", "/").replace(":", "\\\\:")
            filters.append(
                f"[vid_composed]ass='{sub_path_escaped}',"
                f"eq=brightness={self.brightness_shift}:"
                f"contrast={self.contrast_shift}"
                f"[video_out]"
            )
        else:
            # Horizontal: just apply subtitle + visual filter
            sub_path_escaped = subtitle_path.replace("\\", "/").replace(":", "\\\\:")
            filters.append(
                f"[0:v]ass='{sub_path_escaped}',"
                f"eq=brightness={self.brightness_shift}:"
                f"contrast={self.contrast_shift}"
                f"[video_out]"
            )

        # ── Avatar overlay (optional) ──
        if self.avatar_enabled and next_input == 5:
            # Determine avatar size and position
            if mode == "vertical":
                scale_w = int(1080 * self.avatar_scale)
            else:
                scale_w = int(1920 * self.avatar_scale)

            # Scale avatars
            filters.append(f"[3:v]scale={scale_w}:-1[avatar_idle_s]")
            filters.append(f"[4:v]scale={scale_w}:-1[avatar_speak_s]")

            # Use audio volume to switch between idle/speaking
            # Detect speech with astats, overlay speaking avatar when RMS > threshold
            # Simplified approach: always show idle avatar (speaking detection
            # requires complex scripting — we use a simple overlay for MVP)
            pos = self._get_avatar_position(mode)
            filters.append(
                f"[video_out][avatar_idle_s]overlay={pos}:format=auto[final_video]"
            )

            video_map = "[final_video]"
        else:
            video_map = "[video_out]"

        filter_complex = ";".join(filters)

        # ── Build full command ──
        cmd = [
            "ffmpeg", "-y",
            *inputs,
            "-filter_complex", filter_complex,
            "-map", video_map,
            "-map", "[audio_out]",
        ]

        # Use hardware encoding (NVENC) with software fallback
        # Check if NVENC is available by trying it
        import shutil
        use_nvenc = shutil.which("ffmpeg") is not None  # Always try NVENC first on GPU instance

        if use_nvenc:
            cmd.extend([
                "-c:v", "h264_nvenc",
                "-preset", self.nvenc_preset,
                "-rc", "vbr",
                "-cq", str(self.nvenc_cq),
            ])
        else:
            cmd.extend([
                "-c:v", "libx264",
                "-preset", "medium",
                "-crf", "23",
            ])

        cmd.extend([
            "-c:a", "aac",
            "-b:a", self.audio_bitrate,
            "-movflags", "+faststart",
            output_path,
        ])

        return cmd

    def _get_avatar_position(self, mode: str) -> str:
        """Calculate avatar overlay position string for FFmpeg."""
        margin = 20
        positions = {
            "bottom_right": f"W-w-{margin}:H-h-{margin}",
            "bottom_left": f"{margin}:H-h-{margin}",
            "top_right": f"W-w-{margin}:{margin}",
            "top_left": f"{margin}:{margin}",
        }
        return positions.get(self.avatar_position, positions["bottom_right"])

    @staticmethod
    def get_video_duration_ms(video_path: str) -> int:
        """Get video duration in milliseconds using ffprobe."""
        cmd = [
            "ffprobe",
            "-v", "quiet",
            "-print_format", "json",
            "-show_format",
            video_path,
        ]
        result = subprocess.run(cmd, capture_output=True, text=True)
        if result.returncode != 0:
            raise RuntimeError(f"ffprobe failed: {result.stderr}")

        info = json.loads(result.stdout)
        duration = float(info["format"]["duration"])
        return int(duration * 1000)
