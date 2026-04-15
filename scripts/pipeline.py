"""
Master Pipeline Script
=======================
Entry point for the video localization pipeline.
Orchestrates Job A (extraction) and Job B (rendering).

Usage:
    python pipeline.py --mode job_a --url "https://youtube.com/watch?v=..." --work-dir /workspace
    python pipeline.py --mode job_b --work-dir /workspace
"""

import argparse
import asyncio
import json
import logging
import os
import re
import subprocess
import sys
from pathlib import Path

from dotenv import load_dotenv

# ── Setup logging ────────────────────────────────────────────
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)-8s | %(name)-20s | %(message)s",
    datefmt="%H:%M:%S",
    handlers=[
        logging.StreamHandler(sys.stdout),
        logging.FileHandler("pipeline.log", encoding="utf-8"),
    ],
)
logger = logging.getLogger("pipeline")


def load_config(config_path: str = None):
    """Load configuration from .env file."""
    if config_path:
        load_dotenv(config_path)
    else:
        # Try common locations
        for path in [
            "/app/config/config.env",
            "config/config.env",
            "../config/config.env",
        ]:
            if os.path.exists(path):
                load_dotenv(path)
                break


def extract_video_id(url: str) -> str:
    """Extract YouTube video ID from URL."""
    patterns = [
        r"(?:v=|\/v\/|youtu\.be\/|\/embed\/)([a-zA-Z0-9_-]{11})",
        r"([a-zA-Z0-9_-]{11})",
    ]
    for pattern in patterns:
        match = re.search(pattern, url)
        if match:
            return match.group(1)
    # Fallback: use URL hash
    import hashlib
    return hashlib.md5(url.encode()).hexdigest()[:11]


def download_video(url: str, work_dir: Path) -> tuple[str, str]:
    """Download video using yt-dlp. Returns (video_path, audio_path)."""
    video_path = work_dir / "original_video.mp4"
    audio_path = work_dir / "original_audio.wav"

    if video_path.exists() and audio_path.exists():
        logger.info("Video and audio already downloaded. Skipping.")
        return str(video_path), str(audio_path)

    logger.info(f"Downloading video from {url}...")

    # Download video
    subprocess.run(
        [
            "yt-dlp",
            "-f", "bestvideo[ext=mp4]+bestaudio[ext=m4a]/best[ext=mp4]/best",
            "-o", str(video_path),
            "--merge-output-format", "mp4",
            "--no-playlist",
            url,
        ],
        check=True,
    )

    # Extract audio as WAV for processing
    logger.info("Extracting audio track...")
    subprocess.run(
        [
            "ffmpeg", "-y",
            "-i", str(video_path),
            "-vn",
            "-acodec", "pcm_s16le",
            "-ar", "44100",
            "-ac", "2",
            str(audio_path),
        ],
        check=True,
    )

    logger.info("Download complete.")
    return str(video_path), str(audio_path)


# ═══════════════════════════════════════════════════════════════
# JOB A: Data Extraction & Translation
# ═══════════════════════════════════════════════════════════════

def run_job_a(url: str, work_dir: Path):
    """
    Job A: Download → Separate → Transcribe → Translate

    This runs on the Vast.ai GPU instance.
    After completion, the instance is shut down and translation
    goes to Telegram for human review.
    """
    from checkpoint import CheckpointManager
    from separator import AudioSeparator
    from transcriber import Transcriber
    from translator import Translator

    video_id = extract_video_id(url)
    checkpoint = CheckpointManager(str(work_dir))

    # Check for existing checkpoint (crash recovery)
    existing = checkpoint.load()
    if existing and existing.get("video_id") == video_id:
        logger.info(f"Resuming from checkpoint: phase={checkpoint.get_phase()}")
    else:
        checkpoint.initialize(url, video_id)
        logger.info(f"Starting new job for video: {video_id}")

    # ── Step 1: Download ─────────────────────────────────────
    if not checkpoint.is_phase_done("download_done"):
        checkpoint.update_phase("downloading")
        video_path, audio_path = download_video(url, work_dir)
        checkpoint.set_artifact("video_path", video_path)
        checkpoint.set_artifact("audio_path", audio_path)
        checkpoint.update_phase("download_done")
    else:
        video_path = checkpoint.get_artifact("video_path")
        audio_path = checkpoint.get_artifact("audio_path")
        logger.info("Download already complete. Skipping.")

    # ── Step 2: Audio Separation (Demucs) ────────────────────
    if not checkpoint.is_phase_done("separation_done"):
        checkpoint.update_phase("separating")
        separator = AudioSeparator(
            model_name=os.getenv("DEMUCS_MODEL", "htdemucs_ft")
        )
        vocals_path, bgm_path = separator.separate(
            audio_path=audio_path,
            output_dir=str(work_dir),
        )
        checkpoint.set_artifact("vocals_path", vocals_path)
        checkpoint.set_artifact("bgm_path", bgm_path)
        checkpoint.update_phase("separation_done")
    else:
        vocals_path = checkpoint.get_artifact("vocals_path")
        bgm_path = checkpoint.get_artifact("bgm_path")
        logger.info("Separation already complete. Skipping.")

    # ── Step 3: Transcription (WhisperX) ─────────────────────
    if not checkpoint.is_phase_done("transcription_done"):
        checkpoint.update_phase("transcribing")
        transcriber = Transcriber(
            model_name=os.getenv("WHISPERX_MODEL", "large-v3"),
            batch_size=int(os.getenv("WHISPERX_BATCH_SIZE", "16")),
            compute_type=os.getenv("WHISPERX_COMPUTE_TYPE", "float16"),
        )
        transcript = transcriber.transcribe(
            audio_path=vocals_path,
            output_path=str(work_dir / "transcript.json"),
            language="en",
        )
        checkpoint.set_artifact("transcript_path", str(work_dir / "transcript.json"))
        checkpoint.update_phase("transcription_done")
    else:
        logger.info("Transcription already complete. Skipping.")

    # ── Step 4: Translation (Gemini) ─────────────────────────
    if not checkpoint.is_phase_done("translation_done"):
        checkpoint.update_phase("translating")
        translator = Translator(
            api_key=os.getenv("GEMINI_API_KEY", ""),
            model_name=os.getenv("GEMINI_MODEL", "gemini-1.5-pro"),
        )
        translation = translator.translate(
            transcript_path=str(work_dir / "transcript.json"),
            output_path=str(work_dir / "translation.json"),
            target_language="Vietnamese",
        )
        checkpoint.set_artifact("translation_path", str(work_dir / "translation.json"))
        checkpoint.update_phase("translation_done")
    else:
        logger.info("Translation already complete. Skipping.")

    # ── Job A Complete ───────────────────────────────────────
    logger.info("=" * 60)
    logger.info("JOB A COMPLETE")
    logger.info(f"Translation saved: {work_dir / 'translation.json'}")
    logger.info("GPU can be shut down. Awaiting human review via Telegram.")
    logger.info("=" * 60)

    # Output the translation for n8n to capture via SSH
    with open(work_dir / "translation.json", "r", encoding="utf-8") as f:
        translation_data = json.load(f)

    print("\n===TRANSLATION_OUTPUT_START===")
    print(json.dumps(translation_data, ensure_ascii=False))
    print("===TRANSLATION_OUTPUT_END===")

    return translation_data


# ═══════════════════════════════════════════════════════════════
# JOB B: Rendering & Assembly
# ═══════════════════════════════════════════════════════════════

def run_job_b(work_dir: Path):
    """
    Job B: TTS Generation → Audio Assembly → Subtitle Gen → FFmpeg Render

    This runs after human approval, on a fresh Vast.ai GPU instance.
    """
    from checkpoint import CheckpointManager
    from tts_generator import TTSGenerator
    from subtitle_generator import SubtitleGenerator
    from renderer import Renderer

    checkpoint = CheckpointManager(str(work_dir))
    state = checkpoint.load()

    if not state:
        raise RuntimeError("No checkpoint found. Run Job A first.")

    if not checkpoint.is_phase_done("translation_done"):
        raise RuntimeError(
            f"Job A not complete. Current phase: {checkpoint.get_phase()}"
        )

    logger.info(f"Starting Job B for video: {state['video_id']}")

    # Load paths from checkpoint
    video_path = checkpoint.get_artifact("video_path")
    bgm_path = checkpoint.get_artifact("bgm_path")
    translation_path = checkpoint.get_artifact("translation_path")

    if not all([video_path, bgm_path, translation_path]):
        raise RuntimeError("Missing required artifacts from Job A.")

    # ── Step 1: TTS Generation ───────────────────────────────
    if not checkpoint.is_phase_done("tts_done"):
        checkpoint.update_phase("tts_generating")

        tts = TTSGenerator(
            provider=os.getenv("TTS_PROVIDER", "google"),
            fpt_api_key=os.getenv("FPT_API_KEY", ""),
            fpt_voice=os.getenv("FPT_VOICE", "banmai"),
            fpt_speed=os.getenv("FPT_SPEED", "0"),
            google_api_key=os.getenv("GOOGLE_TTS_API_KEY", ""),
            google_voice=os.getenv("GOOGLE_TTS_VOICE", "vi-VN-Neural2-A"),
            google_speaking_rate=float(os.getenv("GOOGLE_TTS_SPEAKING_RATE", "1.0")),
            max_concurrent=5,
            atempo_max=float(os.getenv("ATEMPO_MAX", "1.15")),
            bleed_seconds=float(os.getenv("BLEED_SECONDS", "0.5")),
        )

        tts_dir = str(work_dir / "tts_chunks")
        completed = checkpoint.get_progress("tts_completed", [])

        chunks_meta = asyncio.run(
            tts.generate_all(
                translation_path=translation_path,
                output_dir=tts_dir,
                completed_chunks=completed,
            )
        )

        # Calculate timing adjustments
        timing_data = tts.calculate_timing_adjustments(chunks_meta)

        # Save timing data
        timing_path = work_dir / "timing_data.json"
        with open(timing_path, "w", encoding="utf-8") as f:
            json.dump(timing_data, f, indent=2, ensure_ascii=False)

        checkpoint.set_artifact("tts_dir", tts_dir)
        checkpoint.set_artifact("timing_data_path", str(timing_path))
        checkpoint.update_progress(
            "tts_completed", [c["id"] for c in chunks_meta]
        )
        checkpoint.update_phase("tts_done")
    else:
        logger.info("TTS generation already complete. Skipping.")
        with open(checkpoint.get_artifact("timing_data_path"), "r") as f:
            timing_data = json.load(f)

    # ── Step 2: Prepare dub audio track ──────────────────────
    renderer = Renderer(
        nvenc_preset=os.getenv("NVENC_PRESET", "p4"),
        nvenc_cq=int(os.getenv("NVENC_CQ", "25")),
        audio_bitrate=os.getenv("OUTPUT_AUDIO_BITRATE", "192k"),
        brightness_shift=float(os.getenv("BRIGHTNESS_SHIFT", "0.01")),
        contrast_shift=float(os.getenv("CONTRAST_SHIFT", "1.01")),
        bgm_duck_threshold=float(os.getenv("BGM_DUCK_THRESHOLD", "0.05")),
        bgm_duck_ratio=int(os.getenv("BGM_DUCK_RATIO", "10")),
        bgm_duck_attack=int(os.getenv("BGM_DUCK_ATTACK", "20")),
        bgm_duck_release=int(os.getenv("BGM_DUCK_RELEASE", "300")),
        bgm_base_volume=float(os.getenv("BGM_BASE_VOLUME", "0.3")),
        avatar_enabled=os.getenv("AVATAR_ENABLED", "true").lower() == "true",
        avatar_idle_path=os.getenv("AVATAR_IDLE", "assets/avatar_idle.png"),
        avatar_speaking_path=os.getenv("AVATAR_SPEAKING", "assets/avatar_speaking.png"),
        avatar_position=os.getenv("AVATAR_POSITION", "bottom_right"),
        avatar_scale=float(os.getenv("AVATAR_SCALE", "0.15")),
    )

    # Get video duration
    total_duration_ms = Renderer.get_video_duration_ms(video_path)

    # Build the concatenated dub track
    dub_path = str(work_dir / "full_dub.wav")
    renderer.prepare_dub_track(
        timing_data=timing_data,
        output_path=dub_path,
        total_duration_ms=total_duration_ms,
    )

    # ── Step 3: Generate subtitles ───────────────────────────
    sub_gen = SubtitleGenerator(
        font_name=os.getenv("SUBTITLE_FONT", "Be Vietnam Pro"),
        font_size=int(os.getenv("SUBTITLE_FONTSIZE", "20")),
        primary_color=os.getenv("SUBTITLE_PRIMARY_COLOR", "&H00FFFFFF"),
        outline_color=os.getenv("SUBTITLE_OUTLINE_COLOR", "&H00000000"),
        outline_width=int(os.getenv("SUBTITLE_OUTLINE_WIDTH", "3")),
        box_color=os.getenv("SUBTITLE_BOX_COLOR", "&H80000000"),
    )

    # Horizontal subtitles
    h_subs_path = str(work_dir / "subtitles_h.ass")
    sub_gen.generate(
        timing_data=timing_data,
        output_path=h_subs_path,
    )

    # Vertical subtitles (adjusted for 9:16)
    render_modes = os.getenv("RENDER_MODES", "horizontal,vertical").split(",")
    v_subs_path = None
    if "vertical" in render_modes:
        v_subs_path = str(work_dir / "subtitles_v.ass")
        sub_gen.generate_vertical(
            timing_data=timing_data,
            output_path=v_subs_path,
        )

    # ── Step 4: FFmpeg Render ────────────────────────────────
    checkpoint.update_phase("rendering")
    output_files = []

    # Horizontal (16:9) for YouTube
    if "horizontal" in render_modes:
        h_output = str(work_dir / "output_vi_horizontal.mp4")
        renderer.render_horizontal(
            video_path=video_path,
            bgm_path=bgm_path,
            dub_path=dub_path,
            subtitle_path=h_subs_path,
            output_path=h_output,
        )
        output_files.append(h_output)
        checkpoint.set_artifact("output_horizontal", h_output)

    # Vertical (9:16) for TikTok / YT Shorts
    if "vertical" in render_modes and v_subs_path:
        v_output = str(work_dir / "output_vi_vertical.mp4")
        renderer.render_vertical(
            video_path=video_path,
            bgm_path=bgm_path,
            dub_path=dub_path,
            subtitle_path=v_subs_path,
            output_path=v_output,
        )
        output_files.append(v_output)
        checkpoint.set_artifact("output_vertical", v_output)

    checkpoint.update_phase("render_done")

    # ── Job B Complete ───────────────────────────────────────
    checkpoint.update_phase("complete")

    logger.info("=" * 60)
    logger.info("JOB B COMPLETE — PIPELINE FINISHED")
    for f in output_files:
        size_mb = os.path.getsize(f) / (1024 * 1024)
        logger.info(f"  Output: {f} ({size_mb:.1f} MB)")
    logger.info("GPU can be shut down. Files ready for retrieval.")
    logger.info("=" * 60)

    # Output paths for n8n
    print("\n===OUTPUT_FILES_START===")
    print(json.dumps(output_files))
    print("===OUTPUT_FILES_END===")

    return output_files


# ═══════════════════════════════════════════════════════════════
# CLI Entry Point
# ═══════════════════════════════════════════════════════════════

def main():
    parser = argparse.ArgumentParser(
        description="Video Localization Pipeline — English to Vietnamese"
    )
    parser.add_argument(
        "--mode",
        choices=["job_a", "job_b"],
        required=True,
        help="Pipeline mode: job_a (extract+translate) or job_b (render)",
    )
    parser.add_argument(
        "--url",
        type=str,
        default="",
        help="YouTube URL (required for job_a)",
    )
    parser.add_argument(
        "--work-dir",
        type=str,
        default="/workspace",
        help="Working directory for artifacts",
    )
    parser.add_argument(
        "--config",
        type=str,
        default=None,
        help="Path to config.env file",
    )

    args = parser.parse_args()

    # Load config
    load_config(args.config)

    work_dir = Path(args.work_dir)
    work_dir.mkdir(parents=True, exist_ok=True)

    try:
        if args.mode == "job_a":
            if not args.url:
                parser.error("--url is required for job_a mode")
            run_job_a(args.url, work_dir)

        elif args.mode == "job_b":
            run_job_b(work_dir)

    except Exception as e:
        logger.exception(f"Pipeline failed: {e}")
        # Save error to checkpoint for n8n to read
        try:
            from checkpoint import CheckpointManager
            cp = CheckpointManager(str(work_dir))
            cp.load()
            cp.log_error(str(e))
        except Exception:
            pass
        sys.exit(1)


if __name__ == "__main__":
    main()
