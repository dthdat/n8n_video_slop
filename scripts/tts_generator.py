"""
TTS Generator (FPT.AI + Google Cloud TTS)
==========================================
Async batch TTS generation with dual provider support and checkpoint recovery.
"""

import asyncio
import json
import logging
import os
import struct
import wave
from pathlib import Path
from typing import Optional

import aiohttp
import aiofiles
from pydub import AudioSegment

logger = logging.getLogger(__name__)


class TTSGenerator:
    """Generates Vietnamese TTS audio chunks from translated segments."""

    def __init__(
        self,
        provider: str = "google",
        fpt_api_key: str = "",
        fpt_voice: str = "banmai",
        fpt_speed: str = "0",
        google_api_key: str = "",
        google_voice: str = "vi-VN-Neural2-A",
        google_speaking_rate: float = 1.0,
        max_concurrent: int = 5,
        atempo_max: float = 1.15,
        bleed_seconds: float = 0.5,
    ):
        self.provider = provider.lower()
        self.fpt_api_key = fpt_api_key
        self.fpt_voice = fpt_voice
        self.fpt_speed = fpt_speed
        self.google_api_key = google_api_key
        self.google_voice = google_voice
        self.google_speaking_rate = google_speaking_rate
        self.max_concurrent = max_concurrent
        self.atempo_max = atempo_max
        self.bleed_seconds = bleed_seconds

        logger.info(f"TTS Generator initialized with provider: {self.provider}")

    async def generate_all(
        self,
        translation_path: str,
        output_dir: str,
        completed_chunks: Optional[list] = None,
    ) -> list[dict]:
        """
        Generate TTS audio for all segments asynchronously.

        Args:
            translation_path: Path to the translated JSON
            output_dir: Directory to save chunk WAV files
            completed_chunks: List of already-completed chunk IDs (for resume)

        Returns:
            List of chunk metadata dicts with timing info
        """
        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)
        completed_chunks = set(completed_chunks or [])

        with open(translation_path, "r", encoding="utf-8") as f:
            translation = json.load(f)

        segments = translation["segments"]
        semaphore = asyncio.Semaphore(self.max_concurrent)
        chunks_meta = []

        async with aiohttp.ClientSession() as session:
            tasks = []
            for seg in segments:
                chunk_id = seg["id"]
                if chunk_id in completed_chunks:
                    # Load existing chunk metadata
                    chunk_path = output_dir / f"chunk_{chunk_id:04d}.wav"
                    if chunk_path.exists():
                        audio = AudioSegment.from_wav(str(chunk_path))
                        chunks_meta.append({
                            "id": chunk_id,
                            "path": str(chunk_path),
                            "original_start": seg["start"],
                            "original_end": seg["end"],
                            "original_duration": seg["end"] - seg["start"],
                            "tts_duration": len(audio) / 1000.0,
                            "text": seg["translated"],
                        })
                    continue

                tasks.append(
                    self._generate_chunk(
                        session, semaphore, seg, output_dir
                    )
                )

            results = await asyncio.gather(*tasks, return_exceptions=True)

            for result in results:
                if isinstance(result, Exception):
                    logger.error(f"TTS generation error: {result}")
                    continue
                chunks_meta.append(result)

        # Sort by ID
        chunks_meta.sort(key=lambda x: x["id"])

        # Save metadata
        meta_path = output_dir / "chunks_meta.json"
        with open(meta_path, "w", encoding="utf-8") as f:
            json.dump(chunks_meta, f, indent=2, ensure_ascii=False)

        logger.info(
            f"TTS generation complete. {len(chunks_meta)} chunks generated."
        )
        return chunks_meta

    async def _generate_chunk(
        self,
        session: aiohttp.ClientSession,
        semaphore: asyncio.Semaphore,
        segment: dict,
        output_dir: Path,
    ) -> dict:
        """Generate a single TTS chunk."""
        chunk_id = segment["id"]
        text = segment["translated"]
        chunk_path = output_dir / f"chunk_{chunk_id:04d}.wav"

        async with semaphore:
            logger.info(f"Generating TTS for chunk {chunk_id}: {text[:50]}...")

            if self.provider == "fpt":
                await self._generate_fpt(session, text, chunk_path)
            elif self.provider == "google":
                await self._generate_google(session, text, chunk_path)
            else:
                raise ValueError(f"Unknown TTS provider: {self.provider}")

            # Measure duration
            audio = AudioSegment.from_wav(str(chunk_path))
            tts_duration = len(audio) / 1000.0
            original_duration = segment["end"] - segment["start"]

            logger.info(
                f"Chunk {chunk_id}: TTS={tts_duration:.2f}s, "
                f"Original={original_duration:.2f}s"
            )

            return {
                "id": chunk_id,
                "path": str(chunk_path),
                "original_start": segment["start"],
                "original_end": segment["end"],
                "original_duration": original_duration,
                "tts_duration": tts_duration,
                "text": text,
            }

    async def _generate_fpt(
        self,
        session: aiohttp.ClientSession,
        text: str,
        output_path: Path,
    ):
        """Generate TTS via FPT.AI API."""
        url = "https://api.fpt.ai/hmi/tts/v5"
        headers = {
            "api_key": self.fpt_api_key,
            "voice": self.fpt_voice,
            "speed": self.fpt_speed,
            "format": "wav",
            "Cache-Control": "no-cache",
        }

        async with session.post(url, headers=headers, data=text.encode("utf-8")) as resp:
            if resp.status != 200:
                raise RuntimeError(f"FPT.AI error: {resp.status} - {await resp.text()}")

            result = await resp.json()
            if not result.get("async"):
                raise RuntimeError(f"FPT.AI error: {result}")

            audio_url = result["async"]

        # Poll for the audio file (FPT.AI processes asynchronously)
        for attempt in range(60):  # Max 2 minutes
            await asyncio.sleep(2)
            async with session.get(audio_url) as audio_resp:
                if audio_resp.status == 200:
                    content = await audio_resp.read()
                    if len(content) > 1000:  # Valid audio file
                        async with aiofiles.open(output_path, "wb") as f:
                            await f.write(content)
                        # Convert mp3 to wav if needed
                        self._ensure_wav(output_path)
                        return

        raise TimeoutError(f"FPT.AI audio not ready after 2 minutes for: {text[:50]}")

    async def _generate_google(
        self,
        session: aiohttp.ClientSession,
        text: str,
        output_path: Path,
    ):
        """Generate TTS via Google Cloud Text-to-Speech REST API."""
        url = f"https://texttospeech.googleapis.com/v1/text:synthesize?key={self.google_api_key}"

        payload = {
            "input": {"text": text},
            "voice": {
                "languageCode": "vi-VN",
                "name": self.google_voice,
            },
            "audioConfig": {
                "audioEncoding": "LINEAR16",
                "sampleRateHertz": 24000,
                "speakingRate": self.google_speaking_rate,
            },
        }

        async with session.post(url, json=payload) as resp:
            if resp.status != 200:
                error_text = await resp.text()
                raise RuntimeError(f"Google TTS error: {resp.status} - {error_text}")

            result = await resp.json()
            audio_content = result["audioContent"]

        # Decode base64 audio
        import base64
        audio_bytes = base64.b64decode(audio_content)

        async with aiofiles.open(output_path, "wb") as f:
            await f.write(audio_bytes)

    def _ensure_wav(self, path: Path):
        """Convert to WAV if the file is actually MP3."""
        try:
            with wave.open(str(path), "rb"):
                return  # Already valid WAV
        except (wave.Error, struct.error):
            # Likely MP3 from FPT.AI, convert
            audio = AudioSegment.from_file(str(path))
            audio.export(str(path), format="wav")

    def calculate_timing_adjustments(self, chunks_meta: list) -> list:
        """
        Apply the Time-Boundary Logic Gate to adjust chunk timings.

        Rules:
        - If TTS is shorter than original window → pad with silence
        - If TTS is longer → allow 0.5s bleed, then atempo up to 1.15x
        - If still too long after 1.15x → allow extended bleed (log warning)
        """
        adjusted = []

        for i, chunk in enumerate(chunks_meta):
            original_dur = chunk["original_duration"]
            tts_dur = chunk["tts_duration"]
            adjustment = {
                **chunk,
                "action": "none",
                "atempo": 1.0,
                "silence_pad": 0.0,
                "adjusted_start": chunk["original_start"],
                "adjusted_end": chunk["original_end"],
            }

            if tts_dur <= original_dur:
                # TTS is shorter or equal — pad with silence
                silence_needed = original_dur - tts_dur
                adjustment["action"] = "pad_silence"
                adjustment["silence_pad"] = round(silence_needed, 3)
                logger.debug(
                    f"Chunk {chunk['id']}: Padding {silence_needed:.3f}s silence"
                )

            else:
                # TTS is longer — calculate speed-up
                allowed_dur = original_dur + self.bleed_seconds
                ratio = tts_dur / allowed_dur

                if ratio <= 1.0:
                    # Fits within bleed allowance
                    adjustment["action"] = "bleed"
                    adjustment["adjusted_end"] = round(
                        chunk["original_start"] + tts_dur, 3
                    )
                    logger.debug(
                        f"Chunk {chunk['id']}: Natural bleed ({tts_dur - original_dur:.3f}s)"
                    )

                elif ratio <= self.atempo_max:
                    # Speed up within cap
                    adjustment["action"] = "atempo"
                    adjustment["atempo"] = round(ratio, 3)
                    logger.debug(
                        f"Chunk {chunk['id']}: Atempo {ratio:.3f}x"
                    )

                else:
                    # Exceeds cap — apply max atempo + extended bleed
                    adjustment["action"] = "atempo_bleed"
                    adjustment["atempo"] = self.atempo_max
                    remaining_dur = tts_dur / self.atempo_max
                    adjustment["adjusted_end"] = round(
                        chunk["original_start"] + remaining_dur, 3
                    )
                    logger.warning(
                        f"Chunk {chunk['id']}: Exceeds cap! "
                        f"Atempo {self.atempo_max}x + extended bleed "
                        f"({remaining_dur - original_dur:.3f}s over)"
                    )

            adjusted.append(adjustment)

        return adjusted
