"""
Transcriber (WhisperX)
======================
Forced-alignment transcription with word-level timestamps.
"""

import json
import logging
import gc
from pathlib import Path

import torch
import whisperx

logger = logging.getLogger(__name__)


class Transcriber:
    """Transcribes audio using WhisperX with forced alignment."""

    def __init__(
        self,
        model_name: str = "large-v3",
        batch_size: int = 16,
        compute_type: str = "float16",
    ):
        self.model_name = model_name
        self.batch_size = batch_size
        self.compute_type = compute_type
        self.device = "cuda" if torch.cuda.is_available() else "cpu"

        logger.info(f"Loading WhisperX model '{model_name}' on {self.device}...")
        self.model = whisperx.load_model(
            model_name,
            self.device,
            compute_type=compute_type,
        )
        logger.info("WhisperX model loaded.")

    def transcribe(
        self,
        audio_path: str,
        output_path: str,
        language: str = "en",
    ) -> dict:
        """
        Transcribe audio and produce word-level aligned timestamps.

        Args:
            audio_path: Path to the vocals-only WAV file
            output_path: Path to save transcript JSON
            language: Source language code

        Returns:
            Transcript dict with segments and word-level timestamps
        """
        output_path = Path(output_path)

        # Skip if already transcribed (checkpoint recovery)
        if output_path.exists():
            logger.info(f"Transcript already exists at {output_path}. Loading.")
            with open(output_path, "r", encoding="utf-8") as f:
                return json.load(f)

        # Step 1: Load audio
        logger.info(f"Loading audio from {audio_path}...")
        audio = whisperx.load_audio(audio_path)

        # Step 2: Transcribe
        logger.info("Transcribing with WhisperX...")
        result = self.model.transcribe(
            audio,
            batch_size=self.batch_size,
            language=language,
        )
        logger.info(f"Transcription complete. {len(result['segments'])} segments found.")

        # Step 3: Forced alignment for word-level timestamps
        logger.info("Running forced alignment...")
        model_a, metadata = whisperx.load_align_model(
            language_code=language,
            device=self.device,
        )
        result = whisperx.align(
            result["segments"],
            model_a,
            metadata,
            audio,
            self.device,
            return_char_alignments=False,
        )

        # Clean up alignment model
        del model_a, metadata
        gc.collect()
        torch.cuda.empty_cache()

        # Step 4: Structure the output
        transcript = {
            "language": language,
            "segments": [],
        }

        for i, seg in enumerate(result["segments"]):
            segment = {
                "id": i,
                "text": seg.get("text", "").strip(),
                "start": round(seg.get("start", 0), 3),
                "end": round(seg.get("end", 0), 3),
                "words": [],
            }
            for word in seg.get("words", []):
                if "start" in word and "end" in word:
                    segment["words"].append({
                        "word": word["word"].strip(),
                        "start": round(word["start"], 3),
                        "end": round(word["end"], 3),
                    })
            transcript["segments"].append(segment)

        # Save
        output_path.parent.mkdir(parents=True, exist_ok=True)
        with open(output_path, "w", encoding="utf-8") as f:
            json.dump(transcript, f, indent=2, ensure_ascii=False)

        logger.info(f"Transcript saved to {output_path}")
        return transcript
