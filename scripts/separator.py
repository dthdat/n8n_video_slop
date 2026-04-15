"""
Audio Separator (Demucs)
========================
Isolates vocals from background music/SFX using htdemucs_ft.
"""

import os
import logging
from pathlib import Path

import torch
import torchaudio
from demucs.pretrained import get_model
from demucs.apply import apply_model

logger = logging.getLogger(__name__)


class AudioSeparator:
    """Separates audio into vocals and background using Demucs."""

    # htdemucs stem order: drums=0, bass=1, other=2, vocals=3
    STEM_MAP = {"drums": 0, "bass": 1, "other": 2, "vocals": 3}

    def __init__(self, model_name: str = "htdemucs_ft"):
        self.model_name = model_name
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        logger.info(f"Loading Demucs model '{model_name}' on {self.device}...")
        self.model = get_model(model_name)
        self.model.to(self.device)
        logger.info("Demucs model loaded successfully.")

    def separate(
        self,
        audio_path: str,
        output_dir: str,
        vocals_filename: str = "vocals.wav",
        bgm_filename: str = "bgm.wav",
    ) -> tuple[str, str]:
        """
        Separate audio into vocals and background (drums+bass+other).

        Returns:
            Tuple of (vocals_path, bgm_path)
        """
        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)

        vocals_path = output_dir / vocals_filename
        bgm_path = output_dir / bgm_filename

        # Skip if already separated (checkpoint recovery)
        if vocals_path.exists() and bgm_path.exists():
            logger.info("Separated audio files already exist. Skipping.")
            return str(vocals_path), str(bgm_path)

        logger.info(f"Loading audio from {audio_path}...")
        wav, sr = torchaudio.load(audio_path)

        # Resample to model's sample rate if needed
        if sr != self.model.samplerate:
            logger.info(f"Resampling from {sr}Hz to {self.model.samplerate}Hz...")
            resampler = torchaudio.transforms.Resample(sr, self.model.samplerate)
            wav = resampler(wav)
            sr = self.model.samplerate

        # Ensure stereo
        if wav.shape[0] == 1:
            wav = wav.repeat(2, 1)

        wav = wav.to(self.device)

        logger.info("Running Demucs separation (this may take a few minutes)...")
        with torch.no_grad():
            sources = apply_model(
                self.model,
                wav.unsqueeze(0),  # Add batch dimension
                split=True,        # Memory-efficient for long tracks
                overlap=0.25,
            )[0]  # Remove batch dimension

        # Extract vocals
        vocals = sources[self.STEM_MAP["vocals"]]

        # Combine non-vocal stems for background
        bgm = (
            sources[self.STEM_MAP["drums"]]
            + sources[self.STEM_MAP["bass"]]
            + sources[self.STEM_MAP["other"]]
        )

        # Save to disk
        logger.info(f"Saving vocals to {vocals_path}")
        torchaudio.save(str(vocals_path), vocals.cpu(), sr)

        logger.info(f"Saving background to {bgm_path}")
        torchaudio.save(str(bgm_path), bgm.cpu(), sr)

        # Free GPU memory
        del sources, vocals, bgm, wav
        torch.cuda.empty_cache()

        logger.info("Audio separation complete.")
        return str(vocals_path), str(bgm_path)
