"""
Checkpoint & Crash Recovery System
===================================
Manages status.json for atomic state persistence and crash-safe resume.
"""

import json
import os
import shutil
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Optional


class CheckpointManager:
    """Manages pipeline state checkpoints for crash recovery."""

    PHASES = [
        "initialized",
        "downloading",
        "download_done",
        "separating",
        "separation_done",
        "transcribing",
        "transcription_done",
        "translating",
        "translation_done",
        # ── Job A ends here, Job B begins ──
        "tts_generating",
        "tts_done",
        "rendering",
        "render_done",
        "complete",
    ]

    def __init__(self, work_dir: str):
        self.work_dir = Path(work_dir)
        self.work_dir.mkdir(parents=True, exist_ok=True)
        self.status_file = self.work_dir / "status.json"
        self._state: dict = {}

    def initialize(self, video_url: str, video_id: str) -> dict:
        """Create a fresh checkpoint for a new job."""
        self._state = {
            "video_id": video_id,
            "url": video_url,
            "phase": "initialized",
            "phase_progress": {},
            "artifacts": {},
            "timestamps": {
                "started": datetime.now(timezone.utc).isoformat(),
            },
            "errors": [],
        }
        self._save()
        return self._state

    def load(self) -> Optional[dict]:
        """Load existing checkpoint. Returns None if not found."""
        if not self.status_file.exists():
            return None
        try:
            with open(self.status_file, "r", encoding="utf-8") as f:
                self._state = json.load(f)
            return self._state
        except (json.JSONDecodeError, IOError):
            return None

    def update_phase(self, phase: str, **extra_data):
        """Transition to a new phase. Atomically saves."""
        if phase not in self.PHASES:
            raise ValueError(f"Unknown phase: {phase}. Valid: {self.PHASES}")
        self._state["phase"] = phase
        self._state["timestamps"][phase] = datetime.now(timezone.utc).isoformat()
        if extra_data:
            self._state.update(extra_data)
        self._save()

    def update_progress(self, key: str, value: Any):
        """Update phase_progress sub-dict (e.g., TTS chunk progress)."""
        self._state.setdefault("phase_progress", {})[key] = value
        self._save()

    def set_artifact(self, name: str, path: str):
        """Register an artifact file path."""
        self._state.setdefault("artifacts", {})[name] = str(path)
        self._save()

    def get_artifact(self, name: str) -> Optional[str]:
        """Get an artifact path by name."""
        return self._state.get("artifacts", {}).get(name)

    def get_phase(self) -> str:
        """Get current phase."""
        return self._state.get("phase", "initialized")

    def get_progress(self, key: str, default=None):
        """Get a progress value."""
        return self._state.get("phase_progress", {}).get(key, default)

    def log_error(self, error_msg: str):
        """Append an error to the error log."""
        self._state.setdefault("errors", []).append({
            "time": datetime.now(timezone.utc).isoformat(),
            "error": str(error_msg),
        })
        self._save()

    def is_phase_done(self, phase: str) -> bool:
        """Check if we've already passed a given phase."""
        current_idx = self.PHASES.index(self.get_phase())
        target_idx = self.PHASES.index(phase)
        return current_idx > target_idx

    def get_resume_phase(self) -> str:
        """Determine the phase to resume from after a crash."""
        phase = self.get_phase()
        # If we crashed mid-phase (e.g., "separating"), resume that phase
        # If we completed a phase (e.g., "separation_done"), move to next
        if phase.endswith("_done") or phase in ("initialized", "complete"):
            return phase
        # Crashed mid-work — we need to redo this phase
        return phase

    @property
    def state(self) -> dict:
        return self._state.copy()

    def _save(self):
        """Atomic write: write to temp file, then rename."""
        tmp_file = self.status_file.with_suffix(".tmp")
        with open(tmp_file, "w", encoding="utf-8") as f:
            json.dump(self._state, f, indent=2, ensure_ascii=False)
        shutil.move(str(tmp_file), str(self.status_file))
