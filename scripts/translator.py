"""
Translator (Gemini 1.5 Pro)
============================
Full-context translation with structured JSON output.
"""

import json
import logging
from pathlib import Path

import google.generativeai as genai

logger = logging.getLogger(__name__)

TRANSLATION_PROMPT = """You are a professional English-to-Vietnamese translator specializing in video content localization.

You will receive a JSON transcript with English text segments. Translate each segment into natural, fluent Vietnamese.

CRITICAL RULES:
1. Preserve the exact JSON structure — every segment must keep its "id", "start", and "end" values unchanged.
2. Translate naturally — do NOT translate word-for-word. Use natural Vietnamese phrasing and sentence structure.
3. Keep proper nouns, brand names, and widely-known English terms as-is (e.g., "YouTube", "iPhone", "Google").
4. For technical terms, use the commonly accepted Vietnamese equivalent if one exists (e.g., "machine learning" → "học máy", "artificial intelligence" → "trí tuệ nhân tạo").
5. Match the tone and register of the original (casual stays casual, formal stays formal).
6. Keep translations concise — Vietnamese translations should be similar in spoken length to the English original. Avoid unnecessary padding words.
7. Return ONLY valid JSON — no markdown, no code fences, no explanations.

INPUT FORMAT:
{
    "segments": [
        {"id": 0, "text": "English text here", "start": 0.5, "end": 3.2},
        ...
    ]
}

OUTPUT FORMAT:
{
    "segments": [
        {"id": 0, "original": "English text here", "translated": "Vietnamese translation here", "start": 0.5, "end": 3.2},
        ...
    ]
}
"""


class Translator:
    """Translates transcripts using Gemini 1.5 Pro."""

    def __init__(self, api_key: str, model_name: str = "gemini-1.5-pro"):
        genai.configure(api_key=api_key)
        self.model = genai.GenerativeModel(
            model_name,
            generation_config=genai.GenerationConfig(
                temperature=0.3,  # Low temp for consistent translations
                response_mime_type="application/json",
            ),
        )
        logger.info(f"Gemini translator initialized with {model_name}.")

    def translate(
        self,
        transcript_path: str,
        output_path: str,
        target_language: str = "Vietnamese",
    ) -> dict:
        """
        Translate a transcript JSON file.

        Args:
            transcript_path: Path to the WhisperX transcript JSON
            output_path: Path to save the translated JSON
            target_language: Target language (for prompt context)

        Returns:
            Translation dict with original + translated text per segment
        """
        output_path = Path(output_path)

        # Skip if already translated (checkpoint recovery)
        if output_path.exists():
            logger.info(f"Translation already exists at {output_path}. Loading.")
            with open(output_path, "r", encoding="utf-8") as f:
                return json.load(f)

        # Load transcript
        with open(transcript_path, "r", encoding="utf-8") as f:
            transcript = json.load(f)

        # Prepare the translation input (only id, text, start, end)
        translation_input = {
            "segments": [
                {
                    "id": seg["id"],
                    "text": seg["text"],
                    "start": seg["start"],
                    "end": seg["end"],
                }
                for seg in transcript["segments"]
            ]
        }

        logger.info(
            f"Sending {len(translation_input['segments'])} segments to Gemini..."
        )

        # Send to Gemini with the full context
        prompt = f"{TRANSLATION_PROMPT}\n\nTranslate the following transcript to {target_language}:\n\n{json.dumps(translation_input, indent=2, ensure_ascii=False)}"

        response = self.model.generate_content(prompt)

        # Parse response
        try:
            translation = json.loads(response.text)
        except json.JSONDecodeError:
            # Try to extract JSON from response
            text = response.text.strip()
            if text.startswith("```"):
                text = text.split("\n", 1)[1].rsplit("```", 1)[0].strip()
            translation = json.loads(text)

        # Validate structure
        if "segments" not in translation:
            raise ValueError("Gemini response missing 'segments' key")

        if len(translation["segments"]) != len(transcript["segments"]):
            logger.warning(
                f"Segment count mismatch: expected {len(transcript['segments'])}, "
                f"got {len(translation['segments'])}. Attempting alignment..."
            )
            # Fallback: align by ID
            translated_map = {s["id"]: s for s in translation["segments"]}
            aligned = []
            for orig in transcript["segments"]:
                if orig["id"] in translated_map:
                    aligned.append(translated_map[orig["id"]])
                else:
                    # Fill missing with original text as fallback
                    aligned.append({
                        "id": orig["id"],
                        "original": orig["text"],
                        "translated": orig["text"],  # Untranslated fallback
                        "start": orig["start"],
                        "end": orig["end"],
                    })
                    logger.warning(f"Segment {orig['id']} missing from translation.")
            translation["segments"] = aligned

        # Save
        output_path.parent.mkdir(parents=True, exist_ok=True)
        with open(output_path, "w", encoding="utf-8") as f:
            json.dump(translation, f, indent=2, ensure_ascii=False)

        logger.info(f"Translation saved to {output_path}")
        return translation
