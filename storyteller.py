import json
from dataclasses import dataclass
from typing import List, Dict, Any, Optional

import requests
from loguru import logger


@dataclass
class ClipSuggestion:
    """
    Represents a suggested clip region with semantic reasoning.
    """
    start: float
    end: float
    reason: str


class StoryTeller:
    """
    Handles transcript analysis and clip suggestion by prompting Ollama.
    """

    def __init__(self, config: Dict[str, Any]) -> None:
        """
        Initialize StoryTeller with Ollama-related configuration.

        :param config: Global configuration dictionary including Ollama settings.
        """
        self.ollama_base_url: str = config["ollama"]["base_url"].rstrip("/")
        self.model_name: str = config["ollama"]["model"]
        self.timeout_seconds: int = int(config["ollama"].get("timeout_seconds", 180))
        self.max_retries: int = int(config["ollama"].get("max_retries", 3))
        logger.info("Initialized StoryTeller with model {}", self.model_name)

    def analyze_transcript(
        self,
        transcript: str,
        segments: List[Dict[str, Any]],
        video_duration: float,
        min_len: float,
        max_len: float,
        max_clips: int,
    ) -> List[ClipSuggestion]:
        """
        Analyze transcript with Ollama to identify viral-ready clip timestamps.

        :param transcript: Full transcript text for the video.
        :param segments: List of segment dicts {"start", "end", "text"}.
        :param video_duration: Total video duration in seconds.
        :param min_len: Minimum desired clip length in seconds.
        :param max_len: Maximum desired clip length in seconds.
        :param max_clips: Maximum number of clips to return.
        :return: List of ClipSuggestion objects.
        """
        if not transcript.strip():
            logger.warning("Empty transcript received; skipping Ollama analysis.")
            return []

        prompt = self._build_prompt(transcript, segments, video_duration, min_len, max_len, max_clips)
        raw_content = self._call_ollama(prompt)
        if raw_content is None:
            logger.error("Ollama did not return content; no clips suggested.")
            return []

        suggestions = self._parse_clip_suggestions(raw_content, min_len, max_len, video_duration, max_clips)
        logger.info("Received {} clip suggestions from Ollama.", len(suggestions))
        return suggestions

    def _build_prompt(
        self,
        transcript: str,
        segments: List[Dict[str, Any]],
        video_duration: float,
        min_len: float,
        max_len: float,
        max_clips: int,
    ) -> str:
        """
        Build a precise prompt instructing Ollama to return JSON clip suggestions.

        :return: Prompt string.
        """
        segment_summaries = [
            f"[{s['start']:.2f} - {s['end']:.2f}] {s['text']}"
            for s in segments
        ]
        segments_block = "\n".join(segment_summaries[:200])

        prompt = f"""
You are an expert short-form content editor.

You will receive:
1) The full transcript of a long-form video.
2) A list of transcript segments with timestamps.

Your job is to select the BEST short clips (TikTok/Reels/Shorts style) that are:

- HIGHLY engaging.
- Contain a STRONG hook in the first 3–5 seconds.
- Emotionally charged (excitement, surprise, tension, inspiration).
- Create curiosity loops (unanswered questions, setups without immediate payoff).
- Self-contained enough to stand alone as a 30–60 second viral clip.

CONSTRAINTS:
- Video duration: {video_duration:.2f} seconds.
- Clip length must be between {min_len:.1f} and {max_len:.1f} seconds.
- Prefer clips with a powerful hook in the FIRST 30 seconds of the original video.
- Prefer emotionally intense sections and curiosity loops.
- Avoid segments that are mostly silence, filler, or heavy context with no payoff.
- Always return NO MORE THAN {max_clips} clips.

VERY IMPORTANT OUTPUT REQUIREMENTS:
- You MUST respond with ONLY valid minified JSON.
- DO NOT include any commentary, markdown, or code fences.
- The JSON structure MUST be exactly:

{{
  "clips": [
    {{"start": 12.5, "end": 48.2, "reason": "Strong emotional hook"}},
    ...
  ]
}}

- "start" and "end" are seconds from the start of the original video.
- "reason" briefly explains WHY this clip is strong (hook, emotion, curiosity, etc.).

TRANSCRIPT (FULL):
{transcript}

SEGMENTS WITH TIMESTAMPS (TRUNCATED IF LONG):
{segments_block}

Return ONLY the JSON. Do not wrap in or any other formatting.
"""
        return prompt.strip()

    def _call_ollama(self, prompt: str) -> Optional[str]:
        """
        Call Ollama's chat API and return the raw message content.

        :param prompt: Prompt text to send.
        :return: Raw content string, or None on failure.
        """
        url = f"{self.ollama_base_url}/api/chat"
        payload = {
            "model": self.model_name,
            "messages": [
                {"role": "user", "content": prompt}
            ],
            "stream": False,
        }

        last_error: Optional[Exception] = None
        for attempt in range(1, self.max_retries + 1):
            try:
                logger.info("Calling Ollama (attempt {}/{})", attempt, self.max_retries)
                resp = requests.post(url, json=payload, timeout=self.timeout_seconds)
                resp.raise_for_status()
                data = resp.json()
                content = data.get("message", {}).get("content")
                if not content:
                    logger.error("Ollama response missing 'message.content': {}", data)
                    last_error = RuntimeError("Empty Ollama content")
                    continue
                return content
            except Exception as exc:  # noqa: BLE001
                last_error = exc
                logger.exception("Error calling Ollama (attempt {}): {}", attempt, exc)

        logger.error("Failed to contact Ollama after {} attempts. Last error: {}", self.max_retries, last_error)
        return None

    def _parse_clip_suggestions(
        self,
        raw_content: str,
        min_len: float,
        max_len: float,
        video_duration: float,
        max_clips: int,
    ) -> List[ClipSuggestion]:
        """
        Parse JSON from Ollama, sanitize, and enforce length & range constraints.

        :param raw_content: Raw text returned by the model.
        :return: List of sanitized ClipSuggestion instances.
        """
        text = raw_content.strip()
        if text.startswith("```"):
            text = text.strip("`")
            if "\n" in text:
                first_line, rest = text.split("\n", 1)
                if not first_line.strip().startswith("{"):
                    text = rest

        try:
            parsed = json.loads(text)
        except json.JSONDecodeError:
            logger.exception("Failed to parse Ollama JSON. Raw content: {}", raw_content[:1000])
            return []

        clips_raw = parsed.get("clips", [])
        suggestions: List[ClipSuggestion] = []

        for c in clips_raw:
            try:
                start = float(c.get("start", 0.0))
                end = float(c.get("end", 0.0))
                reason = str(c.get("reason", "")).strip() or "No reason provided"

                if end <= start:
                    continue

                start = max(0.0, min(start, video_duration))
                end = max(0.0, min(end, video_duration))

                duration = end - start
                if duration < min_len:
                    end = min(start + min_len, video_duration)
                    duration = end - start
                if duration > max_len:
                    end = start + max_len
                    duration = end - start

                if duration < min_len * 0.5:
                    continue

                suggestions.append(ClipSuggestion(start=start, end=end, reason=reason))
            except Exception:  # noqa: BLE001
                logger.exception("Error normalizing clip suggestion: {}", c)

        return suggestions[:max_clips]