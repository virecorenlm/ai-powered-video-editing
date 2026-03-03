import subprocess
from pathlib import Path
from typing import Dict, Any, List

from loguru import logger


class VideoEditor:
    """
    Handles ffmpeg-based operations: probing, cropping, cutting, subtitles,
    zoom punch-ins, and silence removal.
    """

    def __init__(self, config: Dict[str, Any]) -> None:
        """
        Initialize VideoEditor with ffmpeg-related configuration.

        :param config: Global configuration dictionary including ffmpeg settings.
        """
        self.cfg = config["ffmpeg"]
        self.target_width = int(self.cfg["target_width"])
        self.target_height = int(self.cfg["target_height"])
        self.ffmpeg_exe = self.cfg.get("executable", "ffmpeg")
        self.ffprobe_exe = self.cfg.get("ffprobe_executable", "ffprobe")
        logger.info("Initialized VideoEditor with target resolution {}x{}", self.target_width, self.target_height)

    def probe_video(self, video_path: Path):
        """
        Probe video metadata using ffprobe to obtain duration and resolution.

        :param video_path: Path to the input video file.
        :return: (duration_seconds, width, height)
        """
        cmd = [
            self.ffprobe_exe,
            "-v", "error",
            "-select_streams", "v:0",
            "-show_entries", "stream=width,height",
            "-show_entries", "format=duration",
            "-of", "default=noprint_wrappers=1:nokey=1",
            str(video_path),
        ]
        logger.info("Probing video metadata with ffprobe.")
        result = subprocess.run(cmd, capture_output=True, text=True, check=True)
        lines = [x.strip() for x in result.stdout.splitlines() if x.strip()]

        if len(lines) < 3:
            raise RuntimeError(f"Unexpected ffprobe output: {result.stdout}")

        width = int(lines[0])
        height = int(lines[1])
        duration = float(lines[2])

        logger.info("Probed video - duration: {:.2f}s, resolution: {}x{}", duration, width, height)
        return duration, width, height

    def build_vertical_filter_chain(
        self,
        apply_zoom: bool,
        subtitles_path: Path,
    ) -> str:
        """
        Build ffmpeg filter chain for vertical 9:16 framing, optional zoom, and subtitles.

        :param apply_zoom: Whether to apply static punch-in zoom.
        :param subtitles_path: Path to .srt subtitles file.
        :return: ffmpeg -vf filter string.
        """
        filters: List[str] = []

        filters.append(f"scale={self.target_width}:-2")
        filters.append(f"crop={self.target_width}:{self.target_height}")

        if apply_zoom and self.cfg.get("enable_zoom_punch_in", True):
            zoom_factor = float(self.cfg.get("zoom_factor", 1.1))
            filters.append(f"scale=iw*{zoom_factor}:ih*{zoom_factor}")
            filters.append(f"crop={self.target_width}:{self.target_height}")

        subs_escaped = str(subtitles_path).replace("\\", "\\\\")
        filters.append(f"subtitles='{subs_escaped}'")

        vf = ",".join(filters)
        logger.debug("Constructed video filter chain: {}", vf)
        return vf

    def build_audio_filter_chain(self) -> str:
        """
        Build ffmpeg audio filter chain, including silence removal.

        :return: ffmpeg -af filter string.
        """
        silence_filter = self.cfg.get("silence_filter", "")
        filters: List[str] = []
        if silence_filter:
            filters.append(silence_filter)
        af = ",".join(filters) if filters else "anull"
        logger.debug("Constructed audio filter chain: {}", af)
        return af

    def generate_srt(
        self,
        segments,
        srt_path: Path,
        clip_start: float,
        clip_end: float,
    ) -> None:
        """
        Generate an SRT file for a specific clip range using global segments.

        :param segments: Global transcript segments with absolute timestamps.
        :param srt_path: Output .srt file path.
        :param clip_start: Clip start time in seconds (absolute).
        :param clip_end: Clip end time in seconds (absolute).
        """
        logger.info("Generating SRT for clip [{:.2f}, {:.2f}] at {}", clip_start, clip_end, srt_path)
        relevant = [s for s in segments if s["end"] >= clip_start and s["start"] <= clip_end]

        lines: List[str] = []
        index = 1
        for seg in relevant:
            start = max(seg["start"], clip_start) - clip_start
            end = min(seg["end"], clip_end) - clip_start
            if end <= start:
                continue
            lines.append(str(index))
            lines.append(f"{self._format_srt_time(start)} --> {self._format_srt_time(end)}")
            lines.append(seg["text"].strip())
            lines.append("")
            index += 1

        srt_path.parent.mkdir(parents=True, exist_ok=True)
        srt_path.write_text("\n".join(lines), encoding="utf-8")
        logger.info("Wrote {} subtitle entries.", index - 1)

    @staticmethod
    def _format_srt_time(seconds: float) -> str:
        """
        Format seconds as SRT timecode.

        :param seconds: Time in seconds.
        :return: String in HH:MM:SS,mmm format.
        """
        total_ms = int(round(seconds * 1000))
        ms = total_ms % 1000
        total_s = total_ms // 1000
        s = total_s % 60
        total_m = total_s // 60
        m = total_m % 60
        h = total_m // 60
        return f"{h:02d}:{m:02d}:{s:02d},{ms:03d}"

    def create_clip(
        self,
        input_video: Path,
        output_video: Path,
        subtitles_path: Path,
        clip_start: float,
        clip_end: float,
        apply_zoom: bool,
    ) -> None:
        """
        Create a single 9:16 viral-ready clip using ffmpeg.

        :param input_video: Source video path.
        :param output_video: Destination video path.
        :param subtitles_path: Path to SRT subtitles (relative to clip start).
        :param clip_start: Clip start time in seconds (absolute).
        :param clip_end: Clip end time in seconds (absolute).
        :param apply_zoom: Whether to apply zoom punch-in filters.
        """
        duration = max(0.1, clip_end - clip_start)
        vf = self.build_vertical_filter_chain(apply_zoom=apply_zoom, subtitles_path=subtitles_path)
        af = self.build_audio_filter_chain()

        cmd = [
            self.ffmpeg_exe,
            "-y",
            "-ss", f"{clip_start:.3f}",
            "-t", f"{duration:.3f}",
            "-i", str(input_video),
            "-vf", vf,
            "-af", af,
            "-c:v", self.cfg.get("video_codec", "h264"),
            "-c:a", self.cfg.get("audio_codec", "aac"),
            "-b:v", self.cfg.get("video_bitrate", "6M"),
            "-b:a", self.cfg.get("audio_bitrate", "160k"),
            "-preset", self.cfg.get("preset", "veryfast"),
            "-crf", str(self.cfg.get("crf", 20)),
            "-movflags", "+faststart",
            str(output_video),
        ]

        logger.info("Running ffmpeg to create clip: {}", " ".join(cmd))
        try:
            subprocess.run(cmd, check=True)
        except subprocess.CalledProcessError as exc:
            logger.exception("ffmpeg failed while creating clip {}: {}", output_video, exc)
            raise