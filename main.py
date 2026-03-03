import queue
import signal
import sys
import threading
import time
from pathlib import Path
from typing import Dict, Any, List, Tuple

import yaml
from faster_whisper import WhisperModel
from loguru import logger
from watchdog.events import FileSystemEventHandler, FileCreatedEvent, FileMovedEvent
from watchdog.observers import Observer

from storyteller import StoryTeller
from editor import VideoEditor


class VideoProcessor:
    """
    Orchestrates the full pipeline for a single video:
    transcription -> AI clip selection -> ffmpeg editing.
    """

    def __init__(self, config: Dict[str, Any]) -> None:
        """
        Initialize the processor with transcription, storytelling, and editing components.

        :param config: Global configuration dictionary.
        """
        self.config = config
        self.paths_cfg = config["paths"]
        self.clips_cfg = config["clips"]
        self.trans_cfg = config["transcription"]

        self.input_dir = Path(self.paths_cfg["input_dir"]).resolve()
        self.output_dir = Path(self.paths_cfg["output_dir"]).resolve()
        self.temp_dir = Path(self.paths_cfg["temp_dir"]).resolve()

        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.temp_dir.mkdir(parents=True, exist_ok=True)

        self.storyteller = StoryTeller(config)
        self.editor = VideoEditor(config)
        self.model = self._init_whisper_model()

        logger.info("VideoProcessor initialized. Input: {}, Output: {}", self.input_dir, self.output_dir)

    def _init_whisper_model(self) -> WhisperModel:
        """
        Initialize a single shared faster-whisper model instance.

        :return: WhisperModel ready to transcribe audio or video.
        """
        logger.info(
            "Loading faster-whisper model '{}' on device '{}' (compute_type={})",
            self.trans_cfg["model_size"],
            self.trans_cfg["device"],
            self.trans_cfg["compute_type"],
        )
        model = WhisperModel(
            self.trans_cfg["model_size"],
            device=self.trans_cfg.get("device", "cpu"),
            compute_type=self.trans_cfg.get("compute_type", "int8"),
        )
        return model

    def process_video(self, video_path: Path) -> None:
        """
        Run the full processing pipeline for a single video file.

        :param video_path: Path to the input video.
        """
        logger.info("Starting processing for video: {}", video_path)

        try:
            duration, width, height = self.editor.probe_video(video_path)
        except Exception:  # noqa: BLE001
            logger.exception("Failed to probe video '{}'; skipping.", video_path)
            return

        try:
            transcript_text, segments = self._transcribe_video(video_path)
        except Exception:  # noqa: BLE001
            logger.exception("Transcription failed for '{}'; skipping.", video_path)
            return

        try:
            clip_suggestions = self.storyteller.analyze_transcript(
                transcript=transcript_text,
                segments=segments,
                video_duration=duration,
                min_len=float(self.clips_cfg["min_length_seconds"]),
                max_len=float(self.clips_cfg["max_length_seconds"]),
                max_clips=int(self.clips_cfg["max_clips_per_video"]),
            )
        except Exception:  # noqa: BLE001
            logger.exception("Ollama analysis failed for '{}'; skipping clip creation.", video_path)
            return

        if not clip_suggestions:
            logger.warning("No clip suggestions produced for '{}'.", video_path)
            return

        base_name = video_path.stem
        logger.info("Got {} clips for '{}'", len(clip_suggestions), base_name)

        for idx, clip in enumerate(clip_suggestions, start=1):
            try:
                self._render_clip(
                    video_path=video_path,
                    clip_start=clip.start,
                    clip_end=clip.end,
                    reason=clip.reason,
                    index=idx,
                    total=len(clip_suggestions),
                    segments=segments,
                )
            except Exception:  # noqa: BLE001
                logger.exception("Failed to render clip {} for '{}'. Continuing with next.", idx, video_path)

        logger.info("Finished processing video: {}", video_path)

    def _transcribe_video(self, video_path: Path) -> Tuple[str, List[Dict[str, Any]]]:
        """
        Transcribe the video using faster-whisper.

        :param video_path: Path to the input video.
        :return: (full transcript text, list of segment dicts).
        """
        logger.info("Starting transcription for {}", video_path)
        segments_iter, info = self.model.transcribe(
            str(video_path),
            beam_size=self.trans_cfg.get("beam_size", 3),
            language=self.trans_cfg.get("language") or None,
            vad_filter=self.trans_cfg.get("vad_filter", True),
        )

        full_text_parts: List[str] = []
        segments: List[Dict[str, Any]] = []

        for seg in segments_iter:
            text = seg.text.strip()
            if not text:
                continue
            full_text_parts.append(text)
            segments.append(
                {
                    "start": float(seg.start),
                    "end": float(seg.end),
                    "text": text,
                }
            )

        full_text = " ".join(full_text_parts)
        logger.info(
            "Transcription complete for {}. Language: {}, duration: {:.2f}s, segments: {}",
            video_path,
            getattr(info, "language", "unknown"),
            getattr(info, "duration", 0.0),
            len(segments),
        )
        return full_text, segments

    def _render_clip(
        self,
        video_path: Path,
        clip_start: float,
        clip_end: float,
        reason: str,
        index: int,
        total: int,
        segments: List[Dict[str, Any]],
    ) -> None:
        """
        Render a single clip with subtitles and vertical formatting.

        :param video_path: Source video.
        :param clip_start: Clip start time in seconds.
        :param clip_end: Clip end time in seconds.
        :param reason: Human-readable reason from Ollama.
        :param index: Clip index (for naming).
        :param total: Total number of clips (for naming).
        :param segments: Global transcript segments to build subtitles from.
        """
        logger.info(
            "Rendering clip {}/{} [{:.2f}, {:.2f}] for '{}'. Reason: {}",
            index,
            total,
            clip_start,
            clip_end,
            video_path.name,
            reason,
        )

        base_name = video_path.stem
        safe_reason = "".join(c for c in reason if c.isalnum() or c in (" ", "-", "_")).strip().replace(" ", "_")
        if not safe_reason:
            safe_reason = "clip"

        output_filename = f"{base_name}_clip{index:02d}_{safe_reason}.mp4"
        output_path = self.output_dir / output_filename

        srt_path = self.temp_dir / f"{base_name}_clip{index:02d}.srt"

        self.editor.generate_srt(
            segments=segments,
            srt_path=srt_path,
            clip_start=clip_start,
            clip_end=clip_end,
        )

        apply_zoom = bool(self.config["ffmpeg"].get("enable_zoom_punch_in", True))
        self.editor.create_clip(
            input_video=video_path,
            output_video=output_path,
            subtitles_path=srt_path,
            clip_start=clip_start,
            clip_end=clip_end,
            apply_zoom=apply_zoom,
        )

        logger.info("Rendered clip saved to {}", output_path)


class DebouncedEventHandler(FileSystemEventHandler):
    """
    Watchdog handler that enqueues video files with debouncing.
    """

    def __init__(self, config: Dict[str, Any], task_queue: "queue.Queue[Path]") -> None:
        """
        Initialize handler with configuration and shared task queue.

        :param config: Global configuration dict.
        :param task_queue: Queue of files to be processed.
        """
        super().__init__()
        self.config = config
        self.task_queue = task_queue
        self.extensions = set(ext.lower() for ext in config["watcher"]["file_extensions"])
        self.debounce_seconds = float(config["watcher"].get("debounce_seconds", 5))
        self._pending: Dict[Path, float] = {}

    def on_created(self, event: FileCreatedEvent) -> None:  # type: ignore[override]
        """
        Handle file creation events by scheduling processing if extension matches.
        """
        if event.is_directory:
            return
        self._maybe_schedule(Path(event.src_path))

    def on_moved(self, event: FileMovedEvent) -> None:  # type: ignore[override]
        """
        Handle file move events (e.g., atomic renames after download).
        """
        if event.is_directory:
            return
        self._maybe_schedule(Path(event.dest_path))

    def _maybe_schedule(self, path: Path) -> None:
        """
        Conditionally schedule a file for processing after debounce delay.

        :param path: Path to candidate input file.
        """
        if path.suffix.lower() not in self.extensions:
            return
        logger.info("Detected new video file: {}", path)
        self._pending[path] = time.time() + self.debounce_seconds

    def drain_pending_to_queue(self) -> None:
        """
        Move debounced files into the processing queue when ready.
        """
        now = time.time()
        ready = [p for p, t in list(self._pending.items()) if t <= now]
        for p in ready:
            logger.info("Enqueuing video for processing: {}", p)
            self.task_queue.put(p)
            self._pending.pop(p, None)


def load_config(path: Path) -> Dict[str, Any]:
    """
    Load YAML configuration from file.

    :param path: Path to config.yaml.
    :return: Configuration dict.
    """
    with path.open("r", encoding="utf-8") as f:
        cfg = yaml.safe_load(f)
    return cfg


def setup_logging(config: Dict[str, Any]) -> None:
    """
    Configure loguru logging to console and rotating file.

    :param config: Global configuration dict containing paths.log_file.
    """
    logger.remove()
    logger.add(sys.stderr, level="INFO", enqueue=True, colorize=True)

    log_file = Path(config["paths"]["log_file"])
    log_file.parent.mkdir(parents=True, exist_ok=True)
    logger.add(
        str(log_file),
        rotation="10 MB",
        retention=10,
        level="INFO",
        enqueue=True,
        backtrace=True,
        diagnose=False,
    )
    logger.info("Logging configured. Log file: {}", log_file)


def worker_loop(processor: VideoProcessor, task_queue: "queue.Queue[Path]", stop_event: threading.Event) -> None:
    """
    Background worker loop that processes videos from the queue.

    :param processor: Shared VideoProcessor instance.
    :param task_queue: Queue containing video paths.
    :param stop_event: Event signaling termination.
    """
    logger.info("Worker thread started.")
    while not stop_event.is_set():
        try:
            video_path = task_queue.get(timeout=1.0)
        except queue.Empty:
            continue

        try:
            processor.process_video(video_path)
        except Exception:  # noqa: BLE001
            logger.exception("Unhandled error while processing video: {}", video_path)
        finally:
            task_queue.task_done()

    logger.info("Worker thread stopping.")


def main() -> None:
    """
    Entry point for the viral video pipeline.
    """
    config_path = Path("config.yaml").resolve()
    if not config_path.exists():
        print(f"config.yaml not found at {config_path}.")
        sys.exit(1)

    config = load_config(config_path)
    setup_logging(config)

    processor = VideoProcessor(config)
    task_queue: "queue.Queue[Path]" = queue.Queue()
    stop_event = threading.Event()

    handler = DebouncedEventHandler(config, task_queue)
    observer = Observer()
    observer.schedule(handler, str(processor.input_dir), recursive=False)
    observer.start()
    logger.info("Watching input directory for new videos: {}", processor.input_dir)

    worker = threading.Thread(target=worker_loop, args=(processor, task_queue, stop_event), daemon=True)
    worker.start()

    def handle_sigint(signum, frame) -> None:  # noqa: ANN001, D401
        """Handle process termination (Ctrl+C)."""
        logger.info("Received termination signal. Shutting down...")
        stop_event.set()
        observer.stop()

    signal.signal(signal.SIGINT, handle_sigint)

    try:
        while not stop_event.is_set():
            handler.drain_pending_to_queue()
            time.sleep(1.0)
    finally:
        observer.stop()
        observer.join()
        worker.join()
        logger.info("Pipeline shut down cleanly.")


if __name__ == "__main__":
    main()