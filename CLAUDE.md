# CLAUDE.md — AI Assistant Guide

This document provides codebase context for AI assistants working on this project.

## Project Overview

AI-powered video editing pipeline that watches a folder for new video files, transcribes them with faster-whisper, asks an Ollama LLM to select the best clip moments, then renders vertical (9:16) short-form clips with burned-in subtitles via ffmpeg.

## Architecture

```
File Watcher (watchdog)
    ↓
Debounce Queue (DebouncedEventHandler)
    ↓
Task Queue (queue.Queue)
    ↓
Worker Thread (worker_loop)
    ↓
VideoProcessor.process_video()
    ├─→ VideoEditor.probe_video()           # ffprobe
    ├─→ _transcribe_video()                 # faster-whisper
    ├─→ StoryTeller.analyze_transcript()    # Ollama LLM
    └─→ For each ClipSuggestion:
            ├─→ VideoEditor.generate_srt()  # SRT with rebased timestamps
            └─→ VideoEditor.create_clip()   # ffmpeg subprocess
```

## Module Responsibilities

| File | Class | Role |
|------|-------|------|
| `main.py` | `VideoProcessor` | Pipeline orchestrator; owns Whisper + calls StoryTeller + VideoEditor |
| `main.py` | `DebouncedEventHandler` | Watchdog handler with per-file debounce before queueing |
| `storyteller.py` | `StoryTeller` | Builds Ollama prompts, calls API, parses/validates `ClipSuggestion` list |
| `storyteller.py` | `ClipSuggestion` | Dataclass: `start: float`, `end: float`, `reason: str` |
| `editor.py` | `VideoEditor` | ffmpeg/ffprobe subprocess wrapper; constructs filter chains |

External dependencies not in `requirements.txt`: **ffmpeg**, **ffprobe**, **Ollama** (must be running locally).

## Directory Layout

```
/
├── main.py             # Entry point, watcher, worker thread
├── storyteller.py      # Ollama LLM integration
├── editor.py           # ffmpeg wrapper
├── config.yaml         # All runtime configuration (no .env)
├── requirements.txt    # pip dependencies
├── setup.txt           # Manual setup guide
├── README.md           # User-facing documentation
├── input/              # Drop videos here (gitignored)
├── output/             # Rendered clips land here (gitignored)
├── temp/               # SRT and intermediate files (gitignored)
└── logs/               # pipeline.log with rotation (gitignored)
```

## Running the Project

```bash
# Create and activate virtualenv
python3 -m venv .venv
source .venv/bin/activate

# Install dependencies
pip install -r requirements.txt

# Ensure ffmpeg and Ollama are installed and Ollama is running
ollama serve   # separate terminal

# Run the pipeline watcher
python main.py
```

Drop a `.mp4 / .mov / .mkv / .avi` file into `./input/` — the watcher picks it up after a configurable debounce.

## Configuration

All settings live in `config.yaml`. No environment variables are used. Key sections:

| Section | Key Settings |
|---------|-------------|
| `paths` | `input_dir`, `output_dir`, `temp_dir`, `log_file` |
| `watcher` | `file_extensions`, `debounce_seconds` |
| `ffmpeg` | codecs, bitrates, dimensions (1080×1920), `enable_zoom_punch_in`, `zoom_factor` |
| `transcription` | `model_size` (tiny→large-v3), `device` (cpu/cuda), `compute_type` |
| `ollama` | `base_url`, `model`, `timeout_seconds`, `max_retries` |
| `clips` | `min_length_seconds`, `max_length_seconds`, `max_clips_per_video` |
| `subtitles` | `font_size`, `font_color`, `outline_color`, `outline_width` |

When adding a new configurable value, always add it to `config.yaml` — never hardcode values in source files.

## Code Conventions

### Python Style
- **Python 3.10+** with extensive type hints throughout
- `snake_case` for functions/methods/variables, `PascalCase` for classes
- Loguru for all logging — use `logger.info/warning/error/exception`, never `print()`
- Access config values via `config["section"]["key"]` dict notation

### Error Handling
- Clip-level failures are caught and logged but do not abort the rest of the pipeline
- Ollama calls retry up to `config["ollama"]["max_retries"]` times with a fixed delay
- ffprobe/ffmpeg subprocess failures raise exceptions that bubble to `process_video()`

### Output Naming
Clips are named: `{base_name}_clip{index:02d}_{sanitized_reason}.mp4`

The reason string is sanitized: lowercased, spaces→underscores, only `[a-z0-9_]` kept, truncated to 30 chars.

### Filter Chain Construction
`VideoEditor.build_vertical_filter_chain()` returns a comma-joined ffmpeg filter string. Modify this method to change crop/scale/zoom/subtitle behavior — do not construct filter strings inline elsewhere.

### Prompt Engineering (StoryTeller)
`StoryTeller._build_prompt()` constructs the Ollama system + user prompt. The model is instructed to return strict JSON matching the `ClipSuggestion` schema. When modifying the prompt, ensure the JSON schema instruction is preserved and `_parse_clip_suggestions()` is updated to match any schema changes.

## Testing

There are no automated tests. Manual testing workflow:

1. Drop a short test video into `./input/`
2. Observe console output for each pipeline stage
3. Check `./logs/pipeline.log` for detailed logs
4. Verify clips appear in `./output/` with correct naming and format

When adding new functionality, consider adding test videos that exercise edge cases (very short clips, no speech, multiple speakers).

## Dependencies

```
watchdog>=4.0.0         # File system event monitoring
faster-whisper>=1.0.0   # CPU-friendly speech-to-text
PyYAML>=6.0.0           # Config parsing
requests>=2.31.0        # Ollama HTTP API calls
loguru>=0.7.2           # Structured logging with rotation
```

System requirements: `ffmpeg`, `ffprobe`, `Ollama` with a loaded model.

## Key Design Decisions

- **No database** — state lives only in the filesystem (input/output dirs and logs)
- **Single worker thread** — videos process serially to avoid resource contention on CPU-bound Whisper
- **Debounced watcher** — waits `debounce_seconds` after the last file event before queuing, so large file copies don't trigger premature processing
- **Config-only LLM selection** — switching models requires only changing `ollama.model` in config.yaml
- **Pure subprocess for ffmpeg** — no Python ffmpeg bindings; keeps the filter chain transparent and debuggable

## Git Branch

Active development branch: `claude/add-claude-documentation-gewcs`
