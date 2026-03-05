# 🎬 AI-Powered Video Editing Pipeline

Automatically transform long-form videos into viral-ready short clips — fully local, no subscriptions, no cloud APIs.

Drop a video into a watched folder and the pipeline handles everything: transcription → AI clip selection → vertical formatting → burned-in subtitles → rendered output.

```
input/my-video.mp4
    │
    ▼
faster-whisper (transcription)
    │
    ▼
Ollama (clip selection — hook detection, curiosity loops, emotional peaks)
    │
    ▼
ffmpeg (crop to 9:16 · zoom punch-in · silence removal · subtitles)
    │
    ▼
output/my-video_clip01_Strong_hook.mp4
output/my-video_clip02_Emotional_peak.mp4
```

---

## Features

- **Watch-folder automation** — drop any video in `./input/` and it processes automatically
- **Whisper transcription** — CPU-friendly `faster-whisper` with VAD silence filtering
- **Ollama clip selection** — LLM analyzes the full transcript and picks the highest-engagement moments based on hooks, emotional intensity, and curiosity loops
- **9:16 vertical output** — auto-crops and scales for TikTok, Reels, and Shorts
- **Zoom punch-in** — configurable scale boost for that native-feeling short-form look
- **Burned-in subtitles** — SRT generated per clip with timestamps aligned to the cut
- **Silence removal** — ffmpeg audio filter strips dead air automatically
- **Debounced file watcher** — waits for writes to complete before processing (handles large uploads and atomic renames)
- **Rotating logs** — loguru with 10MB rotation, kept for 10 files

---

## Requirements

- Python 3.10+
- [ffmpeg](https://ffmpeg.org/download.html) + ffprobe in `PATH`
- [Ollama](https://ollama.com) running locally

---

## Installation

```bash
git clone https://github.com/yournameVirecoreNLM/ai-powered-video-editing.git
cd ai-powered-video-editing

python3 -m venv .venv
source .venv/bin/activate        # Windows: .venv\Scripts\activate

pip install -r requirements.txt
```

Pull a model for Ollama (any instruction-following model works — bigger = better clip selection):

```bash
ollama pull llama3
# or for better quality:
ollama pull qwen2.5:14b
```

---

## Usage

1. Start Ollama:
   ```bash
   ollama serve
   ```

2. Run the pipeline:
   ```bash
   python main.py
   ```

3. Drop a video into the input folder:
   ```bash
   cp my-podcast.mp4 ./input/
   ```

Clips will appear in `./output/` named by source file, clip index, and the AI's reason for selecting it:

```
output/
├── my-podcast_clip01_Strong_emotional_hook.mp4
├── my-podcast_clip02_Curiosity_loop_setup.mp4
└── my-podcast_clip03_Surprising_revelation.mp4
```

---

## Configuration

All settings live in `config.yaml`. No code changes needed.

```yaml
paths:
  input_dir: "./input"       # Drop videos here
  output_dir: "./output"     # Finished clips land here
  temp_dir: "./temp"         # SRT files and intermediate work
  log_file: "./logs/pipeline.log"

watcher:
  file_extensions: [".mp4", ".mov", ".mkv", ".avi"]
  debounce_seconds: 5        # Wait this long after file write before processing

transcription:
  model_size: "small"        # tiny / base / small / medium / large-v3
  device: "cpu"              # cpu or cuda
  compute_type: "int8"       # int8 (CPU) or float16 (GPU)
  language: null             # null = auto-detect, or set e.g. "en"
  beam_size: 3
  vad_filter: true           # Strip silence before transcribing

ollama:
  base_url: "http://localhost:11434"
  model: "llama3"            # Any Ollama model
  timeout_seconds: 180
  max_retries: 3

clips:
  min_length_seconds: 30.0
  max_length_seconds: 60.0
  max_clips_per_video: 5

ffmpeg:
  target_width: 1080
  target_height: 1920        # 9:16 vertical
  video_bitrate: "6M"
  audio_bitrate: "160k"
  video_codec: "h264"
  audio_codec: "aac"
  preset: "veryfast"
  crf: 20
  enable_zoom_punch_in: true
  zoom_factor: 1.1           # 1.0 = no zoom, 1.1 = 10% punch-in
  silence_filter: "silenceremove=stop_periods=-1:stop_duration=0.3:stop_threshold=-35dB"
```

### Transcription model sizes

| Model      | Speed   | Accuracy | RAM     | Best for               |
|------------|---------|----------|---------|------------------------|
| `tiny`     | Fast    | Low      | ~1 GB   | Quick tests            |
| `base`     | Fast    | Decent   | ~1 GB   | Short videos           |
| `small`    | Medium  | Good     | ~2 GB   | Default — good balance |
| `medium`   | Slow    | Great    | ~5 GB   | Long podcasts          |
| `large-v3` | Slowest | Best     | ~10 GB  | Max accuracy           |

### GPU acceleration

Change these two lines in `config.yaml`:

```yaml
transcription:
  device: "cuda"
  compute_type: "float16"
```

---

## Project Structure

```
.
├── main.py           # Entry point — orchestrator, file watcher, worker loop
├── storyteller.py    # Ollama integration — prompt building, JSON parsing, clip validation
├── editor.py         # ffmpeg wrapper — probe, filter chains, SRT generation, clip rendering
├── config.yaml       # All configuration — no code changes needed for tuning
└── requirements.txt
```

### How it works

**`main.py`** sets up a `watchdog` observer on the input directory. New video files are debounced and queued to a background worker thread. The worker calls `VideoProcessor.process_video()` which runs the full pipeline: probe → transcribe → analyze → render.

**`storyteller.py`** builds a structured prompt from the full transcript and timestamped segments, sends it to Ollama's `/api/chat` endpoint, and parses the returned JSON clip suggestions. Clips are sanitized: duration clamped to `[min_len, max_len]`, timestamps bounded to video duration, and results capped at `max_clips`.

**`editor.py`** is a pure ffmpeg/ffprobe subprocess wrapper. It builds filter chains for vertical cropping, zoom punch-in, and subtitle burning. It generates per-clip SRT files from global transcript segments by slicing the relevant time range and rebasing timestamps to zero.

---

## Troubleshooting

**No clips produced**
- Check `./logs/pipeline.log` for Ollama errors
- Make sure `ollama serve` is running and the configured model is pulled
- Try a larger Whisper model — bad transcripts lead to bad clip selection
- Increase `timeout_seconds` in `config.yaml` if Ollama times out on long videos

**Subtitles not appearing**
- Confirm `libass` is compiled into your ffmpeg build: `ffmpeg -filters | grep subtitles`
- On macOS with Homebrew: `brew install ffmpeg` includes libass by default

**CPU transcription is slow**
- Switch to `tiny` or `base` model for faster iteration
- Set `device: cuda` and `compute_type: float16` if you have a GPU

**ffmpeg crop errors on portrait source videos**
- The pipeline is designed for landscape source footage (YouTube, podcasts). Portrait source already in 9:16 will still process but the crop math may not center correctly — adjust `target_width` and `target_height` in `config.yaml` to match your source

**Watchdog not picking up files**
- Check that the file extension is listed under `file_extensions` in `config.yaml`
- Increase `debounce_seconds` for very large files over a slow network mount

---

## License

MIT — see [LICENSE](LICENSE).
