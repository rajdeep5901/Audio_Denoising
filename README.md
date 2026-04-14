# Real-Time Language-Based Selective Audio Denoising

A real-time audio processing system that selectively filters audio streams based on spoken language. The pipeline captures microphone input, identifies the language using a quantized Whisper model, and dynamically mutes or passes through the audio output — effectively "denoising" any speech that isn't in your target language.

## What It Does

The system listens to your microphone feed in real time and:

1. **Detects the spoken language** using OpenAI's Whisper model (via faster-whisper with int8 quantization)
2. **Passes through audio** when the detected language matches your target (default: English)
3. **Mutes audio** when a different language or noise is detected
4. **Applies smooth volume fading** to avoid abrupt cuts or clicking artifacts

This is useful in multilingual environments where you want to isolate speech in a specific language from a mixed audio stream.

## How It Works

The architecture uses a **producer-consumer pattern** with three parallel components:

- **Audio Callback** — Non-blocking I/O handler that captures microphone frames and routes them to processing queues. Runs on the audio thread so it must be fast.
- **Detection Worker** — Background thread that accumulates ~1 second of audio, runs Whisper inference for language identification, and updates the shared language state using a voting mechanism over a sliding window.
- **Routing Worker** — Background thread that reads input frames, checks the current detected language, and applies exponential volume fading before writing to the output queue.

The detection and routing are fully decoupled from the audio I/O, which keeps the stream glitch-free even on slower hardware.

## Setup

### Prerequisites

- Python 3.8+
- A working microphone
- PortAudio (for sounddevice)

### Installation

```bash
git clone https://github.com/aakashdvd/Real-time-langauge-based-selectiive-audio-de-noising.git
cd Real-time-langauge-based-selectiive-audio-de-noising
pip install -r requirements.txt
```

On Linux, you may also need PortAudio headers:

```bash
sudo apt-get install libportaudio2
```

### Dependencies

- `sounddevice` — real-time audio I/O
- `numpy` — audio buffer manipulation
- `faster-whisper` — optimized Whisper inference with CTranslate2 backend

## Running

```bash
python ASR_plus_denoising_test.py
```

Wait for the model to load (you'll see `Model loaded.` in the console), then start speaking.

### Configuration

Edit the constants at the top of `ASR_plus_denoising_test.py`:

```python
TARGET_LANGUAGE = "en"          # Language to pass through ("hi", "es", "fr", etc.)
CONFIDENCE_THRESHOLD = 0.3      # Minimum detection confidence
DETECTION_INTERVAL = 0.5        # Seconds between inference runs
BUFFER_DURATION_S = 1.0         # Audio accumulation window for detection
BLOCK_DURATION_MS = 30          # Audio frame size in milliseconds
```

## Model Details

- **Model**: Whisper `tiny` via faster-whisper
- **Quantization**: int8 (CTranslate2)
- **Device**: CPU
- **Inference window**: 1.0 second of accumulated audio
- **Language voting**: Sliding window majority vote over ~2 seconds of detection history

The tiny model was chosen for its low latency on CPU. It supports 99 languages but works best with English as the target. For better accuracy with non-English targets, consider using the `small` or `base` model at the cost of higher latency.

## Results

| Metric | Value |
|---|---|
| Detection latency | ~500ms (0.5s inference interval) |
| Audio frame size | 30ms |
| End-to-end latency | ~1-2 seconds |
| CPU usage | ~15-25% on modern hardware |
| Supported languages | 99 (Whisper supported) |

The system achieves reliable language filtering with smooth audio transitions. English detection accuracy is highest; other languages may require tuning the confidence threshold.

## Tech Stack

- **Python** — core runtime
- **faster-whisper** — Whisper model inference (CTranslate2 backend)
- **NumPy** — audio array processing
- **sounddevice** — PortAudio bindings for real-time audio I/O
- **threading** — concurrent detection and routing workers
- **queue** — thread-safe inter-component communication

## Files

- `ASR_plus_denoising.py` — Initial simplified version (single-threaded, higher latency)
- `ASR_plus_denoising_test.py` — Optimized multi-threaded version with voting and smooth fading
- `requirements.txt` — Python dependencies

## Troubleshooting

- **"Input Overflow" warnings** — CPU is too slow for the callback. Increase `BLOCK_DURATION_MS` (e.g., to 50ms).
- **Choppy audio** — Close other heavy applications. The detection worker may be starving the routing worker.
- **Wrong language detected** — Increase `CONFIDENCE_THRESHOLD` or ensure microphone volume is adequate.
- **No audio output** — Check that your default audio device is set correctly in your OS settings.

## License

MIT
