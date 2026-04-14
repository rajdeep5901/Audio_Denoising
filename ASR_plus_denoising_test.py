import sounddevice as sd
import numpy as np
import logging
import threading
import queue
import time
from collections import deque, Counter
from faster_whisper import WhisperModel

# --- Configuration ---
logging.basicConfig(level=logging.INFO, format='%(levelname)s: %(message)s')
TARGET_LANGUAGE = "en"
SAMPLE_RATE = 16000
BLOCK_DURATION_MS = 30               # small frame for I/O
BLOCK_SAMPLES = int(SAMPLE_RATE * BLOCK_DURATION_MS / 1000)
DETECTION_INTERVAL = 0.5            # seconds between detection
BUFFER_DURATION_S = 1.0             # accumulate 1s of audio
BUFFER_FRAMES = int(BUFFER_DURATION_S * 1000 / BLOCK_DURATION_MS)
CONFIDENCE_THRESHOLD = 0.3
VOTE_THRESHOLD = 0.7
HISTORY_WINDOW = int(2.0 * 1000 / BLOCK_DURATION_MS)  # 2s window

# --- Queues and State ---
audio_input_q = queue.Queue(maxsize=100)
audio_output_q = queue.Queue(maxsize=100)
detection_q   = queue.Queue(maxsize=50)

state_lock = threading.Lock()
current_language = "unknown"
language_conf = 0.0
history = deque(maxlen=HISTORY_WINDOW)
running = True

# --- Model Initialization ---
model = None

def load_model():
    global model
    logging.info("Loading faster-whisper model (tiny-int8)...")
    model = WhisperModel("tiny", device="cpu", compute_type="int8")
    logging.info("Model loaded.")

# --- Detection Worker ---
def detection_worker():
    global current_language, language_conf, running
    buffer = deque(maxlen=BUFFER_FRAMES)
    last_detect = 0
    while running:
        try:
            frame = detection_q.get(timeout=0.1)
            buffer.append(frame)
        except queue.Empty:
            continue

        now = time.time()
        if len(buffer) >= BUFFER_FRAMES and (now - last_detect) >= DETECTION_INTERVAL:
            audio = np.concatenate(list(buffer))
            try:
                # Whisper expects float32 normalized
                audio_f32 = audio.astype(np.float32)
                _, info = model.transcribe(
                    audio_f32, language=None, beam_size=1,
                    condition_on_previous_text=False,
                    vad_filter=True,
                    vad_parameters=dict(min_silence_duration_ms=100)
                )
                lang = info.language
                conf = info.language_probability
            except Exception as e:
                logging.warning(f"Detection error: {e}")
                lang, conf = "unknown", 0.0

            with state_lock:
                language_conf = conf
                history.append(lang if conf >= CONFIDENCE_THRESHOLD else "unk")
                # vote
                votes = Counter(history)
                lang_vote, _ = votes.most_common(1)[0]
                current_language = lang_vote

            buffer.clear()
            last_detect = now
            logging.info(f"Detected language: {current_language} (conf={language_conf:.2f})")

# --- Routing Worker ---
def routing_worker():
    global running
    current_vol = 0.0
    fade_rate = 0.1
    while running:
        try:
            frame = audio_input_q.get(timeout=0.05)
        except queue.Empty:
            continue
        # Determine forwarding
        with state_lock:
            ok = (current_language == TARGET_LANGUAGE)
        target_vol = 1.0 if ok else 0.0
        # smooth fade
        current_vol += (target_vol - current_vol) * fade_rate
        out_frame = frame * current_vol
        try:
            audio_output_q.put_nowait(out_frame)
        except queue.Full:
            pass

# --- Audio Callback ---
def audio_callback(indata, outdata, frames, time_info, status):
    if status:
        logging.warning(f"Audio status: {status}")
    mono = indata.flatten()
    # enqueue for detection and routing
    try:
        audio_input_q.put_nowait(mono)
    except queue.Full:
        pass
    try:
        detection_q.put_nowait(mono)
    except queue.Full:
        pass
    # output
    try:
        frame = audio_output_q.get_nowait()
        outdata[:] = frame.reshape(outdata.shape)
    except queue.Empty:
        outdata.fill(0)

# --- Main ---
def main():
    global running
    load_model()
    # start workers
    det_t = threading.Thread(target=detection_worker, daemon=True)
    rout_t = threading.Thread(target=routing_worker, daemon=True)
    det_t.start()
    rout_t.start()

    logging.info("Starting audio stream. Speak now...")
    with sd.Stream(
        samplerate=SAMPLE_RATE,
        blocksize=BLOCK_SAMPLES,
        channels=1,
        dtype='float32',
        callback=audio_callback,
        latency='low'
    ):
        try:
            while True:
                time.sleep(1)
        except KeyboardInterrupt:
            logging.info("Stopping...")
            running = False

if __name__ == "__main__":
    main()
