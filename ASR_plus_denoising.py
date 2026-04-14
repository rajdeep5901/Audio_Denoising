

import sounddevice as sd
import numpy as np
import logging
from faster_whisper import WhisperModel

# --- Configuration ---
logging.basicConfig(level=logging.INFO, format='%(levelname)s: %(message)s')
TARGET_LANGUAGE = "en"
SAMPLE_RATE = 16000
BLOCK_DURATION_MS = 700  
BLOCK_SAMPLES = int(SAMPLE_RATE * BLOCK_DURATION_MS / 1000)
CONFIDENCE_THRESHOLD = 0.5

# --- Model Initialization ---
# This is a global variable for the model.
model = None

def audio_callback(indata, outdata, frames, time, status):
    """
    This function is called for each block of audio.
    It now performs detection and routing directly.
    """
    global model

    if status:
        logging.warning(f"Audio stream status: {status}")

    try:
        audio_mono = indata.flatten().astype(np.float32) #1D array of float 32 samples is required by the model.

        # --- Language Detection ---
        # We run the model directly here.
        
        print("-> Analyzing audio chunk...")
        _, info = model.transcribe(audio_mono, language=None, beam_size=1)
        
        detected_lang = info.language
        confidence = info.language_probability
        
        print(f"--> Detected: {detected_lang} (Confidence: {confidence:.2f})")

        # --- Decision Logic ---
        if detected_lang == TARGET_LANGUAGE and confidence >= CONFIDENCE_THRESHOLD:
            print("---> Decision: PLAY (Language matches)")
            outdata[:] = indata  # Pass the audio through
        else:
            print("---> Decision: MUTE (Language does not match or confidence is low)")
            outdata.fill(0) # Mute the audio by sending silence

    except Exception as e:
        logging.error(f"Error during audio processing: {e}")
        outdata.fill(0) # Mute on error to be safe

def main():
    """
    Loads the model, then sets up and runs the audio stream.
    """
    global model
    
    print("="*50)
    print("STEP 2: SIMPLIFIED LANGUAGE FILTER")
    print("="*50)
    
    try:
        print("Loading faster-whisper model (this may take a moment)...")
        # Using the 'tiny' model for faster performance.
        model = WhisperModel("tiny", device="cpu", compute_type="int8")
        print("Model loaded successfully.")
    except Exception as e:
        logging.error(f"Failed to load whisper model: {e}")
        return

    print("\nStarting audio stream...")
    print(f" Target Language: {TARGET_LANGUAGE.upper()}")
    print(" Speak into your microphone.")
    print(" Only the target language will be played back.")
    print("\n Press Ctrl+C to stop.")
    print("="*50 + "\n")

    try:
        with sd.Stream(
            samplerate=SAMPLE_RATE,
            blocksize=BLOCK_SAMPLES,
            channels=1,
            dtype='float32',
            callback=audio_callback
        ):
            while True:
                sd.sleep(1000)

    except KeyboardInterrupt:
        print("\nFilter stopped.")
    except Exception as e:
        logging.error(f"An error occurred with the audio stream: {e}")

if __name__ == "__main__":
    main()
