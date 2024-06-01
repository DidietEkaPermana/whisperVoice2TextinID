`pyaudio` is not available for the `osx-arm64` platform, we can use `sounddevice` and `scipy` as alternatives for recording audio. Here’s how you can adjust the setup and script to use these libraries instead.

## Requirements

1. **Python Libraries**:
   - `faster_whisper`
   - `sounddevice`
   - `scipy`
   - `numpy`

2. **Installation**:
   - Ensure you have MiniConda installed.

## Steps

### 1. Set Up MiniConda Environment
Open your terminal and create a new Conda environment.

```sh
conda create -n whisper-voice2text python=3.9
conda activate whisper-voice2text
```

### 2. Create `requirements.txt`
Create a file named `requirements.txt` with the following content:

```txt
faster_whisper
sounddevice
scipy
numpy
```

### 3. Install Libraries

```sh
pip install -r requirements.txt
```

### 4. Create Python Script
Create a Python script, `voice_to_text.py`, with the following code:

```python
import sounddevice as sd
from scipy.io.wavfile import write
from faster_whisper import WhisperModel
import os


NUM_CORE = os.cpu_count()
# WHISPER_SIZE = 'base'
WHISPER_SIZE = 'large-v3'
whisper_model = WhisperModel(
    WHISPER_SIZE,
    device='cpu',
    compute_type='int8',
    cpu_threads=NUM_CORE // 2,
    num_workers=NUM_CORE // 2
)
# Audio recording parameters
RATE = 16000
RECORD_SECONDS = 5
WAVE_OUTPUT_FILENAME = "recorded.wav"

def record_audio():
    print("Recording...")
    myrecording = sd.rec(int(RECORD_SECONDS * RATE), samplerate=RATE, channels=1, dtype='int16')
    sd.wait()  # Wait until recording is finished
    write(WAVE_OUTPUT_FILENAME, RATE, myrecording)  # Save as WAV file 
    print("Finished recording.")

def transcribe_audio():
    print("Transcribing audio...")
    segments, _ = whisper_model.transcribe(WAVE_OUTPUT_FILENAME)
    text = ''.join(segment.text for segment in segments)
    print("Transcription:")
    # print(result['text'])
    print(text)

if __name__ == "__main__":
    # record_audio()
    transcribe_audio()
```

### 5. Run the Script
Execute the script to record and transcribe your voice.

```sh
python voice_to_text.py
```

## Detailed Explanation

- **Recording Audio**:
  - `sounddevice` is used to capture audio from your microphone.
  - The script records audio and saves it as a WAV file using `scipy`.

- **Transcribing Audio**:
  - The `faster_whisper` library is used to load the model and transcribe the recorded audio.
  - The transcription result is printed in the console.

Feel free to adjust the `RECORD_SECONDS` variable to change the recording duration. This setup use Whisper large v3 model to supports the Indonesian language.
