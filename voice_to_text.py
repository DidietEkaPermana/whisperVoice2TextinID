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