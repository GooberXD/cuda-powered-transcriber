from faster_whisper import WhisperModel
try:
    model = WhisperModel("tiny", device="cuda")
    print("GPU is working!")
except Exception as e:
    print(f"Error: {e}")