from fastapi import FastAPI, UploadFile
import librosa
import numpy as np
from sine import generate_sine_wave  # From adolfintel/sine (April 17)

app = FastAPI(title="SoundWeave Swarm")

@app.post("/swarm-weave")
async def weave_audio(file: UploadFile, mood: str = "Calm"):
    # Load audio (1-3 mins)
    audio, sr = librosa.load(file.file, duration=180)
    # Get tempo
    tempo, _ = librosa.beat.beat_track(y=audio, sr=sr)
    # Swarm effect
    if mood == "Calm":
        base_freq = 432  # Grounding tone
        base_sine = generate_sine_wave(freq=base_freq, duration=len(audio)/sr, sample_rate=sr)
        # Flocking: Two subtle frequency shifts
        swarm1 = generate_sine_wave(freq=base_freq + np.random.uniform(-5, 5), duration=len(audio)/sr, sample_rate=sr)
        swarm2 = generate_sine_wave(freq=base_freq + np.random.uniform(-5, 5), duration=len(audio)/sr, sample_rate=sr)
        # Mix: Low amplitude to preserve original
        audio = audio + 0.1 * (base_sine + 0.05 * (swarm1 + swarm2))
    # Safety: Clip and cap
    audio = np.clip(audio, -1, 1)
    # Save low-quality output
    output_path = "output.wav"
    librosa.output.write_wav(output_path, audio, sr, bitrate=128)
    return {"file": output_path, "mood": mood}
