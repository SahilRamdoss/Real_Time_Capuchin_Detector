import sounddevice as sd
import soundfile as sf
import numpy as np

sr = 44100
duration_s = 3
output_path = "research/sound_testing/sounddevice_test_recording.wav"

print("Recording... make some noise")
recording = sd.rec(int(duration_s * sr), samplerate=sr, channels=1, dtype='float32')
sd.wait()  # blocks until the recording finishes

audio_1d = recording[:, 0]
print(f"dtype: {audio_1d.dtype}")
print(f"shape: {audio_1d.shape}")
print(f"min/max: {audio_1d.min():.4f} / {audio_1d.max():.4f}")
print(f"Audio: {audio_1d}")

# Save the recorded audio to a WAV file, at the same sample rate it was captured at.
sf.write(output_path, audio_1d, sr)
print(f"Saved recording to {output_path}")