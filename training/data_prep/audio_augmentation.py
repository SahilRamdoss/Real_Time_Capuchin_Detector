from audiomentations import Compose, Gain, AddBackgroundNoise, ClippingDistortion
import numpy as np
from pathlib import Path

class AudioAug:
    def __init__(self):
        noise_folder_path = Path.cwd().resolve() / "data" / "raw" / "Parsed_Not_Capuchinbird_Clips"

        self.augment = Compose([
            Gain(min_gain_db=-12.0, max_gain_db=12.0, p=1),
            AddBackgroundNoise(sounds_path=r"C:\Users\udhay\OneDrive\Documents\University\Student_Teams\BIOM\Real_Time_Capuchin_Detector\data\raw\Parsed_Not_Capuchinbird_Clips", noise_rms="absolute", p=1),
            # ClippingDistortion(min_percentile_threshold=0, max_percentile_threshold=40, p=1)
        ])

    def augment_audio(self, audio_samples: np.ndarray, sampling_rate: int) -> np.ndarray:
        return self.augment(audio_samples, sampling_rate)
