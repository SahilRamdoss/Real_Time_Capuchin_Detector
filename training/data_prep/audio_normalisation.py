import numpy as np
import yaml
import scipy
import time
import random
from pathlib import Path

## REMARK: THIS FILE IS USED FOR TRAINING ONLY. HENCE, WE CAN USE TENSORFLOW.IO FOR AUDIO PROCESSING

class AudioNorm:
    def __init__(self, window_size: float = None, sr: int = None):
        self.window_size = window_size
        self.sampling_rate = sr

    @classmethod
    def from_config(cls, config_path: str = None) -> "AudioNorm":
        """
        Use settings in config.yaml to normalize audio. 
        
        Uses the default hardcoded path but can be changed by the optional parameter config_path

        Params:
            config_path (str): The file path for the config.yaml file (OPTIONAL)
        """

        if config_path == None:
            root_dir = Path(__file__).resolve().parent.parent.parent
            config_path = str(root_dir) + "\\configs\\config.yaml"

        with open(config_path, "r") as f:
            config = yaml.safe_load(f)

        audio_cfg = config["audio_config"]

        return cls(**audio_cfg)

    def normalize_sampling_rate(self, audio: np.ndarray, orig_sr: int = None, target_sr: int = None) -> np.ndarray:
        """
        Used to convert a sound sample to a target sampling rate

        Args:
            audio (np.ndarray): The audio for which the sampling rate must be changed
            orig_sr (int): The original sampling rate of the audio
            target_sr (int): The target sampling rate. It uses the default value from config.yaml if no argument is passed.
            
        Returns:
            The sound sample at the target sampling rate
        """

        if orig_sr is None:
            raise ValueError("Original sampling rate cannot be None.")
        elif orig_sr <= 0:
            raise ValueError("Original sampling rate cannot be zero or negative.")
        
        if target_sr == None:
            target_sr = self.sampling_rate
        elif target_sr <= 0:
            raise ValueError("Target sampling rate cannot be zero or negative.")

        if orig_sr == target_sr:
            return audio
        
        # Find the greatest common divisor to get the lowest possible integer ratios
        gcd = np.gcd(orig_sr, target_sr)
        
        # Up-sampling factor
        up = target_sr // gcd
        # Down-sampling factor
        down = orig_sr // gcd
        
        # Apply polyphase resampling
        y_resampled = scipy.signal.resample_poly(audio, up, down)
        
        return y_resampled

    def silence_padding(self, audio_samples: np.ndarray) -> np.ndarray:
        """
        Pads a 1-D audio np.ndarray up to the target window size, by adding samples at the end. 
        If the audio length >= window size ignore.

        Args:
            audio_samples (np.ndarray): Audio samples data. Accepted shape is (Number of samples, )

        Returns:
            padded audio samples of shape (Number of samples,)
        """

        # Calculate the length of the audio
        number_of_samples = audio_samples.shape[0]
        target_sample_count = self.sampling_rate * self.window_size

        # If audio length >= window size, ignore it
        if number_of_samples >= target_sample_count:
            return audio_samples
        
        pad_amount = int(target_sample_count - number_of_samples)

        choice = random.randint(1,2)

        if choice == 1:
            padded = np.pad(audio_samples, (0, pad_amount), mode="constant", constant_values=0.0)
        else:
            padded = np.pad(audio_samples, (pad_amount,0), mode='constant', constant_values=0.0)

        return padded.astype(np.float32)

    def noise_padding(self):
        """
        This will be done in the data augmentation part as Background Noise Injection (SNR between-class mixing)
        """
        raise NotImplementedError
    
    def segmentation(self):
        pass

if __name__ == "__main__":
    audionorm = AudioNorm()

    ##########################################################################################
    # TEST SERIES 1 - Testing normalize_sampling_rate method
    ##########################################################################################


    # # 1. Create 1 second of dummy audio (a basic 440Hz sine wave) at 44100Hz
    # orig_sr = 44100
    # t = np.linspace(0, 1, orig_sr, endpoint=False)
    # dummy_audio = np.sin(2 * np.pi * 440 * t)

    # print(f"Original audio shape: {dummy_audio.shape} (Should be {orig_sr} samples)\n")

    # # ==========================================
    # # WARM-UP RUN (Crucial for accurate benchmarks)
    # # ==========================================
    # # We run it once and throw the result away so SciPy caches everything.
    # _ = audionorm.normalize_sampling_rate(dummy_audio, orig_sr, 22050)
    # # ==========================================

    # # 2. Test Downsampling: 44100 Hz -> 22050 Hz
    # target_sr_down = 22050
    # start = time.perf_counter()
    # downsampled_audio = audionorm.normalize_sampling_rate(dummy_audio, orig_sr, target_sr_down)
    # end = time.perf_counter()
    # print(f"Downsampled audio shape: {downsampled_audio.shape} (Expected: {target_sr_down}). Time taken: {end-start:.6f}s")

    # # 3. Test Upsampling: 44100 Hz -> 48000 Hz
    # target_sr_up = 48000
    # start2 = time.perf_counter()
    # upsampled_audio = audionorm.normalize_sampling_rate(dummy_audio, orig_sr, target_sr_up)
    # end2 = time.perf_counter()
    # print(f"Upsampled audio shape: {upsampled_audio.shape} (Expected: {target_sr_up}). Time taken: {end2-start2:.6f}s")

    # # 4. Test Same Sampling Rate: 44100 Hz -> 44100 Hz
    # start3 = time.perf_counter()
    # same_audio = audionorm.normalize_sampling_rate(dummy_audio, orig_sr, orig_sr)
    # end3 = time.perf_counter()
    # print(f"Same SR audio shape: {same_audio.shape} (Expected: {orig_sr}). Time taken: {end3-start3:.6f}s")
    # print(f"Returned exact same array? {same_audio is dummy_audio}")

    ##########################################################################################
    # TEST SERIES 2 -Testing silence_padding method
    ##########################################################################################

    # audionorm_1 = AudioNorm(3.0, 44100)

    # start = time.perf_counter()

    # random_sound = np.random.random(30000)

    # print(f"Time taken for random sound generation: {time.perf_counter() - start}")

    # # Original Sound
    # print(random_sound)
    # print(random_sound.shape)

    # start = time.perf_counter()

    # padded_sound = audionorm_1.silence_padding(random_sound)

    # print(f"Time taken for padding: {time.perf_counter() - start}")

    # print(padded_sound)
    # print(padded_sound.shape)

    # padded_twice_sound = audionorm_1.silence_padding(padded_sound)

    # print(padded_twice_sound)
    # print(padded_twice_sound.shape)

    audionorm = AudioNorm.from_config()