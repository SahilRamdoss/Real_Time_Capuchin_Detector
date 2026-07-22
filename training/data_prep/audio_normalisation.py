import numpy as np
import yaml
import scipy
import random
from pathlib import Path
from utils.config_finder import find_config_path
from typing import Optional, Tuple

## REMARK: THIS FILE IS USED FOR TRAINING ONLY. HENCE, WE CAN USE TENSORFLOW.IO FOR AUDIO PROCESSING IF NEEDED

class AudioNorm:
    def __init__(self, window_size: float, sr: int, hop_size: float, clip_pad_max_ratio: float):
        """
        Constructor for the AudioNorm class. Initializes the window size, sampling rate, and hop size for audio normalization.

        Args:
            window_size (float): The duration of each audio segment in seconds.
            sr (int): The target sampling rate in Hz.
            hop_size (float): The hop size for overlapping segments in seconds.
        """
        self.window_size = window_size
        self.sampling_rate = sr
        self.hop_size = hop_size
        self.clip_pad_max_ratio = clip_pad_max_ratio

    @classmethod
    def from_config(cls, config_path: Optional[Path] = None) -> "AudioNorm":
        """
        Use settings in config.yaml to normalize audio. 
        
        Uses the default hardcoded path but can be changed by the optional parameter config_path

        Params:
            config_path (str): The file path for the config.yaml file (OPTIONAL)
        """

        if config_path is None:
            config_path = find_config_path(__file__)

        with open(config_path, "r") as f:
            config = yaml.safe_load(f)

        try:
            shared_cfg = config["shared"]
            audio_cfg = config["audio_config"]
            merged_cfg = {
                "window_size": shared_cfg["window_size"],
                "sr": shared_cfg["sr"],
                "hop_size": audio_cfg["hop_size"],
                "clip_pad_max_ratio": audio_cfg["clip_pad_max_ratio"]
            }
        except KeyError as e:
            raise ValueError(
                f"config.yaml is missing expected key {e}. Check that 'shared.sr', 'shared.window_size', and 'audio_config' are present."
            ) from e

        return cls(**merged_cfg)

    def normalize_sampling_rate(self, audio: np.ndarray, orig_sr: int, target_sr: int) -> np.ndarray:
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

    def silence_padding(self, audio_samples: np.ndarray, orig_sr: int) -> Tuple[np.ndarray, int]:
        """
        This method takes the audio and normalizes the sampling rate to the value set in config.yaml. It then
        pads the audio at random amounts at the start and the end to make its length equal to the window size.

        Args:
            audio_samples (np.ndarray) : The audio samples of shape (Number_of_Samples, )
            orig_sr (int) : The sampling rate of the audio

        Returns:
            (The padded audio samples, sampling rate set in config.yaml)
        """
        # Get the target number of samples
        target_samples = int(self.window_size * self.sampling_rate)

        # If the audio is at device sampling rate, resample it
        if orig_sr != self.sampling_rate:
            audio_samples = self.normalize_sampling_rate(audio_samples, orig_sr, self.sampling_rate)

        # Get current number of samples in audio
        audio_samples_amount = len(audio_samples)

        # Calculate the number of samples that need to be padded
        pad_amount = target_samples - audio_samples_amount

        # Check if padding is really needed
        if pad_amount <= 0:
            return audio_samples, self.sampling_rate

        # Generate a random number of samples to pad at start of the call, using a uniform distribution
        pad_start_amount = random.randint(0, pad_amount)
        pad_end_amount = target_samples - audio_samples_amount - pad_start_amount

        audio_samples = np.pad(audio_samples, (pad_start_amount, pad_end_amount), mode='constant', constant_values=0.0)

        return audio_samples, self.sampling_rate
    
    def segmentation(self, audio: np.ndarray, orig_sr: int) -> Tuple[list[np.ndarray], int]:
        """
        Splits audio into overlapping fixed-length segments using a sliding window.
        Incomplete final windows are padded with silence at the end.

        Args:
            audio (np.ndarray): 1-D array of audio samples, shape (N,)

        Returns:
            List of np.ndarray segments, each of shape (window_samples,)
        """

        # Compare audio sampling rate with sampling rate used by config.yaml file
        if orig_sr != self.sampling_rate:
            audio = self.normalize_sampling_rate(audio, orig_sr, self.sampling_rate)
            print(f"Normalizing sampling rate from {orig_sr} to {self.sampling_rate}")

        # Convert time durations to integer sample counts
        window_samples = int(self.window_size * self.sampling_rate)
        hop_samples    = int(self.hop_size   * self.sampling_rate)

        segments = []
        start = 0

        while start < len(audio):
            # Slice 1 window segment from the audio
            # Numpy automatically handles cases where the slice exceeds the array length, returning a shorter array
            segment = audio[start : start + window_samples]

            # If this slice is shorter than the window, pad it to the full window size using silence at the end
            if len(segment) < window_samples:
                segment = self.silence_padding(segment, orig_sr)

            segments.append(segment)
            start += hop_samples   # increment by hop

        return (segments, self.sampling_rate)

    def random_clipping(self, audio_samples: np.ndarray, orig_sr: int) -> Tuple[np.ndarray, int]:
        """
        """

        # Check if audio has same sampling rate as in config.yaml file
        if orig_sr != self.sampling_rate:
            audio_samples = self.normalize_sampling_rate(audio_samples, orig_sr, self.sampling_rate)

        # Calculate the number of samples in a window
        window_sample_amount = int(self.window_size * self.sampling_rate)
        # Calculate the number of samples in the audio
        audio_sample_amount = len(audio_samples)
        # Calculate the maximum amount of silence padding we can apply at the start and end
        max_pad_amount = int(self.clip_pad_max_ratio * window_sample_amount)

        # If audio length < window size, return the original audio
        if audio_sample_amount <= window_sample_amount:
            return audio_samples, self.sampling_rate

        # Calculate the random amount by which you will pad the audio at start and end
        random_pad_start_amount = random.randint(0,max_pad_amount)
        random_pad_end_amount = random.randint(0, max_pad_amount)

        # Obtain the padded audio
        padded_audio = np.pad(audio_samples, (random_pad_start_amount, random_pad_end_amount), mode="constant", constant_values=0.0)

        # Calculate the max_clip_start_index
        max_clip_start_index = len(padded_audio) - window_sample_amount

        # Get the clip start index
        clip_start_index = random.randint(0, max_clip_start_index)

        # Cut the clip from the audio
        clipped_audio = padded_audio[clip_start_index: clip_start_index + window_sample_amount]

        return clipped_audio, self.sampling_rate




if __name__ == "__main__":
    audionorm = AudioNorm.from_config()
    print("Ran comfortably")