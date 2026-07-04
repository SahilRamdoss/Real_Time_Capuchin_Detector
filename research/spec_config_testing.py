import numpy as np
import matplotlib.pyplot as plt
from matplotlib.widgets import Slider
from scipy.signal import butter, sosfilt, spectrogram

# 1. Generate a noisy signal to test the Bandpass Filter
fs = 2000  # Sampling rate (Hz)
t = np.linspace(0, 2, 2 * fs, endpoint=False)

# Target signals
chirp = np.sin(2 * np.pi * (100 * t + 250 * t**2))  # Sweeps from 100Hz to 600Hz
clicks = np.zeros_like(t)
clicks[int(0.6 * fs)] = 7
clicks[int(1.4 * fs)] = 7

# Out-of-band noise components (to be removed by the filter)
low_hum = 0.6 * np.sin(2 * np.pi * 30 * t)       # 30 Hz noise
high_whistle = 0.6 * np.sin(2 * np.pi * 850 * t) # 850 Hz noise

signal_raw = chirp + clicks + low_hum + high_whistle

# 2. Helper function to generate a Mel Filterbank Matrix dynamically
def compute_mel_filterbank(n_mels, n_fft, fs):
    low_mel = 0
    high_mel = 2595 * np.log10(1 + (fs / 2) / 700)
    mel_points = np.linspace(low_mel, high_mel, n_mels + 2)
    hz_points = 700 * (10**(mel_points / 2595) - 1)
    
    bins = np.floor((n_fft + 1) * hz_points / fs).astype(int)
    fb = np.zeros((n_mels, int(n_fft // 2 + 1)))
    
    for m in range(1, n_mels + 1):
        for k in range(bins[m-1], bins[m]):
            fb[m-1, k] = (k - bins[m-1]) / max(1, bins[m] - bins[m-1])
        for k in range(bins[m], bins[m+1]):
            fb[m-1, k] = (bins[m+1] - k) / max(1, bins[m+1] - bins[m])
    return fb

# 3. Setup the Interactive UI Layout
fig, (ax_sig, ax_spec) = plt.subplots(2, 1, figsize=(11, 8), gridspec_kw={'height_ratios': [1, 2]})
plt.subplots_adjust(bottom=0.4)  # Leave space for 4 sliders

# Initial Parameters
init_order = 2
init_nfft = 256
init_hop = 64
init_nmels = 40

# Initial filtering and processing loop
sos = butter(init_order, [100, 700], btype='bandpass', fs=fs, output='sos')
signal_filtered = sosfilt(sos, signal_raw)

# Initial plots
line_raw, = ax_sig.plot(t, signal_raw, color='gray', alpha=0.4, label='Raw (with Noise)')
line_filt, = ax_sig.plot(t, signal_filtered, color='#d62728', alpha=0.8, label='Filtered')
ax_sig.set_title("Time Domain: Filter Order Impact")
ax_sig.legend(loc='upper right')
ax_sig.grid(True, alpha=0.3)

# Initial Mel Spectrogram calculation
f, t_spec, Sxx = spectrogram(signal_filtered, fs=fs, nperseg=init_nfft, noverlap=init_nfft - init_hop, nfft=init_nfft)
fb = compute_mel_filterbank(init_nmels, init_nfft, fs)
mel_Sxx = np.dot(fb, Sxx)

# --- CHANGED SHADING TO 'auto' HERE ---
im = ax_spec.pcolormesh(t_spec, np.arange(init_nmels), 10 * np.log10(mel_Sxx + 1e-10), shading='auto', cmap='magma')
ax_spec.set_title("Mel Spectrogram (Perceptual Time-Frequency Map)")
ax_spec.set_ylabel("Mel Scale (Bands)")
ax_spec.set_xlabel("Time (s)")

# 4. Construct Sliders
ax_color = 'lavender'
ax_slider_order = plt.axes([0.25, 0.25, 0.55, 0.03], facecolor=ax_color)
ax_slider_nfft  = plt.axes([0.25, 0.19, 0.55, 0.03], facecolor=ax_color)
ax_slider_hop   = plt.axes([0.25, 0.13, 0.55, 0.03], facecolor=ax_color)
ax_slider_nmels = plt.axes([0.25, 0.07, 0.55, 0.03], facecolor=ax_color)

s_order = Slider(ax_slider_order, 'Filter Order', 1, 8, valinit=init_order, valstep=1, valfmt='%d')
s_nfft  = Slider(ax_slider_nfft, 'n_fft (Window)', 64, 1024, valinit=init_nfft, valfmt='%0.0f')
s_hop   = Slider(ax_slider_hop, 'hop_length', 16, 512, valinit=init_hop, valfmt='%0.0f')
s_nmels = Slider(ax_slider_nmels, 'n_mels (Bands)', 16, 128, valinit=init_nmels, valstep=4, valfmt='%d')

# 5. Live Update logic
def update(val):
    order = int(s_order.val)
    raw_nfft = int(s_nfft.val)
    nfft = 2**int(np.round(np.log2(raw_nfft)))  # Snap to power of 2
    hop = int(s_hop.val)
    n_mels = int(s_nmels.val)
    
    if hop >= nfft:
        hop = nfft - 4
        
    # Step A: Re-filter the raw signal based on Filter Order
    sos_new = butter(order, [100, 700], btype='bandpass', fs=fs, output='sos')
    signal_filt_new = sosfilt(sos_new, signal_raw)
    line_filt.set_ydata(signal_filt_new)
    
    # Step B: Recompute linear spectrogram and map to custom Mel scale
    f_new, t_new, Sxx_new = spectrogram(signal_filt_new, fs=fs, nperseg=nfft, noverlap=nfft - hop, nfft=nfft)
    fb_new = compute_mel_filterbank(n_mels, nfft, fs)
    mel_Sxx_new = np.dot(fb_new, Sxx_new)
    
    # Step C: Redraw the updated Mel Spectrogram axis
    ax_spec.clear()
    # --- CHANGED SHADING TO 'auto' HERE TOO ---
    ax_spec.pcolormesh(t_new, np.arange(n_mels), 10 * np.log10(mel_Sxx_new + 1e-10), shading='auto', cmap='magma')
    ax_spec.set_title(f"Mel Spectrogram (n_fft={nfft} | hop={hop} | n_mels={n_mels})")
    ax_spec.set_ylabel("Mel Scale (Bands)")
    ax_spec.set_xlabel("Time (s)")
    
    fig.canvas.draw_idle()

s_order.on_changed(update)
s_nfft.on_changed(update)
s_hop.on_changed(update)
s_nmels.on_changed(update)

plt.show()