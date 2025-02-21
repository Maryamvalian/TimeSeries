import numpy as np
import soundfile as sf
from scipy.fft import fft
import matplotlib.pyplot as plt
# Assignment 2- CAS748
#short time fourier transform

def read_audio(file_path):
    data, samplerate = sf.read(file_path)
    return data, samplerate

def get_frames(data, window_size, stride):
    frames = []
    for i in range(0, len(data) - window_size + 1, stride):
        frames.append(data[i:i + window_size])
    return frames

def compute_fft(frames):
    fft_results = []
    for frame in frames:
        fft_values = fft(frame)
        magnitude = np.abs(fft_values[:len(frame) // 2])  # Keep only first half (symmetric)
        fft_results.append(magnitude)
    return fft_results


def plot_spectrogram(results, samplerate):
    results = np.array(results)
    plt.imshow(20 * np.log10(results.T + 1e-10), aspect='auto', origin='lower',
               extent=[0, len(results), 0, samplerate / 2])
    plt.show()

def process_audio(file_path, window_size, stride):
    data, samplerate = read_audio(file_path)
    frames = get_frames(data, window_size, stride)
    fft_results = compute_fft(frames)
    plot_spectrogram(fft_results, samplerate)



#--------Main----------------------
process_audio("a440.wav", 512, 128)      #Testcase
process_audio("in5.wav", 256, 16)