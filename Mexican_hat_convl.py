import numpy as np
import matplotlib.pyplot as plt
from scipy.io import wavfile
from scipy.signal import convolve, hilbert


def mexican_hat_wavelet(length, width):
    t = np.linspace(-width / 2, width / 2, length)
    return (1 - t**2) * np.exp(-t**2 / 2)


audio_file = 'in5.wav'
sample_rate, audio_data = wavfile.read(audio_file)


if len(audio_data.shape) > 1:
    audio_data = audio_data.mean(axis=1)


wavelet_width = 1.0  # width of the wavelet
wavelet_length = 100  # Length of the wavelet in samples
wavelet = mexican_hat_wavelet(wavelet_length, wavelet_width)


convolved_audio = convolve(audio_data, wavelet, mode='same')


analytic_signal = hilbert(audio_data)
envelope = np.abs(analytic_signal)


plt.figure(figsize=(10, 8))


plt.subplot(3, 1, 1)
plt.plot(np.linspace(0, len(audio_data) / sample_rate, len(audio_data)), audio_data)
plt.title('Original Audio Signal')
plt.xlabel('Time [s]')
plt.ylabel('Amplitude')


plt.subplot(3, 1, 2)
plt.plot(np.linspace(0, len(convolved_audio) / sample_rate, len(convolved_audio)), convolved_audio)
plt.title('Convolved Audio Signal with Mexican Hat Wavelet')
plt.xlabel('Time [s]')
plt.ylabel('Amplitude')


plt.subplot(3, 1, 3)
plt.plot(np.linspace(0, len(envelope) / sample_rate, len(envelope)), envelope)
plt.title('Envelope of the Audio Signal')
plt.xlabel('Time [s]')
plt.ylabel('Amplitude')

plt.tight_layout()
plt.show()
