import numpy as np
import soundfile as sf

def generate_sinusoid(frequency, duration, amplitude, samplerate):

    t = np.linspace(0, duration, int(samplerate * duration), endpoint=False)
    signal = amplitude * np.sin(2 * np.pi * frequency * t)
    return signal


#main-----------------------------------------------------

sinusoid = generate_sinusoid(7500, 5.0, 0.5, 44000)
sf.write("sinusoid.wav", sinusoid, 44000)
print("Audio file saved")