#!/usr/bin/env python
# coding: utf-8

# In[63]:


import numpy as np
from scipy.fft import fft, fftfreq
from scipy.io import wavfile
import matplotlib.pyplot as plt
import sys
np.set_printoptions(threshold=sys.maxsize)


# In[64]:


sample_rate, data = wavfile.read("final.wav")


# In[70]:


def dft(data, sample_rate):
    fourier = abs(fft(data))
    timestep = 1/float(sample_rate)
    freq = fftfreq(len(fourier), timestep)
    plt.plot(freq[range(len(fourier)//2)], fourier[range(len(fourier)//2)])

# fourier = abs(fft(data))
# timestep = 1/float(sample_rate)
# freq = fftfreq(len(fourier), timestep)
# plt.plot(freq[range(len(fourier)//2)], fourier[range(len(fourier)//2)])
# plt.show()

dft(data, sample_rate)
plt.savefig("sample_fft.png", dpi=512)


# In[96]:


# amsc, am modulation
carrier_hz = 3000.0
signal_amsc = np.zeros_like(data, dtype=float)
signal_am = np.zeros_like(data, dtype=float)
carrier_signal = np.zeros_like(data, dtype=float)
time = np.zeros_like(data, dtype=float)
for i in range(len(data)):
    base = data[i]/32768.0
    carrier_sample = np.cos(carrier_hz * (i/sample_rate) * 2 * np.pi)
    signal_am[i] = signal_amsc[i] = base * carrier_sample
    signal_am[i] += carrier_sample
    signal_am[i] /= 2
    signal_amsc[i] *= 32768.0
    signal_am[i] *= 32768.0
    carrier_signal[i] = carrier_sample * 32768.0
    time[i] = i/sample_rate
# wavfile.write("amsc_test.wav", sample_rate, signal_amsc.astype(np.int16))
# wavfile.write("carrier_test.wav", sample_rate, carrier_signal.astype(np.int16))
# wavfile.write("am_test.wav", sample_rate, signal_am.astype(np.int16))

plt.plot(time, signal_am)
plt.grid()
plt.axhline(linewidth=1, color='black')
plt.axvline(linewidth=1, color='black')
plt.show()
dft(signal_amsc, sample_rate)


# In[100]:


# amsc demodulation

signal_amsc_demod = np.zeros_like(signal_amsc, dtype=float)
for i in range(len(signal_amsc)):
    signal_amsc_demod[i] = signal_amsc[i]
    carrier_sample = np.cos(carrier_hz * (i/44100) * 2 * np.pi)
    signal_amsc_demod[i] *= carrier_sample
wavfile.write("amsc_demod_test.wav", sample_rate, signal_amsc_demod.astype(np.int16))
dft(signal_amsc_demod, sample_rate)
plt.savefig("amsc_demod_fft.png", dpi=512)


# In[56]:


# am demodulation
signal_am_demod = np.zeros_like(signal_am, dtype=float)
for i in range(len(signal_am)):
    signal_am_demod[i] = abs(signal_am[i])
wavfile.write("am_test_demod.wav", sample_rate, signal_am_demod.astype(np.int16))


# In[57]:


import numpy, math
from numpy import fft

SAMPLE_RATE = 44100 # Hz
NYQUIST_RATE = SAMPLE_RATE / 2.0
FFT_LENGTH = 512

def lowpass_coefs(cutoff):
        cutoff /= (NYQUIST_RATE / (FFT_LENGTH / 2.0))

        # create FFT filter mask
        mask = []
        negatives = []
        l = FFT_LENGTH // 2
        for f in range(0, l+1):
                rampdown = 1.0
                if f > cutoff:
                        rampdown = 0
                mask.append(rampdown)
                if f > 0 and f < l:
                        negatives.append(rampdown)

        negatives.reverse()
        mask = mask + negatives

        # Convert FFT filter mask to FIR coefficients
        impulse_response = fft.ifft(mask).real.tolist()

        # swap left and right sides
        left = impulse_response[:FFT_LENGTH // 2]
        right = impulse_response[FFT_LENGTH // 2:]
        impulse_response = right + left

        b = FFT_LENGTH // 2
        # apply triangular window function
        for n in range(0, b):
                    impulse_response[n] *= (n + 0.0) / b
        for n in range(b + 1, FFT_LENGTH):
                    impulse_response[n] *= (FFT_LENGTH - n + 0.0) / b

        return impulse_response

def lowpass(original, cutoff):
        coefs = lowpass_coefs(cutoff)
        return numpy.convolve(original, coefs)


# In[59]:


# fm modulation
fm_carrier_hz = 10000
max_deviation_hz = 1000
phase = 0
signal_fm = np.zeros_like(data, dtype=float)
for n in range(len(data)):
    phase += (data[n]/32768.0) * np.pi * max_deviation_hz / sample_rate
    phase %= 2 * np.pi
    # quadrature i, q
    i = np.cos(phase)
    q = np.sin(phase)
    
    carrier = 2 * np.pi * fm_carrier_hz * (n/sample_rate)
    output = i * np.cos(carrier) - q * np.sin(carrier)
    signal_fm[n] = output * 32768.0
wavfile.write("fm_test.wav", sample_rate, signal_fm.astype(np.int16))


# In[60]:


# fm demodulation
import random
from math import atan2
input_signal = signal_fm
input_signal /= 32768.0
init_carrier_phase = 2.73 * np.pi * 2
prev_angle = 0.0
istream = []
qstream = []
for n in range(len(input_signal)):
    carrier = 2 * np.pi * fm_carrier_hz * (n/sample_rate) + init_carrier_phase
    istream.append(input_signal[n] * np.cos(carrier))
    qstream.append(input_signal[n] * np.sin(carrier))
istream = lowpass(istream, 1500)
qstream = lowpass(istream, 1500)
prev_output = 0
signal_fm_demod = np.zeros(len(istream), dtype=float)
for n in range(len(istream)):
    i = istream[n]
    q = qstream[n]
    
    # phase of i-q
    angle = atan2(q, i)
    angle_change = prev_angle - angle
    
    # failsafe if unexpectedly large angle change
    if angle_change > np.pi:
        angle_change -= 2 * np.pi
    elif angle_change < -np.pi:
        angle_change += 2 * np.pi
    last_angle = angle
    
    output = angle_change / (np.pi * max_deviation_hz / sample_rate)
    # failsafe if unexpectedly large output
    if abs(output) >= 1:
        output = prev_output
    prev_output = output
    signal_fm_demod[n] = output
signal_fm_demod *= 32768.0
print(signal_fm_demod)
wavfile.write("fm_demod_test.wav", sample_rate, signal_fm_demod.astype(np.int16))


# In[ ]:




