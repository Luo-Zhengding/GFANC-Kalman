import os 
import torch
import torchaudio
import torchaudio.transforms as T
import matplotlib.pyplot as plt
import scipy.signal as signal
import soundfile

#--------------------------------------------------------------
# untion: print_stats()
# Description : print the information of wave file.
#--------------------------------------------------------------
def print_stats(waveform, sample_rate=None, src=None):
    if src:
        print("-" * 10)
        print("Source:", src)
        print("-" * 10)
    if sample_rate:
        print("Sample Rate:", sample_rate)
        print("Shape:", tuple(waveform.shape))
        print("Dtype:", waveform.dtype)
        print(f" - Max:     {waveform.max().item():6.3f}")
        print(f" - Min:     {waveform.min().item():6.3f}")
        print(f" - Mean:    {waveform.mean().item():6.3f}")
        print(f" - Std Dev: {waveform.std().item():6.3f}")
        print()
        print(waveform)
        print()

#-------------------------------------------------------------- 
# Function: plot_waveform()
# Decription : plot the waveform of the wave file 
#--------------------------------------------------------------
def plot_waveform(waveform, sample_rate, title="Waveform", xlim=None, ylim=None):
    waveform = waveform.numpy()

    num_channels, num_frames = waveform.shape
    time_axis = torch.arange(0, num_frames) / sample_rate

    figure, axes = plt.subplots(num_channels, 1)
    if num_channels == 1:
        axes = [axes]
    for c in range(num_channels):
        axes[c].plot(time_axis, waveform[c], linewidth=1)
        axes[c].grid(True)
        if num_channels > 1:
            axes[c].set_ylabel(f'Channel {c+1}')
        if xlim:
            axes[c].set_xlim(xlim)
        if ylim:
            axes[c].set_ylim(ylim)
    figure.suptitle(title)
    plt.show(block=True)

#--------------------------------------------------------------
# Function: plot_specgram()
# Decription : plot the powerspecgram of the wave file 
#-------------------------------------------------------------
def plot_specgram(waveform, sample_rate, title="Spectrogram", xlim=None):
    waveform = waveform.numpy()

    num_channels, num_frames = waveform.shape
    time_axis = torch.arange(0, num_frames) / sample_rate

    figure, axes = plt.subplots(num_channels, 1)
    if num_channels == 1:
        axes = [axes]
    for c in range(num_channels):
        axes[c].specgram(waveform[c], Fs=sample_rate)
        if num_channels > 1:
            axes[c].set_ylabel(f'Channel {c+1}')
        if xlim:
            axes[c].set_xlim(xlim)
    figure.suptitle(title)
    plt.show(block=True)

#--------------------------------------------------------------
# Function: resample_wav()
# Description: resample the waveform using resample_rate
#--------------------------------------------------------------
def resample_wav(waveform, sample_rate, resample_rate):
    resampler = T.Resample(sample_rate, resample_rate, dtype=waveform.dtype)
    resampled_waveform = resampler(waveform)
    return resampled_waveform


#--------------------------------------------------------------
# Function: loading_real_wave_noise()
# Description: loading real noise, filter and resample it. 
#--------------------------------------------------------------
def loading_real_wave_noise(folde_name, sound_name):
    waveform, sample_rate = soundfile.read(os.path.join(folde_name, sound_name))
    bandpass_filter = signal.firwin(1024, [20, 7980], pass_zero='bandpass', window ='hamming', fs=16000) # filter by bandpass filter
    waveform = signal.lfilter(bandpass_filter, 1, waveform) # numpy.ndarray
    waveform = torch.from_numpy(waveform).type(torch.float).unsqueeze(dim=0) # from numpy to tensor
    
    resample_rate = 16000
    waveform = resample_wav(waveform, sample_rate, resample_rate) # resample
    return waveform, resample_rate

"""
# no bandpass filter
def loading_real_wave_noise(folde_name, sound_name):
    SAMPLE_WAV_SPEECH_PATH = os.path.join(folde_name, sound_name)
    waveform, sample_rate = torchaudio.load(SAMPLE_WAV_SPEECH_PATH)
    resample_rate = 16000
    waveform = resample_wav(waveform, sample_rate, resample_rate)
    return waveform, resample_rate
"""