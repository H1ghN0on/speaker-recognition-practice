from math import cos, log10, pi
import numpy as np

def split_meta_line(line, delimiter=' '):    
    """
    :param line: lines of metadata
    :param delimiter: delimeter
    :return: speaker_id: speaker IDs: gender: gender: file_path: path to file
    """
    speaker_id, gender, file_path = line.split(delimiter) 

    if (file_path.endswith("\n")):
        file_path = file_path[:-1] 

    return speaker_id, gender, file_path

def preemphasis(signal, pre_emphasis=0.97):

    """
    :param signal: input signal
    :param pre_emphasis: preemphasis coeffitient
    :return: emphasized_signal: signal after pre-emphasis procedure
    """
    emphasized_signal = []
    for i in range(1, signal.size):
        emphasized_signal.append(signal[i] + -pre_emphasis * signal[i - 1])

    return emphasized_signal

def framing(emphasized_signal, sample_rate=16000, frame_size=0.025, frame_stride=0.01):
    # hamming window    
    """
    :param emphasized_signal: signal after pre-emphasis procedure
    :param sample_rate: signal sampling rate
    :param frame_size: sliding window size
    :param frame_stride: step
    :return: frames: output matrix [nframes x sample_rate*frame_size]
    """

    frame_length, frame_step = frame_size * sample_rate, frame_stride * sample_rate # convert from seconds to samples
    signal_length = len(emphasized_signal)
    frame_length = int(round(frame_length))
    frame_step = int(round(frame_step))
    num_frames = int(
        np.ceil(float(np.abs(signal_length - frame_length)) / frame_step)) # make sure that we have at least 1 frame
    
    pad_signal_length = num_frames * frame_step + frame_length
    z = np.zeros((pad_signal_length - signal_length))
    pad_signal = np.append(emphasized_signal, z) # pad Signal to make sure that all frames have equal number of samples without
                                                 # truncating any samples from the original signal
    print(pad_signal)
    frames = []

    i = 0
    while i < pad_signal_length - 1: 
        frame = pad_signal[i : i + frame_length]
        frames.append(frame)
        i += frame_step

    return frames

def power_spectrum(frames, NFFT=512):
    # Here you need to compute power spectum of framed signal with NFFT fft bins number

    """
    :param frames: framed signal
    :param NFFT: number of fft bins
    :return: pow_frames: framed signal power spectrum
    """

    mag_frames = np.absolute(np.fft.rfft(frames, NFFT))  # Magnitude of the FFT

    pow_frames = np.abs(mag_frames) ** 2

    return pow_frames

def compute_fbank_filters(nfilt=40, sample_rate=16000, NFFT=512):
    # Here you need to compute fbank filters (FBs) for special case (sample_rate & NFFT)

    """
    :param nfilt: number of filters
    :param sample_rate: signal sampling rate
    :param NFFT: number of fft bins in power spectrum
    :return: fbank [nfilt x (NFFT/2+1)]
    """
    
    low_freq_mel = 0
    high_freq = sample_rate / 2

    high_freq_mel = 2595 * log10(1 + high_freq / 700)
    

    mel_points = np.linspace(low_freq_mel, high_freq_mel, nfilt + 2) # equally spaced in mel scale

    hz_points = 700 * (pow(mel_points / 2595, 10) - 1)
    
    bin = np.floor((NFFT + 1) * hz_points / sample_rate)

    fbank = np.zeros((nfilt, int(np.floor(NFFT / 2 + 1))))
    for m in range(1, nfilt + 1):
        f_m_minus = int(bin[m - 1]) # left
        f_m = int(bin[m])           # center
        f_m_plus = int(bin[m + 1])  # right

        for k in range(f_m_minus, f_m):
            fbank[m - 1, k] = (k - bin[m - 1]) / (bin[m] - bin[m - 1])
        for k in range(f_m, f_m_plus):
            fbank[m - 1, k] = (bin[m + 1] - k) / (bin[m + 1] - bin[m])

    return fbank

def compute_fbanks_features(pow_frames, fbank):    
    """
    :param pow_frames: framed signal power spectrum, matrix [nframes x sample_rate*frame_size]
    :param fbank: matrix of the fbank filters [nfilt x (NFFT/2+1)] where NFFT: number of fft bins in power spectrum
    :return: filter_banks_features: log mel FB energies matrix [nframes x nfilt]
    """
    
    ###########################################################
    filter_banks_features = pow_frames * fbank
    
    ###########################################################

    filter_banks_features = np.where(filter_banks_features == 0, np.finfo(float).eps,
                                     filter_banks_features) # numerical stability
    filter_banks_features = np.log(filter_banks_features)

    return filter_banks_features

def compute_mfcc(filter_banks_features, num_ceps=20):
    # Here you need to compute MFCCs features using precomputed log mel FB energies matrix
    
    """
    :param filter_banks_features: log mel FB energies matrix [nframes x nfilt]
    :param num_ceps: number of cepstral components for MFCCs
    :return: mfcc: mel-frequency cepstral coefficients (MFCCs)
    """
    
    ###########################################################
    # Here is your code to compute mfcc features
    
    ###########################################################

    return mfcc