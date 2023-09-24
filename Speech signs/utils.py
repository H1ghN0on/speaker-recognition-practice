import matplotlib.pyplot as plt

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


def make_oscillogram(signal, emphasized_signal, sample_rate):
    """
    :param signal: input signal
    :param empasized_signal: input signal after preemphasize
    :param sample_rate: sample rate of signal
    """
    plt.figure(figsize=(15, 10))
    plt.subplots_adjust(wspace=0, hspace=0.5)
    plot_a = plt.subplot(311)
    plot_a.plot(signal)
    plot_a.set_xlabel('n')
    plot_a.set_ylabel('x(n)')
    plot_a.title.set_text('Original signal')
    plot_a.grid()
    plot_b = plt.subplot(312)
    plot_b.specgram(signal, NFFT=1024, Fs=sample_rate, noverlap=900)
    plot_b.set_xlabel('Time, s')
    plot_b.set_ylabel('Frequency, Hz')
    plot_b.title.set_text('Spectrogram of original signal')
    plot_c = plt.subplot(313)
    plot_c.specgram(emphasized_signal, NFFT=1024, Fs=sample_rate,
    noverlap=900)
    plot_c.set_xlabel('Time, s')
    plot_c.set_ylabel('Frequency, Hz')
    plot_c.title.set_text('Spectrogram of emphasized signal')
    plt.show()
