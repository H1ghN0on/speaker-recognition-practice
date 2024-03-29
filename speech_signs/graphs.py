import matplotlib.pyplot as plt
import math

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

def make_frc(fbank):
    plt.figure(figsize=(15, 5))
    plot_a = plt.subplot()
    plt.subplots_adjust(wspace=0, hspace=1)
    nfilt = fbank.shape[0]
    for k in range(nfilt):
        plot_a.plot(fbank[k,:])
    plot_a.set_xlabel('Frequency bins')
    plot_a.set_ylabel('Amplitude')
    plot_a.title.set_text('FBank')
    plot_a.grid()
    plt.show()


def make_mfcc(filter_banks_features, mfcc):
    plt.figure(figsize=(15, 5))
    plt.subplots_adjust(wspace=0, hspace=0.5)
    plot_a = plt.subplot(211)
    plot_a.imshow(filter_banks_features.T, origin='lower')
    plot_a.set_xlabel('Time bins')
    plot_a.set_ylabel('Frequency band bins')
    plot_a.title.set_text('FBank log energies')
    plot_b = plt.subplot(212)
    im = plot_b.imshow(mfcc.T, origin='lower')
    plot_b.set_xlabel('Time bins')
    plot_b.set_ylabel('Coefficient bins')
    plot_b.title.set_text('MFCCs')
    plt.show()

def make_normalized_mfcc(filter_banks_features_mvn, mfcc_mvn):
    fig = plt.figure(figsize=(15, 5))
    plt.subplots_adjust(wspace=0, hspace=0.5)
    plot_b = plt.subplot(211)
    im_b = plot_b.imshow(filter_banks_features_mvn.T,
    origin='lower')
    plot_b.set_xlabel('Time bins')
    plot_b.set_ylabel('Frequency band bins')
    plot_b.title.set_text('Normalized FBank log energies')
    15
    plot_с = plt.subplot(212)
    im_c = plot_с.imshow(mfcc_mvn.T, origin='lower')
    plt.colorbar(im_c)
    plot_с.set_xlabel('Time bins')
    plot_с.set_ylabel('Coefficient bins')
    plot_с.title.set_text('Normalized MFCCs')
    plt.show()

def make_first_component(coeff1_male, coeff1_female):
    min_coeff1 = min(coeff1_male.min(), coeff1_female.min())
    max_coeff1 = min(coeff1_male.max(), coeff1_female.max())

    plt.hist(coeff1_male, int(math.sqrt(len(coeff1_male))), histtype='step', color='green', range=(min_coeff1, max_coeff1), density=1)
    plt.hist(coeff1_female, int(math.sqrt(len(coeff1_female))), histtype='step', color='red', range=(min_coeff1, max_coeff1), density=1)
    plt.xlabel('MFBs, 1st component'); 
    plt.ylabel('Histogram value')
    plt.title('Normalized histograms'); 
    plt.grid(); 
    plt.show()