import matplotlib.pyplot as plt
import numpy as np

def make_signal_with_rttm(signal, vad_markup_ideal):
    plt.figure(figsize=(16, 4))
    plt.plot(signal, color='green')
    plt.plot(vad_markup_ideal, color='red')
    plt.xlabel('$n$'); 
    plt.ylabel('$x(n)$')
    plt.title('Waveform and markup of ideal VAD'); 
    plt.grid()
    plt.legend(['Waveform', 'Ideal VAD']); 
    plt.show()


def make_frames_energy(E_norm):
    plt.hist(E_norm, int(np.sqrt(len(E_norm))), histtype='step', color='green')
    plt.xlabel('$e_{norm}$')
    plt.ylabel('$\widehat{W}(e_{norm})$')
    plt.title('Histogram of frame energies')
    plt.grid()
    plt.show()

def make_frames_approximate(E_norm, GMM_pdf):
    plt.plot(E_norm, GMM_pdf, color='green')
    plt.xlabel('$e_{norm}$'); 
    plt.ylabel('$W(e_{norm})$')
    plt.title('PDF for frame energies'); 
    plt.grid(); 
    plt.show()


def make_real_VAD(signal, vad_markup_real):
    plt.figure(figsize=(16, 4))
    plt.plot(signal, color='green')
    plt.plot(vad_markup_real, color='blue')
    plt.xlabel('$n$'); 
    plt.ylabel('$x(n)$')
    plt.title('Waveform and markup of real VAD'); 
    plt.grid()
    plt.legend(['Waveform', 'Real VAD']); 
    plt.show()

def make_morph_filter_VAD(signal, vad_markup_ideal, vad_markup_real_filt):
    plt.figure(figsize=(16, 4))
    plt.plot(signal, color='green')
    plt.plot(vad_markup_ideal, color='red')
    plt.plot(vad_markup_real_filt, color='blue')
    plt.xlabel('$n$'); 
    plt.ylabel('$x(n)$')
    plt.title('Waveform and markup of ideal/real VAD'); 
    plt.grid()
    plt.legend(['Waveform', 'Ideal VAD', 'Real VAD']); 
    plt.show()