import os
import torchaudio
import numpy as np

from utils import load_vad_markup, framing, frame_energy, norm_energy, gmm_train
from graphs import make_signal_with_rttm, make_frames_energy, make_frames_approximate


def main():

    # Path to files
    path_to_wav = os.path.join('../data/voxceleb1_test/wav', 'id10271','1gtz-CUIygI/00006.wav')
    path_to_rttm = os.path.join('../sr_labs_book/lab2/ground_truth/rttm', 'id10271_1gtz-CUIygI_00006.rttm')

    # Load signal
    signal, sample_rate = torchaudio.load(path_to_wav)
    signal = signal.numpy().squeeze(axis=0)
    signal = signal/np.abs(signal).max()

    # Load ideal VAD's markup
    vad_markup_ideal = load_vad_markup(path_to_rttm, signal, sample_rate)
    
    # Plot signal
    #make_signal_with_rttm(signal, vad_markup_ideal)


    # squared_signal = signal**2

    # Frame signal with overlap
    window = 320 # window size in samples
    shift = 160 # window shift in samples

    frames = framing(signal, window=window, shift=shift)

    E = frame_energy(frames)
    E_norm = norm_energy(E)

    make_frames_energy(E_norm)

    # Gaussian probability density function (PDF)
    gauss_pdf = lambda value, m, sigma: 1/((abs(sigma) + 1e-10)*np.sqrt(2*np.pi))*np.exp(-(value - m)**2/(2*(abs(sigma) + 1e-10)**2))

    w, m, sigma = gmm_train(E_norm, gauss_pdf, n_realignment=10)

    GMM_pdf = np.zeros(len(E_norm))
    for j in range(len(m)):
        GMM_pdf = GMM_pdf + gauss_pdf(sorted(E_norm), m[j], sigma[j])

    make_frames_approximate(sorted(E_norm), GMM_pdf)

if __name__ ==  '__main__':
    main()