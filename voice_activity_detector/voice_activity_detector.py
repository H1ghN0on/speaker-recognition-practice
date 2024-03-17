import os
import torchaudio
import numpy as np

from utils import load_vad_markup, framing, frame_energy, norm_energy, gmm_train, eval_frame_post_prob
from graphs import make_signal_with_rttm, make_frames_energy, make_frames_approximate, make_real_VAD, make_morph_filter_VAD
from skimage.morphology import opening, closing


def load_signal(path_to_wav):
    signal, sample_rate = torchaudio.load(path_to_wav)
    signal = signal.numpy().squeeze(axis=0)
    signal = signal/np.abs(signal).max()
    return signal, sample_rate

def compute_vad(signal, window_format, vad_markup_ideal, with_graphs = False):

    # Framing
    window, shift = window_format
    frames = framing(signal, window=window, shift=shift)
    E = frame_energy(frames)
    E_norm = norm_energy(E)

    # Gaussian probability density function (PDF)
    gauss_pdf = lambda value, m, sigma: 1/((abs(sigma) + 1e-10)*np.sqrt(2*np.pi))*np.exp(-(value - m)**2/(2*(abs(sigma) + 1e-10)**2))

    # Gaussian training
    w, m, sigma = gmm_train(E_norm, gauss_pdf, n_realignment=10)
    GMM_pdf = np.zeros(len(E_norm))
    for j in range(len(m)):
        GMM_pdf = GMM_pdf + gauss_pdf(sorted(E_norm), m[j], sigma[j])

    # Detect voice activity 
    g0 = eval_frame_post_prob(E_norm, gauss_pdf, w, m, sigma)
    vad_thr = 0.3 # threshold of voice activity detector

    vad_frame_markup_real = (g0 < vad_thr).astype('float32') # frame VAD's markup
    vad_markup_real = np.zeros(len(signal)).astype('float32') # sample VAD's markup

    for idx in range(len(vad_frame_markup_real)):
        vad_markup_real[idx*shift:shift+idx*shift] = vad_frame_markup_real[idx]

    vad_markup_real[len(vad_frame_markup_real)*shift - len(vad_markup_ideal):] = vad_frame_markup_real[-1]

    # Morphology filter
    mask_size = 6000
    vad_markup_real_filt = closing(vad_markup_real, np.ones(mask_size))
    vad_markup_real_filt = opening(vad_markup_real_filt, np.ones(mask_size))

    if with_graphs:
        make_frames_energy(E_norm)
        make_frames_approximate(sorted(E_norm), GMM_pdf)
        make_real_VAD(signal, vad_markup_real)
        make_morph_filter_VAD(signal, vad_markup_ideal, vad_markup_real_filt)

        

def main():

    # Path to files
    path_to_wav = os.path.join('../data/voxceleb1_test/wav', 'id10271','1gtz-CUIygI/00006.wav')
    path_to_rttm = os.path.join('../sr_labs_book/lab2/ground_truth/rttm', 'id10271_1gtz-CUIygI_00006.rttm')

    # Load signal
    signal, sample_rate = load_signal(path_to_wav)
    vad_markup_ideal = load_vad_markup(path_to_rttm, signal, sample_rate)
    

    window = 320 # window size in samples
    shift = 160 # window shift in samples

    compute_vad(signal, [window, shift], vad_markup_ideal, True)


if __name__ ==  '__main__':
    main()