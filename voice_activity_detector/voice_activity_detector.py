import os
import torchaudio
import numpy as np

from utils import load_vad_markup
from graphs import make_signal_with_rttm


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
    make_signal_with_rttm(signal, vad_markup_ideal)

if __name__ ==  '__main__':
    main()