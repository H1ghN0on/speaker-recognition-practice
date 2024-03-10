import os
import sys
sys.path.append(os.path.realpath('..'))
from sr_labs_book.common import download_dataset, extract_dataset
from math import sqrt, pi
from scipy.fftpack import dct
import numpy as np
from matplotlib.pyplot import hist, plot, show, grid, title, \
xlabel, ylabel, legend, axis, imshow
from mpl_toolkits.axes_grid1 import make_axes_locatable
from skimage.morphology import opening, closing
from torchaudio.transforms import Resample
from multiprocessing import Pool
import torchaudio

from graphs import make_oscillogram, make_frc, make_mfcc, make_normalized_mfcc, make_first_component
from utils import split_meta_line, preemphasis, framing, power_spectrum, compute_fbank_filters, \
    compute_fbanks_features, compute_mfcc, mvn_floating

lab1_directory = "../sr_labs_book/lab1"
save_directory = "./data"

# Вынести в глобальные utils
# with open(f'{lab1_directory}/../data/lists/datasets.txt', 'r') as f:
#     lines = f.readlines()
# download_dataset(lines, user='voxceleb1902', password = 'nx0bl2v2', save_path = save_directory)
# extract_dataset(save_path=f'{save_directory}/voxceleb1_test', fname=f'{lab1_directory}/../data/vox1_test_wav.zip')


def load_signal(path_to_wav):
    signal, sample_rate = torchaudio.load(path_to_wav)
    signal = signal.numpy().squeeze(axis=0)
    signal = signal/np.abs(signal).max()
    return signal, sample_rate


def compute_feats(signal, sample_rate, fbanks, with_graphs = False):
    emphasized_signal = preemphasis(signal)
    frames = framing(emphasized_signal) # [[][][]], list of frames
    pow_frames = power_spectrum(frames) # [[][][]], power spectrums of each frame
    filter_banks_features = compute_fbanks_features(pow_frames, fbanks) # [[][][]], result of passing power spectrum through fbank filters then log it
    mfcc = compute_mfcc(filter_banks_features, num_ceps=20) # [[][][][]], mfcc, 20 components
    #filter_banks_features_mvn = mvn_floating(filter_banks_features, 150, 150) # [[][][][][]] normalize and scaling fbanks_features
    #mfcc_mvn = mvn_floating(mfcc, 150, 150) # [[][][]] normalize and scaling mfcc

    if with_graphs:
        make_oscillogram(signal, emphasized_signal, sample_rate)
        make_frc(fbanks)
        make_mfcc(filter_banks_features, mfcc)
        #make_normalized_mfcc(filter_banks_features_mvn, mfcc_mvn)
        

    return filter_banks_features, mfcc

def main():

    path_to_meta = f'{lab1_directory}/metadata/meta.txt'
    p = Pool(1)
    with open(path_to_meta, 'r') as f:
        list_lines = f.readlines()

    speaker_ids, genders, paths = zip(*p.map(split_meta_line, list_lines[1:]))

    #path_to_wav = paths[0]
    #signal = signal[0:int(3.5 * sample_rate)] # keep the first 3.5 s
    
    male_fb_features = []
    female_fb_features = []
    male_mfcc_features = []
    female_mfcc_features = []

    fbanks = compute_fbank_filters(nfilt=40, sample_rate=16000, NFFT=512)  #[[][][][][]], fbank filters for power_spectrums, 40 filters in total

    for (path_to_wav, gender) in zip(paths, genders):
        signal, sample_rate = load_signal(path_to_wav)
        filter_banks_mvn, mfcc_mvn = compute_feats(signal, sample_rate, fbanks)
        if gender == 'm':
            male_fb_features.append(filter_banks_mvn)
            male_mfcc_features.append(mfcc_mvn)
        else:
            female_fb_features.append(filter_banks_mvn)
            female_mfcc_features.append(mfcc_mvn)

    # combine fbank features for each gender

    male_fb_features = np.concatenate(male_fb_features)
    female_fb_features = np.concatenate(female_fb_features)

    # combine mfcfor each gender

    male_mfcc_features = np.concatenate(male_mfcc_features)
    female_mfcc_features = np.concatenate(female_mfcc_features)

    # build the first component 

    comp_number = 1
    make_first_component(male_fb_features[:, comp_number], female_fb_features[:, comp_number])

if __name__ ==  '__main__':
    main()
    
