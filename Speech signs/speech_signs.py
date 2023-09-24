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
from utils import split_meta_line, preemphasis, make_oscillogram


lab1_directory = "../sr_labs_book/lab1"
save_directory = "./data"
# with open(f'{lab1_directory}/../data/lists/datasets.txt', 'r') as f:
#     lines = f.readlines()
# download_dataset(lines, user='voxceleb1902', password = 'nx0bl2v2', save_path = save_directory)
# extract_dataset(save_path=f'{save_directory}/voxceleb1_test', fname=f'{lab1_directory}/../data/vox1_test_wav.zip')

if __name__ ==  '__main__':
    path_to_meta = f'{lab1_directory}/metadata/meta.txt'
    p = Pool(1)
    with open(path_to_meta, 'r') as f:
        list_lines = f.readlines()

    speaker_ids, genders, paths = zip(*p.map(split_meta_line, \
    list_lines[1:]))
    path_to_wav = paths[0]
    # Load signal
    signal, sample_rate = torchaudio.load(path_to_wav)
    signal = signal.numpy().squeeze(axis=0)
    signal = signal/np.abs(signal).max()

    signal = signal[0:int(3.5 * sample_rate)] # keep the first 3.5 s
    emphasized_signal = preemphasis(signal) # emphasized signal

    make_oscillogram(signal, emphasized_signal, sample_rate)
    