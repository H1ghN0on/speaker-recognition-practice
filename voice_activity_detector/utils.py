import numpy as np
import scipy.signal

from skimage.morphology import opening, closing


def load_vad_markup(path_to_rttm, signal, sample_rate):
    # Function to read rttm files and generate VAD's markup in samples
    
    vad_markup = np.zeros(len(signal)).astype('float32')
        
    ###########################################################
    # Here is your code
    
    rttm = open(path_to_rttm, "r")

    for line in rttm:
        rttm_line = line.split(" ")
        
        start_seconds, duration = float(rttm_line[3]), float(rttm_line[4])

        start_sample = int((start_seconds) * sample_rate)
        end_sample = int((start_seconds + duration) * sample_rate)

        vad_markup[start_sample:end_sample] = 1


    ###########################################################
    
    return vad_markup

def framing(signal, window=320, shift=160):
    # Function to create frames from signal

    shape   = (int((signal.shape[0] - window)/shift + 1), window)
    frames  = np.zeros(shape).astype('float32')

    ###########################################################
    # Here is your code

    frames = np.array([signal[i:i+window] for i in range(0, len(signal) - window + 1, shift)])
    
    ###########################################################
    
    return frames

def frame_energy(frames):
    # Function to compute frame energies
    
    E = np.zeros(frames.shape[0]).astype('float32')

    ###########################################################
    # Here is your code

    E = [np.sum(np.abs(frame) ** 2) for frame in frames]

    ###########################################################

    return E

def norm_energy(E):
    # Function to normalize energy by mean energy and energy standard deviation
    
    E_norm = np.zeros(len(E)).astype('float32')

    ###########################################################
    # Here is your code

    mean_energy = np.mean(E)
    std_energy = np.sqrt(np.var(E))


    E_norm = (E - mean_energy) / std_energy

    ###########################################################
    
    return E_norm

def gmm_train(E, gauss_pdf, n_realignment):
    # Function to train parameters of gaussian mixture model
    
    # Initialization gaussian mixture models
    w     = np.array([ 0.33, 0.33, 0.33])
    m     = np.array([-1.00, 0.00, 1.00])
    sigma = np.array([ 1.00, 1.00, 1.00])

    g = np.zeros([len(E), len(w)])


    for n in range(n_realignment):

        # E-step
        ###########################################################
        # Here is your code
        
        for i in range(len(E)):
            denominator = np.sum(w * gauss_pdf(E[i], m, sigma))
            for j in range(len(w)):
                nominator = w[j] * gauss_pdf(E[i], m[j], sigma[j])
                g[i][j] =  nominator / denominator

        ###########################################################

        # M-step
        ###########################################################
        # Here is your code
                
        for j in range(3):
            sum_for_wj = 0
            sum_for_mj = 0
            sum_for_sigmaj = 0

            for i in range(len(E)):
                sum_for_wj += g[i][j]
                sum_for_mj += g[i][j] * E[i]

            w[j] = sum_for_wj / len(E)
            m[j] = sum_for_mj / (len(E) * w[j])

            for i in range(len(E)):
                sum_for_sigmaj += g[i][j] * ((E[i] - m[j]) ** 2)

            sigma[j] = np.sqrt(sum_for_sigmaj / (len(E) * w[j]))

        ##########################################################

    return w, m, sigma

def eval_frame_post_prob(E, gauss_pdf, w, m, sigma):
    # Function to estimate a posterior probability that frame isn't speech

    g0 = np.zeros(len(E))

    ###########################################################
    # Here is your code

    ###########################################################
            
    return g0

def energy_gmm_vad(signal, window, shift, gauss_pdf, n_realignment, vad_thr, mask_size_morph_filt):
    # Function to compute markup energy voice activity detector based of gaussian mixtures model
    
    # Squared signal
    squared_signal = signal**2
    
    # Frame signal with overlap
    frames = framing(squared_signal, window=window, shift=shift)
    
    # Sum frames to get energy
    E = frame_energy(frames)
    
    # Normalize the energy
    E_norm = norm_energy(E)
    
    # Train parameters of gaussian mixture models
    w, m, sigma = gmm_train(E_norm, gauss_pdf, n_realignment=10)
    
    # Estimate a posterior probability that frame isn't speech
    g0 = eval_frame_post_prob(E_norm, gauss_pdf, w, m, sigma)
    
    # Compute real VAD's markup
    vad_frame_markup_real = (g0 < vad_thr).astype('float32')  # frame VAD's markup

    vad_markup_real = np.zeros(len(signal)).astype('float32') # sample VAD's markup
    for idx in range(len(vad_frame_markup_real)):
        vad_markup_real[idx*shift:shift+idx*shift] = vad_frame_markup_real[idx]

    vad_markup_real[len(vad_frame_markup_real)*shift - len(signal):] = vad_frame_markup_real[-1]
    
    # Morphology Filters
    vad_markup_real = closing(vad_markup_real, np.ones(mask_size_morph_filt)) # close filter
    vad_markup_real = opening(vad_markup_real, np.ones(mask_size_morph_filt)) # open filter
    
    return vad_markup_real

def reverb(signal, impulse_response):
    # Function to create reverberation effect
    
    signal_reverb = np.zeros(len(signal)).astype('float32')
    
    ###########################################################
    # Here is your code
    
    ###########################################################
    
    return signal_reverb

def awgn(signal, sigma_noise):
    # Function to add white gaussian noise to signal
    
    signal_noise = np.zeros(len(signal)).astype('float32')
    
    ###########################################################
    # Here is your code
    
    ###########################################################
    
    return signal_noise