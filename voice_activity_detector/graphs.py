import matplotlib.pyplot as plt

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
