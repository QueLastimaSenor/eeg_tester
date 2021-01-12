import numpy as np
import matplotlib.pyplot as plt
import os

def extraction(dirPath, classPath, duration, Create = 0, Segments = 2):
    fs = 512
    fs_factor = 8
    n_fft = 64
    nlap = 48
    #Create отвечает за создание папки для определенного класса
    if Create == 1:
        os.mkdir("img_data/" + classPath)
    image_directory = dirPath
    segments = Segments

    opened_drunk_images = os.listdir(image_directory)
    for i, name in enumerate(opened_drunk_images):
        data = np.loadtxt(image_directory + name)
        # plt.plot(np.linspace(0, len(data) / 512, len(data)), data) 
        # fft = np.fft.fft(data)
        # freqs = np.linspace(0, Fs, len(fft))
        # for i, item in enumerate(freqs):
        #     if item > 51 and item < 462:
        #        fft[i] = 0
        # plt.plot(freqs, np.abs(fft), alpha=0.4)
        # plt.show()
        # ifft = np.fft.ifft(fft)
        # plt.plot(np.linspace(0, len(data)/512, len(data)), np.real(ifft))
        # plt.show()
        new_data = np.zeros(int(len(data)/fs_factor))
        k = 0
        for j in range(0, int(len(data)), fs_factor):
            new_data[k] = data[j]
            k = k + 1
            if k == len(new_data):
                break
        #plt.plot(np.linspace(0, len(data)/Fs, len(data)), data)
        #plt.plot(np.linspace(0, len(new_data)/(Fs/Fs_factor), len(new_data)), new_data)
        #plt.show()
        data = new_data
        if len(data) >= (fs/fs_factor) * duration:
            finish = 0
            for d in range(segments - 1):
                start = finish
                finish = start + int((fs/fs_factor)*(duration/segments))
                #win = np.kaiser(n_fft, 8)
                plt.specgram(data[start:finish], noverlap=nlap, NFFT=n_fft, Fs=int(fs/fs_factor))
                #plt.show()
                plt.axis("off")
                print("{}, segment:{}".format(name, d + 1))
                plt.tight_layout()
                plt.savefig(f'img_data/{classPath}/{i}_{classPath}_segment_{d+1}.png', bbox_inches='tight', pad_inches=0)
                plt.clf()
