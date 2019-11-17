import scipy.io.wavfile as wavfile
import numpy as np
import sys

def readWav(path):
	rate, data = wavfile.read(path)
	return data

def SNR(originalWav, attackedWav):
	signal = np.sum(np.square(originalWav))
	noise = np.sum(np.square(originalWav - attackedWav))
	return 10 * np.log10(signal / noise)

def calculateSNR(originalWavPath, attackedWavPath):
	rate, originalWav = wavfile.read(originalWavPath)
	rate, attackedWav = wavfile.read(attackedWavPath)
	snr = SNR(originalWav.astype('d'), attackedWav.astype('d'))
	return snr

if __name__ == '__main__':
	snr = calculateSNR(sys.argv[1], sys.argv[2])
	print(snr)