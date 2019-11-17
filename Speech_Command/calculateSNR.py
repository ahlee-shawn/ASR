import csv
import sys
import os
from snr import *

def main():
	label = np.asarray(["yes", "no", "up", "down", "left", "right", "on", "off", "stop", "go"])
	attackedRoot = "/home/leeanghsuan/Downloads/output_audio_2"
	originalRoot = "/home/leeanghsuan/Desktop/speech_dataset"

	for i in range(0, 10):
		for j in range(0, 10):
			if i != j:
				sourceLabel = label[i]
				targetLabel = label[j]
				originalDir = originalRoot + "/" + sourceLabel
				attackedDir = attackedRoot + "/" + sourceLabel + "/" + targetLabel
				fileList = np.asarray(os.listdir(attackedDir))
				for k in range(10):
					
					originalWavPath = originalDir + "/" + fileList[k]
					attackedWavPath = attackedDir + "/" + fileList[k]

					snr = calculateSNR(originalWavPath, attackedWavPath)
					print("i = {}\tj = {}\tk = {}\tSNR = {}\t\tPATH = {}".format(i, j, k, snr, attackedWavPath))

if __name__ == "__main__":
	main()