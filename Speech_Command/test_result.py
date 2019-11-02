import numpy as np
import os
import sys

def main():
	label = np.asarray(["yes", "no", "up", "down", "left", "right", "on", "off", "stop", "go"])
	searchRoot="/home/leeanghsuan/Downloads/output_audio"

	for i in range(0, 6):
		for j in range(0, 10):
			if i != j:
				searchDir = searchRoot + "/" + label[i] + "/" + label[j]
				for k in range(0, 50):
					print("i = {}\tj = {}\tk = {}".format(i, j, k))
					inputPath = searchDir + "/" + str(k) + ".wav"
					if os.path.isfile(inputPath) != 0:
						print(inputPath)
						command="python3 verify.py --wavPath " + inputPath + " --label " + label[i] + " --target " + label[j]
						os.system(command)

if __name__ == "__main__":
	main()