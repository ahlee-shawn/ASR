import csv
import numpy as np
import os
import sys
import time

def main():

	with open('success.csv') as success_csv:
		success = np.asarray(list(csv.reader(success_csv))).reshape(10, 10).astype(float)
	with open('iteration.csv') as iteration_csv:
		iteration = np.asarray(list(csv.reader(iteration_csv))).reshape(10, 10).astype(float)

	label = np.asarray(["yes", "no", "up", "down", "left", "right", "on", "off", "stop", "go"])
	search_root="/home/leeanghsuan/Desktop/speech_dataset"

	stop = 0
	for i in range(0, 10):
		for j in range(0, 10):
			if i != j and stop == 0:
				source_label = label[i]
				search_dir = search_root + "/" + source_label
				file_list = np.asarray(os.listdir(search_dir))[: 10]
				for k in range(int(iteration[i][j]), 10):
					print("i = {}\tj = {}\tk = {}".format(i, j, k))
					input_path = search_dir + "/" + file_list[k]
					target_label = label[j]
					ouput_path = "/home/leeanghsuan/Desktop/Speech_Command/" + source_label + "/" + target_label + "/" + str(success[i][j]) + ".wav"
					command="python3 generate_audio_v1.py --wav_path " + input_path + " --new_wav_path " + ouput_path + " --target " + target_label
					exit_code = os.system(command)
					print(exit_code)
					# KeyboardInterrupt
					if exit_code == 2:
						success.tofile('success1.csv',sep=',',format='%d')
						iteration.tofile('iteration1.csv',sep=',',format='%d')
						os.remove("success.csv")
						os.remove("iteration.csv")
						os.rename("success1.csv", "success.csv")
						os.rename("iteration1.csv", "iteration.csv")
						time.sleep(0.5)
						sys.exit(1)
					# success
					if exit_code != 0:
						success[i][j] += 1.0
					iteration[i][j] += 1.0
					# save from time to time
					if k == 9:
						success.tofile('success1.csv',sep=',',format='%d')
						iteration.tofile('iteration1.csv',sep=',',format='%d')
						os.remove("success.csv")
						os.remove("iteration.csv")
						os.rename("success1.csv", "success.csv")
						os.rename("iteration1.csv", "iteration.csv")

	success.tofile('success1.csv',sep=',',format='%d')
	iteration.tofile('iteration1.csv',sep=',',format='%d')
	os.remove("success.csv")
	os.remove("iteration.csv")
	os.rename("success1.csv", "success.csv")
	os.rename("iteration1.csv", "iteration.csv")
	for i in range(10):
		for j in range(10):
			success[i][j] /= 10.0
	np.savetxt('accuracy.txt', success, delimiter=',', fmt='%.2f')
	success.tofile('accuracy1.csv',sep=',',format='%.2f')
	os.remove("accuracy.csv")
	os.rename("accuracy1.csv", "accuracy.csv")

if __name__ == "__main__":
	main()