import csv
import numpy as np
import os
import sys
import time

def main():
	elapsed_time = 0
	with open('/home/leeanghsuan/Desktop/Speech_Command/success.csv') as success_csv:
		success = np.asarray(list(csv.reader(success_csv))).reshape(50, 50).astype(float)
	with open('/home/leeanghsuan/Desktop/Speech_Command/iteration.csv') as iteration_csv:
		iteration = np.asarray(list(csv.reader(iteration_csv))).reshape(50, 50).astype(float)
	with open('/home/leeanghsuan/Desktop/Speech_Command/time.txt', "r") as f:
		elapsed_time = float(f.readline())

	label = np.asarray(["yes", "no", "up", "down", "left", "right", "on", "off", "stop", "go"])
	search_root="/home/leeanghsuan/Desktop/speech_dataset"

	for i in range(0, 10):
		for j in range(0, 10):
			if i != j:
				source_label = label[i]
				search_dir = search_root + "/" + source_label
				file_list = np.asarray(os.listdir(search_dir))[: 50]
				for k in range(int(iteration[i][j]), 50):
					print("i = {}\tj = {}\tk = {}".format(i, j, k))
					input_path = search_dir + "/" + file_list[k]
					target_label = label[j]
					ouput_path = "/home/leeanghsuan/Desktop/Speech_Command/" + source_label + "/" + target_label + "/" + str(success[i][j]) + ".wav"
					command="python3 generate_audio_v2.py --wavPath " + input_path + " --newWavPath " + ouput_path + " --target " + target_label
					start_time = time.time()
					exit_code = os.system(command)
					elapsed_time = elapsed_time + time.time() - start_time

					# KeyboardInterrupt
					if exit_code == 2:
						with open('/home/leeanghsuan/Desktop/Speech_Command/time.txt', "w") as f:
							f.write(str(elapsed_time))
						success.tofile('/home/leeanghsuan/Desktop/Speech_Command/success1.csv',sep=',',format='%d')
						iteration.tofile('/home/leeanghsuan/Desktop/Speech_Command/iteration1.csv',sep=',',format='%d')
						os.remove("/home/leeanghsuan/Desktop/Speech_Command/success.csv")
						os.remove("/home/leeanghsuan/Desktop/Speech_Command/iteration.csv")
						os.rename("/home/leeanghsuan/Desktop/Speech_Command/success1.csv", "/home/leeanghsuan/Desktop/Speech_Command/success.csv")
						os.rename("/home/leeanghsuan/Desktop/Speech_Command/iteration1.csv", "/home/leeanghsuan/Desktop/Speech_Command/iteration.csv")
						time.sleep(0.5)
						sys.exit(1)
					# success
					if os.path.isfile(ouput_path) != 0:
						success[i][j] += 1.0
					iteration[i][j] += 1.0
					# save from time to time
					if k == 9:
						with open('/home/leeanghsuan/Desktop/Speech_Command/time.txt', "w") as f:
							f.write(str(elapsed_time))
						success.tofile('/home/leeanghsuan/Desktop/Speech_Command/success1.csv',sep=',',format='%d')
						iteration.tofile('/home/leeanghsuan/Desktop/Speech_Command/iteration1.csv',sep=',',format='%d')
						os.remove("/home/leeanghsuan/Desktop/Speech_Command/success.csv")
						os.remove("/home/leeanghsuan/Desktop/Speech_Command/iteration.csv")
						os.rename("/home/leeanghsuan/Desktop/Speech_Command/success1.csv", "/home/leeanghsuan/Desktop/Speech_Command/success.csv")
						os.rename("/home/leeanghsuan/Desktop/Speech_Command/iteration1.csv", "/home/leeanghsuan/Desktop/Speech_Command/iteration.csv")
	with open('/home/leeanghsuan/Desktop/Speech_Command/time.txt', "w") as f:
		f.write(str(elapsed_time/4500))
	success.tofile('/home/leeanghsuan/Desktop/Speech_Command/success1.csv',sep=',',format='%d')
	iteration.tofile('/home/leeanghsuan/Desktop/Speech_Command/iteration1.csv',sep=',',format='%d')
	os.remove("/home/leeanghsuan/Desktop/Speech_Command/success.csv")
	os.remove("/home/leeanghsuan/Desktop/Speech_Command/iteration.csv")
	os.rename("/home/leeanghsuan/Desktop/Speech_Command/success1.csv", "/home/leeanghsuan/Desktop/Speech_Command/success.csv")
	os.rename("/home/leeanghsuan/Desktop/Speech_Command/iteration1.csv", "/home/leeanghsuan/Desktop/Speech_Command/iteration.csv")
	for i in range(10):
		for j in range(10):
			success[i][j] /= 10.0
	np.savetxt('/home/leeanghsuan/Desktop/Speech_Command/accuracy.txt', success, delimiter=',', fmt='%.2f')
	success.tofile('/home/leeanghsuan/Desktop/Speech_Command/accuracy1.csv',sep=',',format='%.2f')
	os.remove("/home/leeanghsuan/Desktop/Speech_Command/accuracy.csv")
	os.rename("/home/leeanghsuan/Desktop/Speech_Command/accuracy1.csv", "/home/leeanghsuan/Desktop/Speech_Command/accuracy.csv")

if __name__ == "__main__":
	main()