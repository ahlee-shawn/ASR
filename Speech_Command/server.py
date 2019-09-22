import sys
import socket
import os

if __name__ == '__main__':
	
	serverSocket = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
	host = socket.gethostname()
	portNumber = 8888

	serverSocket.bind((host, portNumber))
	serverSocket.listen(5)
	serverSocket.settimeout(None)

	targetArray = ["yes", "no", "up", "down", "left", "right", "on", "off", "stop", "go"]

	while True:
		clientSocket, addr = serverSocket.accept()    

		'''if os.path.exists("input.wav"):
			os.remove("input.wav")
		if os.path.exists("output.wav"):
			os.remove("output.wav")'''

		target = clientSocket.recv(1)
		print(target.decode())

		size = clientSocket.recv(5)
		print(size.decode())

		with open("input.wav", 'wb') as f:
			for i in range(int(size)):
				inputFile = clientSocket.recv(1)
				f.write(inputFile)

		command = "python3 generate_audio_v2.py --wavPath input.wav --newWavPath output.wav --target " + targetArray[int(target)]
		os.system(command)

		if os.path.exists("output.wav"):
			clientSocket.send("1".encode())
			with open("output.wav", 'rb') as f:
				outputFile = f.read()
			size = len(outputFile)
			clientSocket.send(str(size).encode())
			#print(size)
			#print(type(outputFile))
			for output in outputFile:
				'''print(type(output))
				print(output)
				print(bytes([output]))
				print(len(bytes([output])))'''
				clientSocket.send(bytes([output]))
		else:
			clientSocket.send("0".encode())