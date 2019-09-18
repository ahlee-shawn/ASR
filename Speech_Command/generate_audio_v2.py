import tensorflow as tf
import numpy as np
import multiprocessing as mp
from util import *
from attacker import Attacker
import os

def worker(dataQueue, dataQueueLock, mtDNAQueue, mtDNAQueueLock, i, quit, foundit,data,label,outputTensor,targetLabel,newWavPath):
	print("%d started" % i)
	config = tf.ConfigProto()
	config.gpu_options.allow_growth = True
	with tf.Session(config=config) as sess:
		#newAudio = gen_attack(sess,data,label,outputTensor,targetLabel)
		attack = Attacker(sess, data, label, outputTensor, targetLabel, os.getpid(), i)
		newAudio = attack.run(dataQueue, dataQueueLock, mtDNAQueue, mtDNAQueueLock, quit)
		foundit.set()
		if newAudio != None:
			store_wav(newWavPath,newAudio)
	print("%d is done" % i)


if __name__ == '__main__':
	flags = tf.flags
	flags.DEFINE_string("wavPath", "0a2b400e_nohash_0.wav", "the source wav path")
	flags.DEFINE_string("graphPath", "my_frozen_graph.pb", "the attacked Graph path")
	flags.DEFINE_string("newWavPath", "new_version_para.wav", "the new wav path")
	flags.DEFINE_string("labelPath", "conv_labels.txt", "label path")
	flags.DEFINE_string("target", "yes", "the target")
	FLAGS = flags.FLAGS

	data = load_wav(FLAGS.wavPath)
	label = load_labels(FLAGS.labelPath)
	
	try:
		targetLabel = label.index(FLAGS.target)
	except:
		print("ERROR: The attack target is not in the label list")

	#import graph
	load_graph(FLAGS.graphPath)
	
	outputTensor = tf.get_default_graph().get_tensor_by_name("labels_softmax:0")

	quit = mp.Event()
	foundit = mp.Event()
	dataQueue = mp.Queue()
	dataQueueLock = mp.Lock()
	mtDNAQueue = mp.Queue()
	mtDNAQueueLock = mp.Lock()

	all_processes = []
	for i in range(mp.cpu_count()): #for i in range(mp.cpu_count()):
		p = mp.Process(target=worker, args=(dataQueue, dataQueueLock, mtDNAQueue, mtDNAQueueLock, i, quit, foundit, data, label, outputTensor, targetLabel, FLAGS.newWavPath))
		p.start()
		all_processes.append(p)
	foundit.wait()
	quit.set()
	for process in all_processes: 
		process.terminate() 
