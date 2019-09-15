import tensorflow as tf
import numpy as np
from util import *
from attacker import Attacker

if __name__ == '__main__':
	flags = tf.flags
	flags.DEFINE_string("wavPath","0a2b400e_nohash_0.wav","the source wav path")
	flags.DEFINE_string("graphPath","my_frozen_graph.pb","the attacked Graph path")
	flags.DEFINE_string("newWavPath","new_version.wav","the new wav path")
	flags.DEFINE_string("labelPath","conv_labels.txt","label path")
	flags.DEFINE_string("target","yes","the target")
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

	with tf.Session() as sess:
		#newAudio = gen_attack(sess,data,label,outputTensor,targetLabel)
		attack = Attacker(sess,data,label,outputTensor,targetLabel)
		newAudio = attack.run()
		store_wav(FLAGS.newWavPath,newAudio)
