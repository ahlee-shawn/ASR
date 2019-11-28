import tensorflow as tf
import numpy as np
import sys

def load_graph(path):
	graph_def = tf.GraphDef()
	with open(path, 'rb') as f:
		graph_def.ParseFromString(f.read())
		tf.import_graph_def(graph_def,name = '')

def load_wav(path):
	with open(path,'rb') as f:
		data = f.read()
	return data

def inference(sess, data, outputTensor):
	#input into model to get score
	result, = sess.run(outputTensor, feed_dict = {"wav_data:0": data})
	return result

if __name__ == '__main__':
	labelArray = np.array(["_silence_", "_unknown_", "yes", "no", "up", "down", "left", "right", "on", "off", "stop", "go"])
	flags = tf.flags
	flags.DEFINE_string("wavPath","0a2b400e_nohash_0.wav","the source wav path")
	flags.DEFINE_string("graphPath","my_frozen_graph.pb","the attacked Graph path")
	flags.DEFINE_string("label","yes","the label")
	FLAGS = flags.FLAGS

	data = load_wav(FLAGS.wavPath)

	#import graph
	load_graph(FLAGS.graphPath)
	
	outputTensor = tf.get_default_graph().get_tensor_by_name("labels_softmax:0")

	with tf.Session() as sess:
		result = inference(sess, data, outputTensor)
		index = np.argmax(result)
		f = open("inferenceResult.txt", "a+")
		f.write("{0} with label '{1}' -> Inference Result: {2}\n".format(FLAGS.wavPath, FLAGS.label, labelArray[index]))
		f.close()