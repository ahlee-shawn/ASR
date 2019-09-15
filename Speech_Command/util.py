import tensorflow as tf
import numpy as np

def load_graph(path):
	graph_def = tf.GraphDef()
	with open(path, 'rb') as f:
		graph_def.ParseFromString(f.read())
		tf.import_graph_def(graph_def,name = '')
			
def load_labels(path):
	"""Read in labels, one label per line."""
	return [line.rstrip() for line in tf.gfile.GFile(path)]

def load_wav(path):
	with open(path,'rb') as f:
		data = f.read()
	return data

def store_wav(path,data):
	with open(path,'wb') as f:
		new_data = bytes(data)
		f.write(new_data)
