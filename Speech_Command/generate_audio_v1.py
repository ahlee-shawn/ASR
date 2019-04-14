import tensorflow as tf
import numpy as np

effect_bit = 3

def laod_graph(path):
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

def generate_first_population(data):
	new_bytes_array = bytearray(data)

	for i in range(44,len(data),2):
		if np.random.rand() < 0.0005:			
			noise = int(np.random.choice(range(0,2*effect_bit+1)))	
			new = (new_bytes_array[i+1] ^ noise).to_bytes(2,'little',signed=False)
			new_bytes_array[i+1] = new[0]
	
	return bytes(new_bytes_array)

def crossover(father,mother):
	father = bytearray(father)
	mother = bytearray(mother)

	for i in range(44,len(father)):
		if np.random.rand() < 0.5:
			mother[i] = father[i]

	return bytes(mother)

def mutation(data):
	new_bytes_array = bytearray(data)

	for i in range(44,len(data)):
		if np.random.rand() < 0.0005:
			noise = int(np.random.choice(range(0,2*effect_bit+1)))
			new = (new_bytes_array[i] ^ noise).to_bytes(2,'little',signed=False)
			new_bytes_array[i] = new[0]

	return bytes(new_bytes_array)

def inference(sess,data,output_tensor):
	predict_result, = sess.run(output_tensor, feed_dict = {"wav_data:0": data})
	return predict_result

def gen_attack(sess,data,label,output_tensor,target):
	max_iteration = 1000
	pop_size = 20
	elite_size = 3
	global effect_bit
	population = []
	for i in range(pop_size):
		population.append(generate_first_population(data))

	number_of_no_change = 0
	prev = 0

	for iter in range(max_iteration+1):
		score = []
		target_scores = []
		maximun_liklelihood = []
		for now_index in population:
			model_output = inference(sess,now_index,output_tensor)
			score.append(model_output)
			target_scores.append(model_output[target])
			predict_result = model_output.argsort()[-1]
			maximun_liklelihood.append(predict_result)

		top_k = np.array(target_scores).argsort()[-elite_size:][::-1]

		if iter > 0:
			if prev == maximun_liklelihood[top_k[0]]:
				number_of_no_change = number_of_no_change + 1
			else:
				number_of_no_change = 0
		if number_of_no_change > 20 and effect_bit < 8:
			effect_bit = effect_bit + 1

		prev = maximun_liklelihood[top_k[0]]

		print("target label index:  %d" %(target))
		print("The predict result of target top 1 score population:  %d" %(maximun_liklelihood[top_k[0]]))
		print("score of the predict result of target top 1 score population:  " ,end = '')
		print(score[top_k[0]][maximun_liklelihood[top_k[0]]])
		print("target top 1 score:  ",end = '')
		print(target_scores[top_k[0]])
		print('')

		for i in range(len(maximun_liklelihood)):
			if maximun_liklelihood[i] == target:
				print("success")
				return population[i]
		if iter == max_iteration:
			print("fail")
			return population[top_k[0]]

		sum = 0.0
		for i in target_scores:
			sum = sum + i

		selection_prob = []
		for i in target_scores:
			selection_prob.append(i/sum)

		next_generation = []
		for i in range(pop_size-elite_size):
			father = np.random.choice(pop_size, p = selection_prob)
			mother = np.random.choice(pop_size, p = selection_prob)
			child = crossover(population[father], population[mother])
			next_generation.append(child)

		for i in top_k:
			next_generation.append(population[i])

		for i in range(len(next_generation)):
			next_generation[i] = mutation(next_generation[i])

		population = next_generation



if __name__ == '__main__':
	flags = tf.flags
	flags.DEFINE_string("wav_path","0a2b400e_nohash_0.wav","Wav path")
	flags.DEFINE_string("graph_path","my_frozen_graph.pb","Graph path")
	flags.DEFINE_string("new_wav_path","new.wav","the new wav path")
	flags.DEFINE_string("label_path","conv_labels.txt","label path")
	flags.DEFINE_string("target","yes","the target")
	FLAGS = flags.FLAGS

	data = load_wav(FLAGS.wav_path)
	label = load_labels(FLAGS.label_path)
	
	target_label = 0
	for i in range(len(label)):
		if label[i] == FLAGS.target:
			target_label = i
	
	laod_graph(FLAGS.graph_path)
	
	output_tensor = tf.get_default_graph().get_tensor_by_name("labels_softmax:0")

	with tf.Session() as sess:
		new_audio = gen_attack(sess,data,label,output_tensor,target_label)
		store_wav(FLAGS.new_wav_path,new_audio)


	
