import tensorflow as tf
import numpy as np
import random

class Attacker():
	def __init__(self, sess, data, label, outputTensor, targetLabel, pId, processorNumber, effectBit = 3):
		self.sess = sess
		self.data = data
		self.label = label
		self.outputTensor = outputTensor
		self.targetLabel = targetLabel
		self.effectBit = effectBit
		self.processorNumber = processorNumber
		np.random.seed(pId)

	def generate_first_population(self):
		newBytesArray = bytearray(self.data)
		#plus random noise for every two bytes when np.random.rand() < 0.0005, 44 is the header length of wav file
		for i in range(44,len(newBytesArray),2):
			if np.random.rand() < 0.0005:
				noise = int(np.random.choice(range(0, 2**self.effectBit)))
				effectPart = newBytesArray[i+1] % (2**self.effectBit)
				plusNoise = effectPart ^ noise
				newBytesArray[i+1] = newBytesArray[i+1] - effectPart + plusNoise
				
		return bytes(newBytesArray)
	    
	def _crossover(self, father, mother):
		father = bytearray(father)
		mother = bytearray(mother)
		#let mother's byte = father's byte at 50% #now: random crossoverRate
		for i in range(44,len(father)):
			if np.random.rand() < self.crossoverRate:
				mother[i] = father[i]

		return bytes(mother)

	def crossover(self, selectionProb):
		nextGeneration = []
		nextGenerationMtDNA = []
		i = 0
		while(i < self.populationSize-self.eliteSize):
			father_idx = np.random.choice(self.populationSize, p = selectionProb)
			mother_idx = np.random.choice(self.populationSize, p = selectionProb)
			if self.mtDNA[father_idx] != self.mtDNA[mother_idx]:
				child = self._crossover(self.population[father_idx], self.population[mother_idx])
				nextGeneration.append(child)
				nextGenerationMtDNA.append(self.mtDNA[mother_idx])
				i += 1

		return nextGeneration, nextGenerationMtDNA
		

	def _mutation(self, currentOffspring):
		newBytesArray = bytearray(currentOffspring)
		#plus random noise for every two bytes when np.random.rand() < 0.0005, 44 is the header length of wav file
		for i in range(44,len(newBytesArray),2):
			if np.random.rand() < self.mutationRate:
				noise = int(np.random.choice(range(0, 2**self.effectBit)))
				effectPart = newBytesArray[i+1] % (2**self.effectBit)
				plusNoise = effectPart ^ noise
				newBytesArray[i+1] = newBytesArray[i+1] - effectPart + plusNoise
	            
		return bytes(newBytesArray)

	def mutation(self, nextGeneration):
		for i in range(len(nextGeneration)):
			nextGeneration[i] = self._mutation(nextGeneration[i])

		return nextGeneration

	def inference(self, currentOffspring):
		#input into model to get score
		predictResult = self.sess.run(self.outputTensor, feed_dict = {"wav_data:0": currentOffspring})
		return predictResult[0]

	def calaulate_fitness(self):
		self.score = []
		self.targetScore = []
		self.predictResult = []

		for offspring in self.population:
			#get the fitness score
			modelOutput = self.inference(offspring)
			self.score.append(modelOutput)
			self.targetScore.append(modelOutput[self.targetLabel])

			highestScoreIndex = modelOutput.argsort()[-1]
			self.predictResult.append(highestScoreIndex)


	#if the solution space is not enough
	def check_stuck(self, iteration):
		if iteration > 0 and self.resultOfPrevBest == self.predictResult[self.eliteSet[0]]:
			self.numberNoChange = self.numberNoChange + 1
		else:
			self.numberNoChange = 0

		if self.numberNoChange > 100:
			self.effectBit = self.effectBit + 1
			self.numberNoChange = 0

	def check_success(self, iteration):
		currentBest = self.eliteSet[0]
		resultOfCurrentBest = self.predictResult[currentBest]
		if resultOfCurrentBest == self.targetLabel:
			print("success")
			return self.population[currentBest],0
		elif iteration == self.maxIteration:
			print("fail")
			return self.population[currentBest],1
		else:
			return self.population[currentBest],2

	def calculate_selection_prob(self):
		selectionProb = self.targetScore/np.sum(self.targetScore)	
		return selectionProb

	def initialize_mtDNA(self):
		temp = []
		for i in range(self.processorNumber * 100, self.processorNumber * 100 + self.populationSize):
			temp.append(i)
		return temp

	def run(self, quit):
		self.maxIteration = 1000
		self.populationSize = random.randint(20,50) #randomize population size
		self.eliteSize = 3
		self.mutationRate = np.random.normal(loc=0.0005, scale=0.0001, size=1)
		self.crossoverRate = np.random.normal(loc=0.5, scale=0.1, size=1)

		self.population = []
		for i in range(self.populationSize):
			self.population.append(self.generate_first_population())

		self.mtDNA = self.initialize_mtDNA()

		self.numberNoChange = 0
		self.resultOfPrevBest = 0

		for iteration in range(self.maxIteration+1):
			#One process has found the answer
			if quit.is_set():
				break

			self.calaulate_fitness()

			#get the first k population which has highest target score, k = elite size  
			self.eliteSet = np.array(self.targetScore).argsort()[-self.eliteSize:][::-1]
			self.check_stuck(iteration)
			#update Prev best
			self.resultOfPrevBest = self.predictResult[self.eliteSet[0]]

			self.print_stat()

			result, stat = self.check_success(iteration)
			if stat == 0 or stat == 1:
				return result

			selectionProb = self.calculate_selection_prob()
			nextGeneration, nextGenerationMtDNA = self.crossover(selectionProb)
			for i in self.eliteSet:
				nextGenerationMtDNA = nextGenerationMtDNA + [self.mtDNA[i]]
				nextGeneration = nextGeneration + [self.population[i]]

			nextGeneration = self.mutation(nextGeneration)
			self.population = nextGeneration
			self.mtDNA = nextGenerationMtDNA
			if iteration % 100 == 0:
				self.mtDNA = self.initialize_mtDNA()


	def print_stat(self):
		currentBest = self.eliteSet[0]
		resultOfCurrentBest = self.predictResult[currentBest]
		print("Processor Number: %s" %(self.processorNumber))
		print("Population Size: %s" %(self.populationSize))
		print("target label:  %s" %(self.label[self.targetLabel]))
		print("The predict result of target top 1 score population:  %s" %(self.label[resultOfCurrentBest]))
		print("score of the predict result of target top 1 score population:  " ,end = '')
		print(self.score[currentBest][resultOfCurrentBest])
		print("target top 1 score:  ",end = '')
		print(self.targetScore[currentBest])
		print('')
