
from sklearn.neighbors import KNeighborsClassifier
import Modelo
from Modelo import *
class KNN(Modelo):
	
	def __init__(self):
		Modelo.__init__(self)
		self.w = "distance"
		self.kneighbors = 3
		self.algorithm = "brute"
		self.model = KNeighborsClassifier(n_neighbors=self.kneighbors,weights=self.w,algorithm = self.algorithm ,n_jobs = -1)
	
	def configModelo(self,w,k,alg):
		self.model = KNeighborsClassifier(n_neighbors=k,weights=w,algorithm = alg ,n_jobs = -1)		

	def probar(self):
		self.testResults = []
		for i in range(len(self.testLabels)):
			self.testResults.append(self.model.predict([self.testData[i]])[0])
	def predecir(self,vec):
		return self.model.predict([vec])

