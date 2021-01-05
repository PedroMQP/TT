from sklearn.naive_bayes import MultinomialNB
import Modelo
from Modelo import *
class NaiveBayes(Modelo):
	def __init__(self):
		Modelo.__init__(self)
		self.alpha = 0
		self.model = MultinomialNB(alpha = self.alpha)

	def configModelo(self,a):
		self.model = MultinomialNB(alpha = a)		

	def probar(self):
		self.testResults = self.model.predict(self.testData)
	def predecir(self,vec):
		return self.model.predict([vec])