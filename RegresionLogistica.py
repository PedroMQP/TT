from sklearn.linear_model import LogisticRegression
import Modelo
from Modelo import *
class RegresionLogistica(Modelo):
	def __init__(self):
		Modelo.__init__(self)
		self.solver = "liblinear"
		self.penalty = "l1"
		self.max_iter = 5500
		self.model = LogisticRegression(solver = self.solver,penalty = self.penalty,max_iter = self.max_iter,n_jobs = -1)

	def configModelo(self,s,p):
		self.model = LogisticRegression(solver = s,penalty = p,max_iter = self.max_iter)		

	def probar(self):
		self.testResults = self.model.predict(self.testData)
	def predecir(self,vec):
		return self.model.predict([vec])
