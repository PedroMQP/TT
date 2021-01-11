import sklearn
import Utils
import os
import ModeloVectorial
from ModeloVectorial import *
from Utils import *
from sklearn.metrics import precision_score

class Modelo:
	def __init__(self):
		self.model = None
		self.trainData = []
		self.trainLabels = []
		self.testData = []
		self.testLabels = []
		self.testResults = []
	 
	def configModelo(self):
		pass
	
	def setTrainData(self,ejemplos,etiquetas):
		self.trainData = ejemplos
		self.trainLabels = etiquetas

	def setTestData(self,ejemplos,etiquetas):
		self.testData = ejemplos
		self.testLabels = etiquetas

		
	def entrenar(self):
		self.model.fit(self.trainData, self.trainLabels)

	def metricas(self):
		cm = sklearn.metrics.confusion_matrix(self.testLabels,self.testResults,labels=[1,2,3])
		print("Reales/Preds  Malos   Regulares  Excelentes")
		print("Malos         ",cm[0][0],"      ",cm[0][1],"       ",cm[0][2],"   ")
		print("Regulares     ",cm[1][0],"     ",cm[1][1],"       ",cm[1][2],"   ")
		print("Excelentes    ",cm[2][0],"     ",cm[2][1],"       ",cm[2][2],"   ")
		precision = sklearn.metrics.precision_score(self.testLabels,self.testResults,average = None)
		p1 = sklearn.metrics.precision_score(self.testLabels,self.testResults,average = 'macro')
		p2 = sklearn.metrics.precision_score(self.testLabels,self.testResults,average = 'micro')
		p3 = sklearn.metrics.precision_score(self.testLabels,self.testResults,average = 'weighted')

		recall = sklearn.metrics.recall_score(self.testLabels,self.testResults,average = None)
		accuaracy = sklearn.metrics.accuracy_score(self.testLabels,self.testResults)

		specifity = (
					(cm[1][1]+cm[2][2])/(cm[1][1]+cm[2][2] +cm[1][0]+cm[2][0] )+
					(cm[0][0]+cm[2][2])/(cm[0][0]+cm[2][2] +cm[0][1]+cm[2][1] )+
					(cm[1][1]+cm[0][0])/(cm[1][1]+cm[0][0] +cm[0][2]+cm[1][2] )
	     			 )/3
		f1 = sklearn.metrics.f1_score(self.testLabels,self.testResults,average = None)
		return [precision,recall,accuaracy,specifity,f1]
	
	def probar(self):
		pass

	def graficarResultados(self):
		pass

