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

	def precision(self):
		cm = sklearn.metrics.confusion_matrix(self.testLabels,self.testResults,labels=[0,1,2])
		print("Reales/Preds  Malos   Regulares  Excelentes")
		print("Malos         ",cm[0][0],"      ",cm[0][1],"       ",cm[0][2],"   ")
		print("Regualres     ",cm[1][0],"     ",cm[1][1],"       ",cm[1][2],"   ")
		print("Excelentes    ",cm[2][0],"     ",cm[2][1],"       ",cm[2][2],"   ")
		print("-----------------------------------------------------")
		precision = sklearn.metrics.precision_score(self.testLabels,self.testResults,average = None)
		return precision
	
	def probar(self):
		pass

	def graficarResultados(self):
		pass

