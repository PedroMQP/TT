import Preprocesamiento
import KNN
import NaiveBayes
import RegresionLogistica
import Libro
import Utils
import os

from Utils import *
from Preprocesamiento import *
from KNN import *
from NaiveBayes import *
from RegresionLogistica import *
from sklearn.model_selection import train_test_split
from sklearn.model_selection import cross_val_score

class Prototipo:
	def __init__(self):
		self.knn = KNN()
		self.nv = NaiveBayes()
		self.logreg = RegresionLogistica()
		self.libros = []
		self.prepro = Preprocesamiento()
		self.vectores = ModeloVectorial()
		self.prediccionesKNN = []
	
	def procesarTextos(self):
		vocabulary = set()
		if ("vocabulary.json" in os.listdir("./Recursos/")):
			vocabulary = Utils.loadObject("./Recursos/vocabulary.json")
		maxLen = 999999# Maxima mongitud de caracteres soportados por libro
		for lib in self.libros:
			lemmas = []
			if not( str(lib.num)+".json" in os.listdir("./Recursos/lemmas/")):	
				booktam = len(lib.texto)
				txt = self.prepro.deleteSpecialChars(lib.texto)
				if (booktam < maxLen):#1000000 es el numero maximo de caracteres soportada por cada procesamiento
					self.prepro.setText(txt)
					lemmas= self.prepro.lemmatize_delSW()
				else:
					i1 = 0
					i2 = 0

					prop = booktam // (maxLen -2)  # Cantidad de porciones de texto de tamano maximo
					porc = int((maxLen -2)  * (booktam / (maxLen -2)  - booktam // (maxLen -2) )) #sobrante de las porciones a procesar
					for i in range(prop):
						i2 = i2+(maxLen -2 )
						self.prepro.setText(txt[i1:i2])
						lemmas= lemmas + self.prepro.lemmatize_delSW()
						i1 = i2
					if(porc > 2):
						i2 = i2 + porc -1
						self.prepro.setText(txt[i1:i2])
						lemmas= lemmas + self.prepro.lemmatize_delSW()

				Utils.saveObject(lemmas,"./Recursos/lemmas/"+str(lib.num)+".json")
				
				vocabulary = vocabulary | set(lemmas) 
		Utils.saveObject(vocabulary,"./Recursos/vocabulary.json")

	def generarVectores(self):
		vocabulary = Utils.loadObject("./Recursos/vocabulary.json") 
		self.vectores.setVocabulary(vocabulary)
		for lib in self.libros:
			print("nuevo libro")
			lemmas = Utils.loadObject("./Recursos/lemmas/"+str(lib.num)+".json")
			self.vectores.addVector(lemmas,lib.estrellas )
		self.vectores.setTFIDF()
		Utils.saveObject(self.vectores,"./Recursos/ModeloVectorial.json")

	def seleccionarModelos(self):
		
		#Configuración de los parametros de los modelos 
		  #          weigths    kn   alg 
		knnConf = [["distance", 3,"brute"],
				   ["distance", 5,"brute"],
				   ["distance", 6,"brute"],
				   ["distance", 7,"brute"],
				   ["distance", 9,"brute"],
				   ["distance", 11,"brute"],

				   ["uniform", 3,"brute"],
				   ["uniform", 5,"brute"],
				   ["uniform", 6,"brute"],
				   ["uniform", 7,"brute"],
				   ["uniform", 9,"brute"],
				   ["uniform", 11,"brute"]			
		]
		nvConf = [0.05,0.1,0.2,0.3,0.4,0.5,0.6,0.7,0.8,0.9,0.99]
		#‘l1’, ‘l2’, ‘elasticnet’, ‘none',{‘newton-cg’, ‘lbfgs’, ‘liblinear’, ‘sag’, ‘saga’
					#penalty   solver 
		reglConf = [
					["lbfgs","none"],
					["lbfgs","l2"],

					["newton-cg","none"],
					["newton-cg","l2"],

					#["sag","none"],
					#["sag","l2"],

					["saga","none"],
					["saga","l1"],
					["saga","l2"],
					["saga","elasticnet"],

					["liblinear","l1"],
					["liblinear","l2"],

		]
		bestKNN = [[],0," "]
		bestNV = [[],0," "]
		bestLogReg = [[],0," "]
		self.vectores =  Utils.loadObject("./Recursos/ModeloVectorial.json")
		for k in range(1):

			if (k == 0):
				print("Pruebas con TFIDF")
			else:
				print("Pruebas con TF")
				self.vectores.setTF()

			m = []
			r = []
			e = []
			for i in range(len(self.vectores.vectors)):
				if (self.vectores.etiquetas[i] == 1):
					m.append(self.vectores.vectors[i])
				elif (self.vectores.etiquetas[i] == 2):
					r.append(self.vectores.vectors[i])
				elif (self.vectores.etiquetas[i] == 3):
					e.append(self.vectores.vectors[i])
			X = []
			X = m + r + e
			X_train, X_test, y_train, y_test = train_test_split(X, 
				[0]*len(m)	 + [1]*len(r) + [2]*len(e), test_size=0.3, random_state= 0)
			

			print("KNN")
			for c in knnConf:
				print("Pesos:",c[0],"kn:",c[1])
				self.knn.configModelo(c[0],c[1],c[2])
				cv = cross_val_score(self.knn.model, X_train, y_train, cv=5)
				print("Precision promedio",cv.mean())
				if cv.mean() > bestKNN[1]:
					tv = "tf"
					if k == 1:
						tv = "tf-idf"
					bestKNN = [c,cv.mean(),tv]
			
			print("Regresion logistica")
			for c in reglConf:
				print("Solver:",c[0],"penalty:",c[1])
				self.logreg.configModelo(c[0],c[1])
				cv = cross_val_score(self.logreg.model, X_train, y_train, cv=5)
				print("Precision promedio",cv.mean())
				if cv.mean() > bestKNN[1]:
					tv = "tf"
					if k == 1:
						tv = "tf-idf"
					bestLogReg = [c,cv.mean(),tv]

			print("Naive Bayes")
			for c in nvConf:
				print("alpha:",c)
				self.nv.configModelo(c)
				cv = cross_val_score(self.nv.model, X_train, y_train, cv=5)
				print("Precision promedio",cv.mean())
				if cv.mean() > bestNV[1]:
					tv = "tf"
					if k == 1:
						tv = "tf-idf"
					bestNV = [c,cv.mean(),tv]
		
		print("Los mejores parametros para KNN son:",bestKNN[0])
		print("con vectores tipo:",bestKNN[2])
		print("con una precision promedio de:",bestKNN[1])
		print("Los mejores parametros para Naive Bayes son:",bestNV[0])
		print("con vectores tipo:",bestNV[2])
		print("Con una precision promedio de:",bestNV[1])
		#print("Los mejores parametros para Regresion logistica son:",bestLogReg[0])
		#print("Con una precision promedio de:",bestLogReg[1])
		bestConfgs = []
		bestConfgs.append(bestKNN)
		bestConfgs.append(bestNV)
		bestConfgs.append(bestLogReg)
		Utils.saveObject(bestConfgs,"./Recursos/ModelConfigs.json")


	def entrenarModelos(self):
		configs = Utils.loadObject("./Recursos/ModelConfigs.json")
		print(configs)  
		knnConf = configs[0]
		nvConf = configs[1]
		lrConf = configs[2]
		self.vectores =  Utils.loadObject("./Recursos/ModeloVectorial.json")
		m = []
		r = []
		e = []
		for i in range(len(self.vectores.vectors)):
			if (self.vectores.etiquetas[i] == 1):
				m.append(self.vectores.vectors[i])
			elif (self.vectores.etiquetas[i] == 2):
				r.append(self.vectores.vectors[i])
			elif (self.vectores.etiquetas[i] == 3):
				e.append(self.vectores.vectors[i])
		X = []
		X = m + r + e
		X_train, X_test, y_train, y_test = train_test_split(X, 
				[0]*len(m)	 + [1]*len(r) + [2]*len(e), test_size=0.3, random_state= 0)
		print("Datos para entrenamiento")
		print("Libros malos:",y_train.count(0))
		print("Libros regulares:",y_train.count(1))
		print("Libros buenos:",y_train.count(2))	
		self.knn.setTrainData(X_train,y_train)
		self.nv.setTrainData(X_train,y_train)
		self.logreg.setTrainData(X_train,y_train)

		self.knn.setTestData(X_test,y_test)
		self.nv.setTestData(X_test,y_test)
		self.logreg.setTestData(X_test,y_test)

		#X = m[:25] + r[:25] + e[:25]
		#self.knn.setTrainData(X,[0]*25+[1]*25+[2]*25)
		#self.nv.setTrainData(X,[0]*25+[1]*25+[2]*25)

		#self.knn.setTestData(m[25:] + r[25:] + e[25:],[0]*len(m[25:]) + [1]*len(r[25:]) + [2]*len(e[25:]))
		#self.nv.setTestData(m[25:] + r[25:] + e[25:],[0]*len(m[25:]) + [1]*len(r[25:]) + [2]*len(e[25:]))

		self.knn.configModelo(knnConf[0][0],knnConf[0][1],knnConf[0][2])
		self.nv.configModelo(nvConf[0])
		self.logreg.configModelo(lrConf[0][0],lrConf[0][1])
		self.knn.entrenar()
		Utils.saveObject(self.knn,"./Recursos/KNN.json")
		self.nv.entrenar()
		Utils.saveObject(self.nv,"./Recursos/NaiveBayes.json")
		self.logreg.entrenar()
		Utils.saveObject(self.logreg,"./Recursos/LogReg.json")

	def generarResumen(self):
		self.knn = Utils.loadObject("./Recursos/KNN.json")
		self.nv = Utils.loadObject("./Recursos/NaiveBayes.json")
		self.logreg = Utils.loadObject("./Recursos/LogReg.json")

		print(len(self.nv.trainLabels))
		print("KNN")
		self.knn.probar()
		print("Precision",self.knn.precision())
		print("Naive Bayes")
		self.nv.probar()
		print("Precision",self.nv.precision())
		print("Regresion logistica")
		self.logreg.probar()
		print("Precision",self.logreg.precision())
	
	def predecirExterno(self,txt):
		lemmas= []
		print("Cargando texto")
		booktam = len(txt)
		if (booktam < 999999):
			self.prepro.setText(txt)
			lemmas= self.prepro.lemmatize_delSW()
		else:
			i1 = 0
			i2 = 0
			prop = booktam // (maxLen -2)  # Cantidad de porciones de texto de tamano maximo
			porc = int((maxLen -2)  * (booktam / (maxLen -2)  - booktam // (maxLen -2) )) #sobrante de las porciones a procesar
			for i in range(prop):
				i2 = i2+(maxLen -2 )
				self.prepro.setText(txt[i1:i2])
				lemmas= lemmas + self.prepro.lemmatize_delSW()
				i1 = i2
			if(porc > 2):
				i2 = i2 + porc -1
				self.prepro.setText(txt[i1:i2])
				lemmas= lemmas + self.prepro.lemmatize_delSW()
		vocabulary = Utils.loadObject("./Recursos/vocabulary.json") 
		print("Cargando vocabulario")
		mvec = ModeloVectorial()
		mvec.setVocabulary(vocabulary)
		print("Generando vector")
		mvec.addVector(lemmas)
		print("Cargando modelos")
		knn = Utils.loadObject("./Recursos/KNN.json")
		print("La clase es",knn.predecir(mvec.vectors[0]))
		return 0

	def graficarResultados(self):		
		pass
import os
prot = Prototipo()
#prot.libros = Utils.getLibros("./Libros de Goodreads")

#prot.procesarTextos()	
prot.generarVectores()

prot.seleccionarModelos()
prot.entrenarModelos()
prot.generarResumen()
ext = Utils.getText("./Libros de Goodreads/1.txt")
prot.predecirExterno(ext)