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
	def separarConjuntos(self):#separa los ids en sus categoria  
		m = []
		r = []
		b = []
		for lib in self.libros:
			if(lib.estrellas == 1):
				m.append(lib.num)
			elif(lib.estrellas == 2):
				r.append(lib.num)
			elif(lib.estrellas == 3):
				b.append(lib.num)
		return [m,r,b]


	def procesarTextos(self):
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
				
	def generarVectores(self):
		m,r,e = self.separarConjuntos()
		vocabulary = set()
		for i in range(25):
			lemmas = Utils.loadObject("./Recursos/lemmas/"+str(m[i])+".json")
			vocabulary = vocabulary | set(lemmas)
			lemmas = Utils.loadObject("./Recursos/lemmas/"+str(r[i])+".json")
			vocabulary = vocabulary | set(lemmas)
			lemmas = Utils.loadObject("./Recursos/lemmas/"+str(e[i])+".json")
			vocabulary = vocabulary | set(lemmas)
		print("Vocabulario de tamaño",len(vocabulary))
		self.vectores.setVocabulary(vocabulary)
		Utils.saveObject(vocabulary,"./Recursos/vocabulary.json") 
		for i in range(25):
			lemmas = Utils.loadObject("./Recursos/lemmas/"+str(m[i])+".json")
			self.vectores.addVector(lemmas,1)
			lemmas = Utils.loadObject("./Recursos/lemmas/"+str(r[i])+".json")
			self.vectores.addVector(lemmas,2)
			lemmas = Utils.loadObject("./Recursos/lemmas/"+str(e[i])+".json")
			self.vectores.addVector(lemmas,3)
		self.vectores.setTFIDF()
		Utils.saveObject(self.vectores,"./Recursos/ModeloVectorial.json")

	def seleccionarModelos(self):
		self.vectores =  Utils.loadObject("./Recursos/ModeloVectorial.json")
		
		X_train = self.vectores.vectors
		y_train = []
		for v in range(len(X_train)):
			y_train.append(self.vectores.etiquetas[v])

		
		
		#Configuración de los parametros de los modelos 
		  #          weigths    kn   alg 
		knnConf = [

				   ["distance", 2,"brute"],			
				   ["distance", 3,"brute"],
				   ["distance", 4,"brute"],
				   ["distance", 5,"brute"],
				   ["distance", 6,"brute"],
				   ["distance", 7,"brute"],
				   ["distance", 9,"brute"],
				   ["distance", 11,"brute"],

				   ["uniform", 2,"brute"],			
				   ["uniform", 3,"brute"],
				   ["uniform", 4,"brute"],
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

					#["saga","none"],
					#["saga","l1"],
					#["saga","l2"],
					#["saga","elasticnet"],

					["liblinear","l1"],
					["liblinear","l2"],

		]
		bestKNN = [[],0," "]
		bestNV = [[],0," "]
		bestLogReg = [[],0," "]
		for k in range(2):
			if (k == 0):
				print("Pruebas con TFIDF")
			else:
				print("Pruebas con TF")
				self.vectores.setTF()
				X_train = self.vectores.vectors

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
				if cv.mean() > bestLogReg[1]:
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
					if k == 0:
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


		vocabulary = Utils.loadObject("./Recursos/vocabulary.json")
		mvTest = ModeloVectorial()
		mvTest.setVocabulary(vocabulary)
		m,r,e = self.separarConjuntos()
		y_test = []
		y_train = []
		X_train = self.vectores.vectors
		for i in range(25,len(m)):
			lemmas = Utils.loadObject("./Recursos/lemmas/"+str(m[i])+".json")
			mvTest.addVector(lemmas,1)
			y_test.append(1)
		for i in range(25,len(r)):	
			lemmas = Utils.loadObject("./Recursos/lemmas/"+str(r[i])+".json")
			mvTest.addVector(lemmas,2)
			y_test.append(2)
		for i in range(25,len(e)):	
			lemmas = Utils.loadObject("./Recursos/lemmas/"+str(e[i])+".json")
			mvTest.addVector(lemmas,3)
			y_test.append(3)

		self.vectores =  Utils.loadObject("./Recursos/ModeloVectorial.json")
		X_train = self.vectores.vectors
		y_train = []
		for v in range(len(X_train)):
			y_train.append(self.vectores.etiquetas[v])
		print("Datos para entrenamiento")
		print("Libros malos:",y_train.count(1))
		print("Libros regulares:",y_train.count(2))
		print("Libros buenos:",y_train.count(3))	
		self.knn.setTrainData(X_train,y_train)
		self.nv.setTrainData(X_train,y_train)
		self.logreg.setTrainData(X_train,y_train)

		if (configs[0][-1] == 'tf'):
			mvTest.setTF()
		elif (configs[0][-1] == 'tf-idf'):
			mvTest.setTFIDF()
		X_test = mvTest.vectors

		self.knn.setTestData(X_test,y_test)
		if (configs[1][-1] == 'tf'):
			mvTest.setTF()
		elif (configs[1][-1] == 'tf-idf'):
			mvTest.setTFIDF()
		X_test = mvTest.vectors
		
		self.nv.setTestData(X_test,y_test)
		if (configs[2][-1] == 'tf'):
			mvTest.setTF()
		elif (configs[2][-1] == 'tf-idf'):
			mvTest.setTFIDF()
		X_test = mvTest.vectors
		
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
		print("Datos de prueba")
		print("Libros malos:",self.nv.testLabels.count(1))
		print("Libros regulares:",self.nv.testLabels.count(2))
		print("Libros excelentes:",self.nv.testLabels.count(3))
		print("KNN")
		self.knn.probar()
		m = self.knn.metricas()
		print("Precision:",m[0])
		print("Recall(Exhaustividad):",m[1])
		print("Exactitud:",m[2])
		print("Especifidad:",m[3])
		print("F1:",m[4])
		print("-----------------------------------------------------")

		print("Naive Bayes")
		self.nv.probar()
		m = self.nv.metricas()
		print("Precision:",m[0])
		print("Recall(Exhaustividad):",m[1])
		print("Exactitud:",m[2])
		print("Especifidad:",m[3])
		print("F1:",m[4])

		print("-----------------------------------------------------")
		
		print("Regresion logistica")
		self.logreg.probar()
		m = self.logreg.metricas()
		print("Precision:",m[0])
		print("Recall(Exhaustividad):",m[1])
		print("Exactitud:",m[2])
		print("Especifidad:",m[3])
		print("F1:",m[4])
		print("-----------------------------------------------------")
	
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
prot.libros = Utils.getLibros("./Libros de Goodreads")

#prot.procesarTextos()	
#vc = Utils.loadObject("./Recursos/vocabulary.json")
#print("tam",len(vc))
prot.generarVectores()
prot.seleccionarModelos()
prot.entrenarModelos()
prot.generarResumen()
ext = Utils.getText("./Libros de Goodreads/1.txt")
prot.predecirExterno(ext)