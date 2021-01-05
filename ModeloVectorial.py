import numpy as np #Se utiliza numpy para crear arreglos ya que estos son más rapidos que las listas
import math
class ModeloVectorial:

	#El vocabulario debe ser una lista en el orden que se desee que esten las caracteristicas en los vectores
	def __init__(self,lists = None ,vocabulary = None,ldata = None):# Pasamos como parametro una lista de datos para hacer corresponder cada uno con cada vector
		self.idf = []
		self.vectors = []
		self.etiquetas = []
		self.tfidf = 0
		self.vocabulary = {}#Diccionario con lemmas como llaves y dimension com valor
		self.tamVoc = 0
	#Inicializamos el diccionario con las unidades del vocabulario como clave y su ubicación en el vector como valor
		if not(vocabulary is None):
			self.setVocabulary(vocabulary)
			if not(lists is None):
				if not(ldata is None):
					for i in range(len(lists)):
						self.addVector(lists[i],ldata[i])#Los elementos de ldata deben estar en orden respecto a la lista de lemmas que les corresponde
				else:
					for i in range(len(lists)):
						self.addVector(lists[i])


	def setVocabulary(self,vocabulary):
		voc = list(vocabulary)
		voc.sort()

		self.tamVoc = len(voc)
		for i in range(self.tamVoc):
			self.vocabulary[voc[i]] = i

	def setTFIDF(self):

		if(self.tfidf == 0 ):# Vuelve los vectores tf del modelo vectorial a tfidf
			self.getIDF()
			self.tfidf = 1
			for i in range(len(self.vectors)):
				for j in range(self.tamVoc):
					self.vectors[i][j] =self.vectors[i][j]*self.idf[j] 
			print("Los vectores ahora usan el TF-IDF")
		else:
			print("Los vectores ya usan el TF-IDF")

	def setTF(self):
		if(self.tfidf == 1 ):# Vuelve los vectores tf del modelo vectorial a tfidf
			for i in range(len(self.vectors)):
				for j in range(self.tamVoc):

						self.vectors[i][j] =self.vectors[i][j]/self.idf[j] 
			print("Los vectores ahora usan el TF")
		else:
			print("Los vectores ya usan el TF")

	def addVocab(self,newVocabulary):
		newF = list(set(newVocabulary) - set(self.vocabulary.keys())) 
		if(len(newF) > 0):
			cont = 0
			for feature in newF: 
				self.vocabulary[feature] = self.tamVoc + cont 
				cont = cont + 1
			self.tamVoc = len(self.vocabulary)
		return newF

	def addVector(self,t,etiqueta = None):# Agregamos un nuevo vector nuestro espacio vectorial
		tf = self.getTF(t)
		if(self.tfidf):
			self.getIDF()
			for i in range(len(tf)):
				tf[i] = tf[i]*idf[i]
		nv = np.array([tf])
		if not(etiqueta is None):
			self.etiquetas.append(etiqueta)# Agremamos la etiqueta correspondiente
				
		if self.vectors == []:#El espacio vectorial esta vacio
			self.vectors = nv
		else:
			self.vectors = np.concatenate((self.vectors,nv),axis = 0)
	
	
	def getTF(self,tokens):#Formamos la lista con el term frequency(tf) de cada token de la lista
		vector = [0]*self.tamVoc
		for lemma in tokens:
			if lemma in self.vocabulary.keys():
				vector[self.vocabulary[lemma]] = vector[self.vocabulary[lemma]] + 1
		return vector

	def getIDF(self):#Formamos la lista con el inverse document frequency (idf) de cada lemma
		N = len(self.vectors) # Numero de documentos o filas en el espacio vectorial
		self.idf = [0]*self.tamVoc
		for i in range(N): 
			vector = self.vectors[i]
			for lemma in self.vocabulary.keys():#Contamos por cada lema si dicho lemma esta al menos una vez en cada documento 
				index = self.vocabulary[lemma] 
				if(vector[index] > 0):
					self.idf[index] = self.idf[index] + 1
		for i in range(self.tamVoc):
			self.idf[i] = math.log(N/(1 + self.idf[i])) + 1


	

	def addFeatures(self,lfeatures):# Agrega una nueva caracteristica al final de de cada vector 
		lf = self.addVocab(lfeatures)#Agregamos las nuevas caracteristicas al vocabulario
		for feature in lf:
			n.vectors = np.insert(self.vectors, self.vectors.shape[1], np.array([0]*len(self.vectors)), 1)
		self.tamVoc = len(self.vocabulary)

	def delFeature(self,feature):
		self.vectors = np.delete(self.vectors,self.vocabulary[feature] ,axis = 1 )
		tam = len(self.vocabulary)
		for v in self.vocabulary.items():#Actualizamos los indices correspondientes a cada lemma
			if(v[1] >  self.vocabulary[feature]):
				self.vocabulary[v[0]] = self.vocabulary[v[0]] - 1
		self.vocabulary.pop(feature)
		self.tamVoc = len(self.vocabulary)

