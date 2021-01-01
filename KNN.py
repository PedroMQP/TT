import sklearn
import UsualTools
import os
import ModeloVectorial
from ModeloVectorial import *
from UsualTools import *
from sklearn.neighbors import KNeighborsClassifier

class KNN:
	w = 'distance'
	kneighbors = 3

	books = UsualTools.loadObject("./Recursos/BooksList.json")
	#UsualTools.saveObject(books,"./Recursos/BooksList.json")
	#books = UsualTools.loadObject("./Recursos/BooksList.json")
	bad = []
	regular = []
	excellent = []
	llemmas = []
	#ldatos =  sorted(os.listdir("./Recursos/lemmas/"))
	#for l in ldatos:
	#	llemmas.append(UsualTools.loadObject("./Recursos/lemmas/"+l))
	#vocabulary = UsualTools.loadObject("./Recursos/vocabulary.json")
	#ldatos.sort()
	#mv = ModeloVectorial(llemmas,list(vocabulary),ldatos)
	#UsualTools.saveObject(mv,"./Recursos/vectores.json")
	mv = UsualTools.loadObject("./Recursos/vectores.json")
	for book in books:
		if book.estrellas == 1:
			bad.append(mv.vectors[mv.indexC[str(book.num)+".json"] - 1])
		elif book.estrellas == 2:
			regular.append(mv.vectors[mv.indexC[str(book.num)+".json"] - 1])
		elif book.estrellas == 3:
			excellent.append(mv.vectors[mv.indexC[str(book.num)+".json"] - 1])
	print("Neighbors:",kneighbors,"Weights",w)
	print('BAD:',len(bad))
	print('REGULAR:',len(regular))
	print('EXCELLENT:',len(excellent))
	X = []
	X = X + bad[:20]
	X = X + regular[:20]
	X = X + excellent[:20]
	y = [1]*20 + [2]*20 + [3]*20
	print(len(X),len(y))
	neigh = KNeighborsClassifier(n_neighbors=kneighbors,weights=w,n_jobs = -1)
	neigh.fit(X, y)
	UsualTools.saveObject(neigh,"./Recursos/KNNModel.json")
	print("-----------------------Libros malos-------------------")
	V = 0
	F = 0
	F2 = 0
	F3 = 0
	predics = []
	for i in range(21,len(bad)):
		pred = neigh.predict([bad[i]])
		predics.append(pred)
		if (pred == 1):
			V = V +1
		else:
			F = F + 1
			if (2 == pred):
				F2 = F2 + 1
			else:
				F3 = F3 + 1
	print("Verdaderos",V)
	print("Falsos",F)
	print("   Regulares:",F2)
	print("   Excelentes:",F3)
	print("-----------------------Libros regulares-------------------")
	V = 0
	F = 0	
	F1 = 0
	F3 = 0
	for i in range(21,len(regular)):
		pred = neigh.predict([regular[i]])
		print(neigh.predict_proba([regular[i]]))

		predics.append(pred)
		if (pred == 2):
			V = V +1
		else:
			F = F + 1
			if (1 == pred):
				F1 = F1 + 1
			else:
				F3 = F3 + 1
	print("Verdaderos",V)
	print("Falsos",F)
	print("   Malos:",F1)
	print("   Excelentes:",F3)
	print("-----------------------Libros excelentes-------------------")
	V = 0
	F = 0
	F2 = 0
	F1 = 0	
	for i in range(21,len(excellent)):
		pred = neigh.predict([excellent[i]])

		predics.append(pred)
		if (pred == 3):
			V = V +1
		else:
			F = F + 1
			if (1 == pred):
				F1 = F1 + 1
			else:
				F2 = F2 + 1
	print("Verdaderos",V)
	print("Falsos",F)
	print("   Malos:",F1)
	print("   Regulares:",F2)
	print("-----------------------------------------------------")
	print("Matriz de confusión")
	true_labels = [0]*len(bad[21:])+[1]*len(regular[21:]) + [2]*len(excellent[21:])
	cm = sklearn.metrics.confusion_matrix(true_labels,predics[:],labels=[0,1,2])# Matriz de confusión de las pruebas
	print("Reales/Preds  Malos   Regulares  Excelentes")
	print("Malos         ",cm[0][0],"      ",cm[0][1],"       ",cm[0][2],"   ")
	print("Regualres     ",cm[1][0],"     ",cm[1][1],"       ",cm[1][2],"   ")
	print("Excelentes    ",cm[2][0],"     ",cm[2][1],"       ",cm[2][2],"   ")
	print("-----------------------------------------------------")
	precision = sklearn.metrics.accuracy_score(true_labels,predics[:])
	print("La precision es ", precision)
	recall = sklearn.metrics.recall_score(true_labels,predics[:],average = 'micro')
	print("El recall es: ",recall)# La capacidad del modelo de encontrar ejemplos positivos

KNN = KNN()
