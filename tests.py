import ModeloVectorial
import TextNormalizer
import Utils
import os
from ModeloVectorial import *
from TextNormalizer import *
from UsualTools import *
maxLen = 1000000# Maxima mongitud de caracteres soportados por libro
exp = TextNormalizer()
llemmas = []
#----------------------------------------------------- Obtiene objetos tipo libros y los guardar -----------------------------------
#books = UsualTools.getLibros("./Libros de Goodreads")
#UsualTools.saveObject(books,"./Recursos/BooksList.json")
#-----------------------------------------------------
books = UsualTools.loadObject("./Recursos/BooksList.json")
exp.setVocabulary(UsualTools.loadObject("./Recursos/vocabulary.json"))
#----------------------------------------------------- Lemmatiza los libros y los guarda -------------------------------------------

for book in books:
	if (not (str(book.num)+".json" in os.listdir("./Recursos/lemmas/")) ):
		print("-----------------")
		print("Titulo:",book.nombre)
		book.texto = exp.delExtraInfoPG("./Libros de Goodreads/"+str(book.num)+".txt")
		book.texto = exp.deleteSpecialChars(book.texto)
		booktam = len(book.texto)
		lemmas = []
		if (len(book.texto) < maxLen):#1000000 es el numero maximo de caracteres soportada por cada procesamiento
			exp.setText(book.texto)
			lemmas= exp.lemmatize_delSW()
		else:#Si excede el numero de caracteres se divide en bloques y luego se guntan los lemmas
			print("Libro grande")
			i1 = 0
			i2 = 0

			prop = booktam // (maxLen -2)  # Cantidad de porciones de texto de tamano maximo
			porc = int((maxLen -2)  * (booktam / (maxLen -2)  - booktam // (maxLen -2) )) #sobrante de las porciones a procesar
			print(porc,"/")
			for i in range(prop):
				i2 = i2+(maxLen -2 )
				exp.setText(book.texto[i1:i2])
				lemmas= lemmas + exp.lemmatize_delSW()
				i1 = i2
			if(porc > 2):
				i2 = i2 + porc -1
				exp.setText(book.texto[i1:i2])
				lemmas= lemmas + exp.lemmatize_delSW()
		nle = len(lemmas)
		llemmas.append(lemmas)
		print("Los lemmas son	 " + str(nle))
		print("vocabulario es " + str(len(exp.vocabulary)))
		UsualTools.saveObject(lemmas,f"./Recursos/lemmas/{book.num}.json")
		UsualTools.saveObject(exp.vocabulary,"./Recursos/vocabulary.json")
#--------------------------------------------------------------------------------------------------------------------------------

#for l in os.listdir("./Recursos/lemmas/") :
#	llemmas.append(UsualTools.loadObject("./Recursos/lemmas/"+l))
#UsualTools.saveObject(exp.vocabulary,"./Recursos/vocabulary.json")
#mv = ModeloVectorial(llemmas,list(exp.vocabulary))
#suma  = 0
#UsualTools.saveObject(mv,"./Recursos/vectores.json")
#print(mv.vectors[0][9137])
#mv = UsualTools.loadObject("./Recursos/vectores.json")
#vectores = mv.vectors
#vsav = list(vectores)
#cont = 0

#mv.getIDF()
#print(mv.idf)
#vector=vectores[0]
#cont = 0
#for i in list(vector):
#	cont = cont + i
#print(cont)
#cont = 0

#print(len(llemmas[0]))
vcc = UsualTools.loadObject("./Recursos/vocabulary.json")

print("vocabulario es " + str(len(vcc)))