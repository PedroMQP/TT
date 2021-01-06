import jsonpickle #serializar objetos complejos en json 
import json
import pandas as pd
import Libro
from Libro import *
class Utils:
	
	@staticmethod
	def getText(fname):
		try:
			file = open(fname,"r",encoding = "utf8")
			txt = file.read()
		except:
			file = open(fname,"r",errors='ignore')
			txt = file.read()
		file.close()
		return txt
	@staticmethod
	def getTextLines(fname):
		try:
			file = open(fname,"r",encoding = "utf8")
			txt = file.readlines()
		except:
			file = open(fname,"r",errors='ignore')
			txt = file.readlines()
		file.close()
		return txt
	@staticmethod
	def saveObject(obj,fname):
		frozen = jsonpickle.encode(obj)
		with open(fname, "w") as file:
			file.write(frozen)
			file.close()

	@staticmethod
	def loadObject(fname):
		with open(fname) as file:
			frozen = file.read() #json.load(file)
			obj = jsonpickle.decode(frozen)
			return obj
	


	@staticmethod
	def getLibros(addr):#Retorna una lista de objetos tipo Libro de una direcci[on dada
		contador  = 0
		lst = []
		anterior = 600
		actual = 0
		dataset = pd.read_csv('Libros.csv')
		lista = dataset.iloc[:,[1,3]].values
		for lib in lista:
			cont = 0
			for i in range(1,156): #(f"Libros de Goodreads/{i}.txt")
				fname = addr+f"/{i}.txt"

				with open(fname, errors='ignore') as file: #open('Libros de Goodreads/'+str(i)+'.txt') as file:  #open(f'{'Libros de Goodreads/'}{i}{'.txt'}')
					x = file.read()
					actual = x.find(lib[0], 0, 500)
					if(actual!=-1):
						if( True):   
							anterior = actual
							contador+=1
							nuevoLibro = Libro(lib[0], lib[1], x,i)
							lst.append(nuevoLibro) 
		return lst
	def delExtraInfoPG(fname):#Elimina información extra de los libros del Projecto Gutenberg
		cad = ["*** END","***END"]# Cadenas que indica que el libro termino
		txtL = Utils.getTextLines(fname)
		txt= " "
		nl = len(txtL)
		ctxt = " "
		cut1 = 30#Indica cuantas lineas del inicio queremos quitar
		cut2 = nl-300 #Indica la linea apartir donde el contenido del libro termina
		if (nl >40): # Si tiene m[as de 10 lioneas es mas probable que sea del proyecto gutenberg
			cont = 200
			while(cont > 0):
				if(cad[0] in txtL[nl - cont -250] or cad[1] in txtL[nl - cont -250]):
					cut2 = nl - cont -250
					break
				cont = cont -1
			for i in range(cut1,cut2):
				txt = txt + txtL[i]
		else:
			txt = txtL[nl - 1]
		return txt.strip()