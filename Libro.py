class Libro:
	def __init__(self, nombre, calif, texto, num):
		self.num = num
		self.calificacion = calif
		self.texto = texto
		self.nombre = nombre
		self.a = 5
		self.b = 4.0
		self.c = 3.85
		self.d = 3.6
		self.e = 3.5
		self.f = 0
		self.estrellas = self.generaEstrellas(calif)

	def generaEstrellas(self,calif):
    
		if ((calif <= self.a) and (calif >= self.b)):
			return 3    
		elif ((calif <= self.c) and (calif >= self.d)):
			return 2
		elif ((calif <= self.e) and (calif >= self.f)):
			return 1  
		else:
			return -1
	def configEstrellas(e1,e2,r1,r2,m1,m2):
		
		self.a = e1
		self.b = e2
		self.c = r1
		self.d = r2
		self.e = m1
		self.f = m2
		self.estrellas = self.generaEstrellas(calif)

		