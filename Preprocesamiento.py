import spacy
import nltk
import re

class Preprocesamiento:
	def __init__(self):
		self.nlp = spacy.load("en_core_web_sm")
		self.stopwords = set(nltk.corpus.stopwords.words('english')) | self.nlp.Defaults.stop_words#Utilizamos las stopwords proporcionadas por nltk y spacy
		self.tokens = []
	
	def setText(self,text):
		self.tokens= self.nlp(text)

	
	def deleteSpecialChars(self,text):
		expr = r'[^a-zA-Z]'
		filtered = " " 
		for token in text.split():
			cadena = re.sub(expr, r' ', token)
			if (len(cadena) > 0):
				filtered =filtered +" "+ cadena.lower() 	
		return filtered 
	
	def getTokens(self):
		toks = []
		for tok in toks:
			toks.append(tok)
		return toks
	def getLemmas(self):
		lemmas = []
		for tok in tokens:
			if not(tok in stopwords):
				lemmas.append(tok)
		
	def etiquetarPos(self):
		lemasPos = []
		for tok in self.tokens:
			lemmasPos.append(tok.text+" "+tok.pos_)
		return lemmasPos

	def delStopWords(self):
		ss = []
		for tok in self.tokens:
			if (not(tok.text in self.stopwords) ):
				ss.append(tok)
		return ss		


	def lemmatize_delSW(self):# Lemmatiza y quita las stopwords
		lemmas = []
		for tok in self.tokens:
			if (not(tok.text in self.stopwords) and tok.text != " " and not(tok.lemma_ in self.stopwords)):
					lemmas.append(tok.lemma_)
					#print(lemmas[-1],len(lemmas[-1]))
		return lemmas
pro = Preprocesamiento()
pro.setText("hi  how are you")
print(pro.lemmatize_delSW())