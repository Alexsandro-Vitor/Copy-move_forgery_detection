import numpy as np
import cv2
import math
import KDtree as kd
import time
from skimage import feature

def rgb_para_cinza(img):
	'''
	Converte a imagem de RGB para escala de cinza. Eu uso no lugar da funcao do opencv porque ela da pesos diferentes às cores.
	'''
	b,g,r = cv2.split(img)
	return (b // 3 + g // 3 + r // 3)

def lbp_original(img, x, y, p, r):
	'''
	Obtem o LBP de um pixel da imagem
	
	Args:
		img: A imagem cujo pixel tera o LBP calculado
		x: A coordenada x do pixel
		y: A coordenada y do pixel
		p: O número de vizinhos usado 
		r: A distância dos vizinhos para o pixel
	
	Returns:
		int: O valor de LBP daquele pixel em bits
	'''
	#(rows, columns) = img.shape
	saida = ""
	for i in range(p):
		angulo = (2 * math.pi * i) / p
		yVizinho = y + round(r * math.sin(angulo))
		xVizinho = x + round(r * math.cos(angulo))
		
		#Caso o vizinho esteja fora da imagem, é considerado o pixel da imagem mais proximo dele, talvez fique, talvez nao
		xVizinho = min(max(0, xVizinho), columns - 1)
		yVizinho = min(max(0, yVizinho), rows - 1)
		x = min(max(0, x), columns - 1)
		y = min(max(0, y), rows - 1)
		
		if img[yVizinho, xVizinho] >= img[y, x]:
			saida = saida + "1"
		else:
			saida = saida + "0"
	return saida

def bits_para_int(entrada):
	'''
	Converte uma string de bits para um int
	'''
	saida = 0
	for i in entrada:
		if (i == "1"):
			saida = (saida << 1) + 1
		else:
			saida = saida << 1
	return saida

def transicoes01(lbp):
	'''
	Conta o numero de transicoes de 0 para 1 e de 1 para 0 no LBP
	'''
	saida = 0
	for i in range(len(lbp)):
		if (lbp[i-1] == "0" and lbp[i] == "1") or (lbp[i-1] == "1" and lbp[i] == "0"):
			saida += 1
	return saida

def busca01(lbp):
	'''
	Busca o indice do primeiro bit 1 no LBP
	'''
	for i in range(len(lbp)):
		if (lbp[i-1] == "0" and lbp[i] == "1"):
			return i
	return -1

def num1s(lbp):
	'''
	Conta o numero de bits 1 no LBP
	'''
	saida = 0
	for i in lbp:
		if i == "1":
			saida += 1
	return saida

def lbp_u2(img, x, y, p, r):
	'''
	Calcula o LBP uniforme de uma imagem em escala de cinza
	'''
	lbp = lbp_original(img, x, y, p, r)
	lbpInt = bits_para_int(lbp)
	if lbpInt == 0:
		return 0
	elif lbpInt == ((1 << p) - 1):
		return p * (p-1) + 1
	elif transicoes01(lbp) > 2:
		return p * (p-1) + 2
	else:
		return 1 + busca01(lbp) + p * (num1s(lbp) - 1)

def lbp_riu2(img, x, y, p, r):
	'''
	Calcula o LBP uniforme invariante a rotacao de uma imagem em escala de cinza
	'''
	lbp = lbp_original(img, x, y, p, r)
	if transicoes01(lbp) <= 2:
		return num1s(lbp)
	else:
		return p + 1

def gerar_blocos(img, b = 10):
	'''
	Cria os blocos para calcular o histograma de cada um. Os blocos sao representados apenas pela linha e coluna do pixel mais acima e a esquerda do bloco.
	'''
	#(rows, columns) = img.shape
	for r in range(rows - b + 1):
		for c in range(columns - b + 1):
			yield (r, c)

def calcular_lbps(img, numVariacoes = 3):
	#(rows, columns) = img.shape
	saida = np.zeros((rows, columns, numVariacoes), dtype=int)
	for r in range(rows):
		for c in range(columns):
			saida[r, c, 0] = lbp_u2(img, c, r, 8, 1)
			saida[r, c, 1] = lbp_u2(img, c, r, 12, 2)
			saida[r, c, 2] = lbp_riu2(img, c, r, 16, 2)
	return saida

def gerar_histograma(lbps, lbpMax):
	'''
	Gera um histograma de um conjunto de LBPs
	'''
	histograma = np.zeros(lbpMax + 1)
	for lbp in lbps:
		histograma[int(lbp)] += 1
	return histograma / len(lbps)

def gerar_histograma_blocos(img, b, lbps, lbpMax):
	saida = []
	for bloco in gerar_blocos(img, b):
		(r, c) = bloco
		lista = []
		for x in range(r, r+b):
			for y in range(c, c+b):
				lista.append(lbps[x, y])
		saida.append(gerar_histograma(lista, lbpMax))
	return saida

def dist(vA, vB):
	return sum((c1 - c2) ** 2 for c1, c2 in zip(vA, vB))

from sklearn.neighbors import KNeighborsClassifier

def knn(vetores, pixelsBloco):
	saida = np.zeros(len(vetores))
	
	#tempoKnn = time.time()
	#for i in range(20): #range(len(vetores)):
	#	maisProximo = i
	#	menorDist = math.inf
	#	for j in range(len(vetores)):
	#		if ((i != j) and (dist(vetores[i], vetores[j]) < menorDist)):
	#			menorDist = dist(vetores[i], vetores[j])
	#			maisProximo = j
	#	saida[i] = maisProximo
	#print("Tempo do Knn com array: " + str(time.time() - tempoKnn))
	#print(saida[:20])

	neigh = KNeighborsClassifier(n_neighbors=1)
	for i in range(len(vetores)):
		vetor = vetores[i]
		vetores[i] = np.array([10 for _ in vetores[i]])
		neigh.fit(vetores, np.array([x for x in range(len(vetores))]))
		vetores[i] = vetor
		saida[i] = neigh.predict([vetor])
	
	return saida

def identificar(vA, vB, vC):
	saida = np.zeros(len(vA), dtype=int)
	print(vA[0])
	for i in range(len(vA)):
		if vA[i] == vB[i] or vB[i] == vC[i] or vC[i] == vA[i]:
			saida[i] = 1
		else:
			saida[i] = 0
	return saida

def editar(img, blocos, b):
	#(rows, columns, _) = img.shape
	i = 0
	for x in range(rows - b + 1):
		for y in range(columns - b + 1):
			if blocos[i] == 1:
				img[x:x+b, y:y+b] = [0, 0, 0]
			i += 1
	return img

tempo = time.time()
img = cv2.imread("Teste 3.bmp")
cinza = rgb_para_cinza(img)
b = 15
(rows, columns) = cinza.shape
#print("Tabela LBPS")
tempoLBP = time.time()
lbpA = feature.local_binary_pattern(cinza, 8, 1, method="nri_uniform")
lbpB = feature.local_binary_pattern(cinza, 12, 2, method="nri_uniform")
lbpC = feature.local_binary_pattern(cinza, 16, 2, method="uniform")
print("Tempo de calculo do LBP: " + str(time.time() - tempoLBP))

tempoHist = time.time()
histogramaA = gerar_histograma_blocos(cinza, b, lbpA, 8 * (8-1) + 2)
histogramaB = gerar_histograma_blocos(cinza, b, lbpB, 12 * (12-1) + 2)
histogramaC = gerar_histograma_blocos(cinza, b, lbpC, 16 + 1)
print("Tempo de geracao dos histogramas: " + str(time.time() - tempoHist))
print(len(histogramaC))
#print(len(histogramaC[0]))

tempoKnn = time.time()
proximosA = knn(histogramaA, b*b)
print("Tempo de calculo do KNN A: " + str(time.time() - tempoKnn))
tempoKnnB = time.time()
proximosB = knn(histogramaB, b*b)
print("Tempo de calculo do KNN B: " + str(time.time() - tempoKnnB))
tempoKnnC = time.time()
proximosC = knn(histogramaC, b*b)
print("Tempo de calculo do KNN C: " + str(time.time() - tempoKnnC))
print("Tempo de calculo do KNN: " + str(time.time() - tempoKnn))

print(proximosA[:30])
print(proximosB[:30])
print(proximosC[:30])

#print(proximosC)

#tempoId = time.time()
#identificacao = identificar(proximosA, proximosB, proximosC)
#print("Tempo de identificacao de copias: " + str(time.time() - tempoId))

#tempoEdit = time.time()
#img = editar(img, identificacao, b)
#print("Tempo de edicao: " + str(time.time() - tempoEdit))

print("Tempo de Execucao: " + str(time.time() - tempo))

cv2.imshow("Colorido", img)
cv2.imshow("Cinza", cinza)
cv2.waitKey(0)
cv2.destroyAllWindows()
