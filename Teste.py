import numpy as np
import cv2
import math
import KDtree as kd
import time
from skimage import feature
from sklearn.neighbors import KNeighborsClassifier

def rgb_para_cinza(img):
	'''
	Converte a imagem de RGB para escala de cinza. Eu uso no lugar da funcao do opencv porque ela da pesos diferentes às cores.
	'''
	b,g,r = cv2.split(img)
	return (b // 3 + g // 3 + r // 3)

def diferenca(imgA, imgB):
	'''
	Checa a diferenca entre duas imagens
	'''
	for r in range(rows):
		for c in range(columns):
			if imgA[r, c, 0] != imgB[r, c, 0] or imgA[r, c, 1] != imgB[r, c, 1] or imgA[r, c, 2] != imgB[r, c, 2]:
				orig[r, c] = [255, 255, 255]
			else :
				orig[r, c] = [0, 0, 0]

def gerar_blocos(img):
	'''
	Cria os blocos para calcular o histograma de cada um. Os blocos sao representados apenas pela linha e coluna do pixel mais acima e a esquerda do bloco.
	'''
	r = 0
	while r < rows - b + 1:
		c = 0
		while c < columns - b + 1:
			yield (r, c)
			c += espaco
		r += espaco

def gerar_histograma(lbps, lbpMax):
	'''
	Gera um histograma de um conjunto de LBPs
	'''
	histograma = np.zeros(lbpMax + 1)
	for lbp in lbps:
		histograma[int(lbp)] += 1
	return histograma / len(lbps)

def gerar_histograma_blocos(img, lbps, lbpMax):
	'''
	Gera histogramas de todos os blocos nos quais as imagens foram divididas
	'''
	saida = []
	for (r, c) in gerar_blocos(img):
		lista = []
		for x in range(r, r+b):
			for y in range(c, c+b):
				lista.append(lbps[x, y])
		saida.append(gerar_histograma(lista, lbpMax))
	return saida

def knn(vetores, pixelsBloco):
	'''
	Obtem o vizinho mais proximo (que nao seja ele mesmo) de cada histograma
	'''
	saida = np.zeros(len(vetores), dtype=int)

	neigh = KNeighborsClassifier(n_neighbors=1)
	for i in range(len(vetores)):
		vetor = vetores[i]
		vetores[i] = np.array([100 for _ in vetores[i]])
		neigh.fit(vetores, np.array([x for x in range(len(vetores))]))
		vetores[i] = vetor
		saida[i] = neigh.predict([vetor])
	
	return saida

def identificar(vA, vB, vC):
	'''
	Identifica os blocos copiados e nao copiados
	'''
	saidaA = np.zeros(len(vA), dtype=bool)
	saidaB = np.zeros(len(vA), dtype=bool)
	for i in range(len(vA)):
		if vA[i] == vB[i] or vB[i] == vC[i] or vC[i] == vA[i]:
			saidaA[i] = True
		if vA[i] == vB[i] and vA[i] == vC[i]:
			saidaB[i] = True
	return (saidaA, saidaB)

def editar(imgA, blocosA, blocosB):
	imgB = np.copy(imgA)
	i = 0
	for (r, c) in gerar_blocos(imgA):
		if blocosA[i] == 1:
			imgA[r:r+b, c:c+b] = [0, 0, 0]
		if blocosB[i] == 1:
			imgB[r:r+b, c:c+b] = [0, 0, 0]
		i += 1
	return (imgA, imgB)

tempo = time.time()
orig = cv2.imread("Original.bmp")
img = cv2.imread("Modificado.bmp")
cinza = rgb_para_cinza(img)
b = 3
espaco = 3
(rows, columns) = cinza.shape
print("Experimento com b = " + str(b) + ", espaco = " + str(espaco))

tempoDiff = time.time()
diferenca(orig, img)
print("Tempo de calculo da diferenca real: " + str(time.time() - tempoDiff))

tempoLBP = time.time()
lbpA = feature.local_binary_pattern(cinza, 8, 1, method="nri_uniform")
lbpB = feature.local_binary_pattern(cinza, 12, 2, method="nri_uniform")
lbpC = feature.local_binary_pattern(cinza, 16, 2, method="uniform")
print("Tempo de calculo do LBP: " + str(time.time() - tempoLBP))

tempoHist = time.time()
histogramaA = gerar_histograma_blocos(cinza, lbpA, 8 * (8-1) + 2)
histogramaB = gerar_histograma_blocos(cinza, lbpB, 12 * (12-1) + 2)
histogramaC = gerar_histograma_blocos(cinza, lbpC, 16 + 1)
print("Tempo de geracao dos histogramas: " + str(time.time() - tempoHist))
print("Num. de histogramas: " + str(len(histogramaC)))

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

tempoId = time.time()
(idA, idB) = identificar(proximosA, proximosB, proximosC)
print("Tempo de identificacao de copias: " + str(time.time() - tempoId))

print(idA[:35])
print(idB[:35])

tempoEdit = time.time()
(imgA, imgB) = editar(img, idA, idB)
print("Tempo de edicao: " + str(time.time() - tempoEdit))

print("Tempo de Execucao: " + str(time.time() - tempo))

cv2.imshow("Imagem A", imgA)
cv2.imshow("Imagem B", imgB)
cv2.imshow("Diferenca real", orig)
cv2.waitKey(0)
cv2.destroyAllWindows()
