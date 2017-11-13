import numpy as np
import cv2
import math
import KDtree as kd

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
	(rows, columns) = img.shape
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
	(rows, columns) = img.shape
	for r in range(rows - b + 1):
		for c in range(columns - b + 1):
			yield (r, c)

def calcular_lbps(img, numVariacoes = 3):
	(rows, columns) = img.shape
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
	histograma = np.zeros(lbpMax + 1, dtype=int)
	for lbp in lbps:
		histograma[lbp] += 1
	return histograma

def gerar_histograma_blocos(img, b, lbps, lbpMax):
	saida = []
	for bloco in gerar_blocos(img):
		(r, c) = bloco
		lista = []
		for x in range(r, r+b):
			for y in range(c, c+b):
				lista.append(lbps[x, y])
		saida.append(gerar_histograma(lista, lbpMax))
	return saida

def dist(vA, vB):
	saida = 0
	for i in range(len(vA)):
		saida += (vA[i] - vB[i])*(vA[i] - vB[i])
	return saida

def knn(vetores):
	saida = []
	tree = kd.KdTree(vetores, kd.Orthotope([0] * len(vetores[0]), [100] * len(vetores[0])))
	for i in range(10):# range(len(vetores)):
		print("achando "+str(i))
		saida.append(kd.find_nearest(len(vetores[0]), tree, vetores[i]).nearest)
	
	#for i in range(len(vetores)):
	#	maisProximo = i
	#	menorDist = 1000000
	#	for j in range(len(vetores)):
	#		if ((i != j) and (dist(vetores[i], vetores[j]) < menorDist)):
	#			menorDist = dist(vetores[i], vetores[j])
	#			maisProximo = j
	#	saida.append(maisProximo)
	
	return saida

def identificar(vA, vB, vC):
	saida = np.zeros(len(vA), dtype=int)
	for i in range(len(vA)):
		if (vA[i] == vB[i] or vB[i] == vC[i] or vC == vA[i]):
			saida[i] = 1
		else:
			saida[i] = 0
	return saida

def editar(img, blocos, b = 10):
	(rows, columns, _) = img.shape
	i = 0
	for x in range(rows - b + 1):
		for y in range(columns - b + 1):
			if blocos[i] == 1:
				img[x:x+b, y:y+b] = [0, 0, 0]
			i += 1
	return img

img = cv2.imread("Teste 3.jpg")
cinza = rgb_para_cinza(img)
print("Shape")
print(img.shape)
print(cinza.shape)
print("LBPS")
print(lbp_u2(cinza, 25, 25, 8, 1))
print(lbp_riu2(cinza, 25, 25, 8, 1))
print(gerar_histograma([], 10))
print("Tabela LBPS")
lbps = calcular_lbps(cinza)
print(lbps)
histogramaA = gerar_histograma_blocos(cinza, 10, lbps[:,:,0], 8 * (8-1) + 2)
histogramaB = gerar_histograma_blocos(cinza, 10, lbps[:,:,1], 12 * (12-1) + 2)
histogramaC = gerar_histograma_blocos(cinza, 10, lbps[:,:,2], 16 + 1)
print(histogramaC)
print(len(histogramaC))
print(len(histogramaC[0]))
#proximosA = knn(histogramaA)
#proximosB = knn(histogramaB)
proximosC = knn(histogramaC)
print(proximosC)
#identificacao = identificar(proximosA, proximosB, proximosC)
#print(identificacao)
#img = editar(img, identificacao)

#cv2.imshow("Colorido", img)
#cv2.imshow("Cinza", cinza)
#cv2.waitKey(0)
#cv2.destroyAllWindows()
