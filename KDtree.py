from random import seed, random
from operator import itemgetter
from collections import namedtuple
from math import sqrt, inf
from copy import deepcopy


def sqd(p1, p2):
	temp = sum((c1 - c2) ** 2 for c1, c2 in zip(p1, p2))
	if temp == 0:
		return inf
	else:
		return temp


class KdNode(object):
	__slots__ = ("dom_elt", "split", "left", "right")

	def __init__(self, dom_elt, split, left, right):
		self.dom_elt = dom_elt
		self.split = split
		self.left = left
		self.right = right


class Orthotope(object):
	__slots__ = ("min", "max")

	def __init__(self, mi, ma):
		self.min, self.max = mi, ma


class KdTree(object):
	__slots__ = ("n", "bounds")

	def __init__(self, pts, bounds):
		def nk2(split, exset, tamanho):
			#print("not " + str(exset) + " = " + str(not exset))
			if not exset:
				return None
			exset.sort(key=itemgetter(split))
			m = len(exset) >> 1
			d = exset[m]
			while m + 1 < len(exset) and exset[m + 1][split] == d[split]:
				m += 1

			s2 = (split + 1) % tamanho  # cycle coordinates
			return KdNode(d, split, nk2(s2, exset[:m], tamanho), nk2(s2, exset[m + 1:], tamanho))
		
		#O ultimo elemento de cada amotra e seu indice
		for i in range(len(pts)):
			pts[i][-1] = i
		#Se tem amostras, insere as
		if pts:
			self.n = nk2(0, pts, len(pts[0]) - 1)
		self.bounds = bounds

T3 = namedtuple("T3", "nearest dist_sqd nodes_visited")


def find_nearest(k, t, p):
	def nn(kd, target, hr, max_dist_sqd):
		if kd is None:
			return T3([0.0] * k, float("inf"), 0)

		nodes_visited = 1
		s = kd.split
		pivot = kd.dom_elt
		left_hr = deepcopy(hr)
		right_hr = deepcopy(hr)
		left_hr.max[s] = pivot[s]
		right_hr.min[s] = pivot[s]

		if target[s] <= pivot[s]:
			nearer_kd, nearer_hr = kd.left, left_hr
			further_kd, further_hr = kd.right, right_hr
		else:
			nearer_kd, nearer_hr = kd.right, right_hr
			further_kd, further_hr = kd.left, left_hr

		n1 = nn(nearer_kd, target, nearer_hr, max_dist_sqd)
		nearest = n1.nearest
		dist_sqd = n1.dist_sqd
		nodes_visited += n1.nodes_visited
 
		if dist_sqd < max_dist_sqd:
			max_dist_sqd = dist_sqd
		d = (pivot[s] - target[s]) ** 2
		if d > max_dist_sqd:
			return T3(nearest, dist_sqd, nodes_visited)
		d = sqd(pivot, target)
		if d < dist_sqd:
			nearest = pivot
			dist_sqd = d
			max_dist_sqd = dist_sqd
 
		n2 = nn(further_kd, target, further_hr, max_dist_sqd)
		nodes_visited += n2.nodes_visited
		if n2.dist_sqd < dist_sqd:
			nearest = n2.nearest
			dist_sqd = n2.dist_sqd

		return T3(nearest, dist_sqd, nodes_visited)

	return nn(t.n, p, t.bounds, float("inf"))


def show_nearest(k, heading, kd, p):
	print(heading + ":")
	print("Point:            ", p)
	#print(find_nearest(k, KdTree([], Orthotope([0, 0], [10, 10])), p))
	n = find_nearest(k, kd, p)
	print(n)
	print("Nearest neighbor: ", n.nearest[0:-1])
	print("Indice + proximo: ", n.nearest[-1])
	print("Distance:         ", sqrt(n.dist_sqd))
	print("Nodes visited:    ", n.nodes_visited, "\n")

if __name__ == "__main__":
	seed(1)
	P = lambda *coords: list(coords)
	#print(P(5, 5))
	tree = KdTree([P(2, 3, 0), P(5, 4, 0), P(9, 6, 0), P(4, 7, 0), P(8, 1, 0), P(7, 2, 0), P(0, 0, 0), P(10, 10, 0)], Orthotope(P(0, 0), P(10, 10)))
	show_nearest(2, "Wikipedia example data", tree, P(10, 8))