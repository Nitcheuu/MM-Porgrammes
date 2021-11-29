import random
import numpy as np
import graphviz as g
import pandas as pd
"""objets = {
	"Objet1": 0.5,
	"Objet2": 0.25,
	"Objet3": 0.25
}

stats = np.zeros(len(objets))

for i in range(1000):
	nombre_alea = random.uniform(0, 1)
	intervalle = 0
	position = 0
	for o in objets:
		if nombre_alea <= intervalle + objets[o]:
			stats[position] +=1
			break
		intervalle += objets[o]
		position += 1

print(stats)

def algo_etude_stats()"""
"""df = pd.read_excel("./../../datasets/C6.xlsx")
print(df)
MP = np.array(df)
print(MP)"""

def get_position_intervalle(MP, nombre_alea, objet_actuel):
	intervalle = 0
	debut_intervalle = 0
	for i in range(len(MP)):
		if nombre_alea <= debut_intervalle + MP[i][objet_actuel]:
			return i
		intervalle += 1
		debut_intervalle += MP[i][objet_actuel]
	return -1


def algo_etude_stats(MP, X, k):
	stats_globales = np.array(np.zeros((k) * len(X)), dtype=np.float64)
	stats_globales = stats_globales.reshape(k, len(X))
	for i in range(len(X)):
		for y in range(X[i]):
			objet_actuel = i
			stats = np.array([])
			for rep in range(k):
				nombre_alea = random.uniform(0, 1)
				objet_actuel = get_position_intervalle(MP, nombre_alea, objet_actuel)
				position_sujet = np.zeros(len(X))
				position_sujet[objet_actuel] += 1 / np.sum(X)
				stats = np.append(stats, position_sujet)
				stats = stats.reshape( rep + 1, len(X))
			stats_globales += stats
	stats_globales = np.append(X / np.sum(X), stats_globales)
	stats_globales = stats_globales.reshape(k+1, len(X))
	return stats_globales





