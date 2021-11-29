import streamlit as st
import numpy as np
import random as r
import pandas as pd
import graphviz as g
from Probas.probas import algo_etude_probas
from Stats.stats import algo_etude_stats

liste_exos = ["Personalisé", "Exercice C1", "Exercice C2", "Exercice C3", "Exercice C4", "Exercice C5", "Exercice C6",
              "Exercice B1", "Exercice B2", "Exercice B3", "Exercice B4", "Exercice B5"]

st.sidebar.title("Paramètres")

k = st.sidebar.slider(label="Valeur de k : ", min_value=1, max_value=100)

dataset = st.sidebar.selectbox("Importer un dataset : ", liste_exos)

if dataset == "Personalisé":

    nombre_objets = st.sidebar.selectbox("Nombre d'objets : ", range(10))

    # Liste qui contiendra le nom des objets
    objets = []
    for i in range(nombre_objets):
        # Récupérer le nom des objets
        objets.append(st.sidebar.text_input(f"Label ou nom de l'objet {i+1}"))

    # Liste qui contiendra toutes les probabilités
    P = []
    st.sidebar.text("Définition de la matrice P : ")
    for i in range(nombre_objets):
        for y in range(nombre_objets):
            # Récupérer les probabilités
            P.append(st.sidebar.slider(f"P({objets[y]}|{objets[i]}) : ",min_value=0., max_value=1., step=0.01, key=f"[{i}][{y}]"))

    # Traitement des données récupérées pour la matrice P
    # On convertit P en array Numpy par soucis de performance du programme
    MP = np.array(P, dtype=np.float64).reshape((nombre_objets, nombre_objets), order='F')
    # On crée un dataframe pour l'affichage dans l'application
    P_df = pd.DataFrame(P)
    # Changement des noms et colonnes
    #resume_df.index = objets
    for i in range(nombre_objets):
        P_df = P_df.rename(columns={i:objets[i]}, index={i:f"{objets[i]}"})


else:
    dataset_id = dataset.replace("Exercice ", "")
    P_df = pd.read_excel(f"datasets/{dataset_id}.xlsx")
    MP = np.array(P_df)
    objets = list(P_df.columns)
    nombre_objets = len(MP)

# Liste qui contiendra la loi de X pour k=0
st.sidebar.text("Définition de la loi de X pour k = 0 : ")
X = []
for i in range(nombre_objets):
    X.append(st.sidebar.slider(f"P(X = {objets[i]}) pour k = 0 : ",min_value=0., max_value=1., step=0.01, key=f"X[{i}]"))

# Traitement des données pour X
X = np.array(X, dtype=np.float64).reshape((nombre_objets, 1), order='F')
X_df = pd.DataFrame(X)
for i in range(nombre_objets):
    X_df = X_df.rename(index={i:f"i={objets[i]}"})
X_df = X_df.rename(columns={0:"P(X=i) pour k=0"})

# Partie stats #

st.sidebar.write("Partie statistiques")

st.sidebar.text("Définir la répartition des sujets pour k=0")

X_stats = []
for i in range(nombre_objets):
    X_stats.append(st.sidebar.slider(f"Pour l'objet {objets[i]}", min_value=0, max_value=1000))

stats = algo_etude_stats(MP, np.array(X_stats), k)

valid_button = st.sidebar.button("Faire l'étude avec ces paramètres")

# Définition du graphe
graph = g.Digraph()
for i in range(nombre_objets):
    for y in range(nombre_objets):
        if MP[y][i] != 0.0:
            graph.edge(objets[i], objets[y], label=f"{MP[y][i]}", color="red")

######## AFFICHAGE ########

st.title("Etude des probabilités")

st.write("Matrice des probabilités conditionelles: ")
st.write(P_df)
try:
    st.text(f"On lit : P({objets[nombre_objets-1]}|{objets[0]}) = {P[nombre_objets-1][0]}")
except:
    pass

st.write("Loi de X pour k=0 : ")
st.write(X_df)

st.write("Graphe qui modélise la matrice des probabilités : ")
st.graphviz_chart(graph)

X, X_data = algo_etude_probas(MP, X, k, nombre_objets)

st.write("Valeur de P(X) en fonction de k : ")
st.line_chart(pd.DataFrame(X_data.T, columns=objets))
try:
    st.text(f"On lit : quand k = {k//2}, P(X={objets[0]}) = {X_data[0][k//2]}")
except:
    pass

X = np.array(X, dtype=np.float64).reshape((nombre_objets, 1), order='F')
X_df = pd.DataFrame(X)
for i in range(nombre_objets):
    X_df = X_df.rename(index={i:f"i={objets[i]}"})
X_df = X_df.rename(columns={0:f"P(X=i) pour k={k}"})

st.write(f"Loi de X pour k = {k} : ")
st.write(X_df)

st.title("Etude statistiques")
st.write(f"Pour {np.sum(X_stats)} sujet(s)")
st.line_chart(pd.DataFrame(stats, columns=objets))

try:
    st.write(f"On lit : quand k = {k//2}, il y a {stats.T[0][k//2] * 100}% de sujet(s) dans {objets[0]}")
except:
    pass

st.button("Refaire une étude statistiques")



