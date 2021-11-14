import streamlit as st
import numpy as np
import random as r
import pandas as pd
import graphviz as g
from Probas.probas import algo_etude_probas


st.sidebar.title("Paramètres")

nombre_objets = st.sidebar.selectbox("Nombre d'objets : ", range(10))

k = st.sidebar.slider(label="Valeur de k : ", min_value=1, max_value=100)

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

# Liste qui contiendra la loi de X pour k=0
st.sidebar.text("Définition de la loi de X pour k = 0 : ")
X = []
for i in range(nombre_objets):
        X.append(st.sidebar.slider(f"P(X = {objets[i]}) pour k = 0 : ",min_value=0., max_value=1., step=0.01, key=f"X[{i}]"))

valid_button = st.sidebar.button("Faire l'étude avec ces paramètres")

# Traitement des données récupérées pour la matrice P
# On convertit P en array Numpy par soucis de performance du programme
P = np.array(P, dtype=np.longdouble).reshape((nombre_objets, nombre_objets), order='F')
# On crée un dataframe pour l'affichage dans l'application
P_df = pd.DataFrame(P)
# Changement des noms et colonnes
#resume_df.index = objets
for i in range(nombre_objets):
    P_df = P_df.rename(columns={i:objets[i]}, index={i:f"{objets[i]}"})

# Traitement des données pour X
X = np.array(X, dtype=np.longdouble).reshape((nombre_objets, 1), order='F')
X_df = pd.DataFrame(X)
for i in range(nombre_objets):
    X_df = X_df.rename(index={i:f"i={objets[i]}"})
X_df = X_df.rename(columns={0:"P(X=i) pour k=0"})
# Définition du graphe
graph = g.Digraph()
for i in range(nombre_objets):
    for y in range(nombre_objets):
        if P[y][i] != 0.0:
            graph.edge(objets[i], objets[y], label=f"{P[y][i]}", color="red")



######## AFFICHAGE ########

st.write("Matrice des probabilités : ")
st.write(P_df)
try:
    st.text(f"On lit : P({objets[nombre_objets-1]}|{objets[0]}) = {P[nombre_objets-1][0]}")
except:
    pass

st.write("Loi de X pour k=0) : ")
st.write(X_df)

st.write("Graphe qui modélise la matrice des probabilités : ")
st.graphviz_chart(graph)

X, X_data = algo_etude_probas(P, X, k, nombre_objets)

st.write("Valeur de P(X) en fonction de k : ")
st.line_chart(pd.DataFrame(X_data.T, columns=objets))
try:
    st.text(f"On lit : quand k = {k//2}, P(X={objets[0]}) = {X_data[0][k//2]}")
except:
    pass


X = np.array(X, dtype=np.longdouble).reshape((nombre_objets, 1), order='F')
X_df = pd.DataFrame(X)
for i in range(nombre_objets):
    X_df = X_df.rename(index={i:f"i={objets[i]}"})
X_df = X_df.rename(columns={0:f"P(X=i) pour k={k}"})

st.write(f"Loi de X pour k = {k} : ")
st.write(X_df)



