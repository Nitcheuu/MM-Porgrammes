import numpy as np
import matplotlib.pyplot as plt


"""def read_file(chemin_fichier):
    objets = {}
    with open(chemin_fichier, 'r', encoding="utf-8") as config:
        for obj in config.read().splitlines()[2].split(";"):
            objets[obj] = len(objets)
    print(objets)


read_file("../config.txt")"""

"""objets = {
    'A': 0,
    'B': 1,
    'C': 2
}

X = np.array([1, 0, 0], dtype=np.longdouble)
X = X.reshape((len(objets), 1))
X_data = np.array(X)
""""""P_data = np.append(P_data, P_data)
P_data = P_data.reshape((3, 2), order='F')""""""

k = 100

P = np.array([0.15, 0.85, 0, 0.05, 0, 0.95, 0, 0.01, 0.99], dtype=np.longdouble)
P = P.reshape((3, 3), order='F')"""


def algo_etude_probas(P, X, k, nombre_objets):
    X_data = np.array(X)
    for i in range(k):
        X = np.dot(P, X)
        X_data = np.append(X_data, X)

    X_data = X_data.reshape((k+1, nombre_objets)).T
    return X, X_data


"""plt.plot(X_data[objets['A']])
plt.plot(X_data[objets['B']])
plt.plot(X_data[objets['C']])
plt.show()"""