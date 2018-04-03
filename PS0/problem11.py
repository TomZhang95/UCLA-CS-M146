#Problem 11
import numpy as np

matrix = [[1, 0], [1, 3]]
e_vals, e_vecs = np.linalg.eig(matrix)
index = 0
for i in range(1, len(e_vals)):
    if e_vals[i] > e_vals[index]:
        index = i
print(e_vals[index])
print(e_vecs[index])