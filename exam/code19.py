import numpy as np
r_array = np.array([
    [-1, -1, -1, -1, 0, -1],
    [-1, -1, -1, 0, -1, 100],
    [-1, -1, -1, 0, -1, -1],
    [-1, 0, 0, -1, 0, -1],
    [0, -1, -1, 0, -1, 100],
    [-1, 0, -1, -1, 0, 100]], dtype=np.float32)
q_array = np.zeros(36).reshape(6,6)
alpha, gamma = 1,0.8
for epoch in range(20):
    for i in range(6):
        for j in range(6):
            if r_array[i][j]>=0:
                q_array[i][j]=q_array[i][j]+alpha*(r_array[i][j]+gamma*max(q_array[j])-q_array[i][j])
            else:
                continue
print((q_array/np.max(q_array)*100).round())