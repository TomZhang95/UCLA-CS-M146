
import numpy as np
import matplotlib.pyplot as plt


mean_0 = (1, 1)
cov_identity = [[1,0],[0,1]]
cov_doubled = [[2,0],[0,2]]
cov_d = [[1, 0.5],[0.5, 1]]
cov_e = [[1, -0.5], [-0.5, 1]]
x1, x2 = np.random.multivariate_normal(mean_0, cov_e, 1000).T
plt.plot(x1, x2)
plt.show()