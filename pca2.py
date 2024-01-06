import matplotlib.pyplot as plt
import sklearn.decomposition as deco
import numpy as np

plt.style.use('seaborn-whitegrid')

fig, axs = plt.subplots(4)


#x = np.linspace(1, 4, 1)
#x = np.arange(1, 10, 1)
x = np.random.randint(11, size=2)

y = x + np.random.randint(11, size=2)

print(x)
print(y)

xnorm = (x - np.mean(x))/np.std(x)
ynorm = (y - np.mean(y))/np.std(y)

print(f"xnorm: \n {xnorm}")
print(f"ynorm: \n {ynorm}")

X = np.vstack((xnorm, ynorm))

print(f"np.vstack((xnorm, ynorm) \n {X}")

axs[0].plot(X[0], X[1], "o")


cov = np.cov(X)
axs[1].plot(cov[0], cov[1], "o")

print(cov)

eigenvalues, eigenvectors = np.linalg.eig(cov)

print(eigenvectors)
print(eigenvalues)

axs[2].plot(eigenvalues)
axs[3].plot(eigenvectors)


plt.show()
