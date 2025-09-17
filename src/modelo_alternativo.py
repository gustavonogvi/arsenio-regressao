import numpy as np
import matplotlib.pyplot as plt

dados = np.loadtxt("data/arseniodatasetALT.csv", delimiter=",", skiprows=1)

X = dados[:,0]
Y = dados[:,1]
xbar = np.mean(X)
ybar = np.mean(Y)

a = np.sum((X-xbar)*(Y-ybar)) / np.sum((X - xbar)** 2)
b = ybar - a*xbar
print(a)
print(b)

ypred = a*X + b

print(ypred)

#Equação ilustrada:
print(f"A equação da reta é Y = {a:.2f}x + {b:.2f}")

r_squared = 1 - (np.sum((Y - ypred)**2)
                 /np.sum((Y - ybar)**2))

print(f"R² score: {r_squared:.3f}")
print("cerca de 80.4% da variação na variável Arsênio na Água pode ser atribuída ao modelo.")


plt.scatter(Y, ypred, color="blue", label="Valores preditos")
plt.plot(Y, Y, color="red", label="Reta ideal (y = y)")

plt.xlabel("Valor real (Arsenio_Unhas)")
plt.ylabel("Valor predito")
plt.title("Regressão Linear - Valores reais vs preditos")
plt.legend()
plt.show()