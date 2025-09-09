import numpy as np

dados = np.loadtxt("data/arseniodatasetALT.csv", delimiter=",", skiprows=1)

X = dados[:,0]
Y = dados[:,1]
xbar = np.mean(X)
ybar = np.mean(Y)

a = np.sum((X-xbar)*(Y-ybar)) / np.sum((X - xbar)** 2)
b = ybar - a*xbar
print(a)
print(b)

ytest = a*X + b

#Equação ilustrada:
print(f"A equação da reta é Y = {a:.2f}x + {b:.2f}")

r_squared = 1 - (np.sum((Y - ytest)**2)
                 /np.sum((Y - ybar)**2))

print(f"R² score: {r_squared:.3f}")
print("80.4% da variação na variável Arsênio na Água pode ser atribuída ao modelo.")

#fig = px.scatter(x=X, y=Y, 
#                 trendline="ols", 
#                 labels={"x":"x","y":"y"})
#fig.show()