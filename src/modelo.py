import os

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

PATH = os.path.abspath("data/arseniodataset.csv")

df = pd.read_csv(PATH)

# Variables
y = df["Arsenio_Unhas"].values.reshape(-1, 1)
x = df[["Idade", "Uso_Beber", "Uso_Cozinhar", "Arsenio_Agua"]].values

# 1ª: Add columns of ones to have the intercept
x = np.hstack([np.ones((x.shape[0], 1)), x])

# 2ª: Get the coefficients by Normal Equation
beta = np.linalg.inv(x.T @ x) @ (x.T @ y)
print("------------------------------------------------")
print(f"(intercept + predictors): {beta}")
print("------------------------------------------------")

# 3ª: Y predictor
y_predictor = x @ beta
print("------------------------------------------------")
print(f"(y_predictor): {y_predictor}")
print("------------------------------------------------")

# 4ª: R**2
ss_total = np.sum(((y - y.mean()) ** 2))
ss_res = np.sum(((y - y_predictor) ** 2))
r_score = 1 - (ss_res / ss_total)
print("------------------------------------------------")
print(f"R²: {r_score}")
print("------------------------------------------------")

# 4.1ª: R² adjusted
n = x.shape[0]
p = x.shape[1] - 1  # Remove intercept
r2_ajustado = 1 - (1 - r_score) * (n - 1) / (n - p - 1)
print("------------------------------------------------")
print(f"R² adjusted: {r2_ajustado}")
print("------------------------------------------------")

# 5ª: Testing datas with Idade=5, Uso_Beber = 5, Uso_Cozinhar = 5, Arsenio_Agua = 0.135
x_test = np.array([1, 5, 5, 5, 0.135])
y_pred_list = []

for i in range((len(x_test))):
    y_pred = beta[0] + beta[1] * x_test[i] + beta[2] * x_test[i]
    y_pred_list.append(y_pred)

y_pred_list = np.array(y_pred_list)

plt.scatter(y, y_predictor, color="blue", label="Valores preditos")
plt.plot(y, y, color="red", label="Reta ideal (y = y)")

plt.scatter([y_pred_list], [y_pred_list], color="red", label="Letra B")

plt.xlabel("Valor real (Arsenio_Unhas)")
plt.ylabel("Valor predito")
plt.title("Regressão Linear - Valores reais vs preditos")
plt.legend()
plt.show()
