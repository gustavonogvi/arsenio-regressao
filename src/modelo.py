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

# 4ª: R²
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

# 5ª: Testing datas with Idade=30, Uso_Beber = 5, Uso_Cozinhar = 5, Arsenio_Agua = 0.135
y_teste = beta[0] + beta[1] * 30 + beta[2] * 5 + beta[3] * 5 + beta[4] * 0.135
print("------------------------------------------------")
print(f"y_teste: {y_teste}")
print("------------------------------------------------")


plt.scatter(y, y_predictor, color="blue", label="Valores preditos")
plt.plot(y, y, color="red", label="Reta ideal (y = y)")
plt.xlabel("Valor real (Arsenio_Unhas)")
plt.ylabel("Valor predito")
plt.title("Regressão Linear - Valores reais vs preditos")
plt.legend()
plt.show()

# ITEM F:

residuos = y - y_predictor

df_residuos = pd.DataFrame({
    "Observação i": range(1, n + 1),
    "Valor observado (yi)": y.ravel(),
    "Valor ajustado (ŷi)": y_predictor.ravel(),
    "Resíduo (ei)": residuos.ravel()
})

print("\nTabela de Resíduos:")
print(df_residuos)