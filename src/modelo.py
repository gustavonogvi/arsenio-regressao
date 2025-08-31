import os

import numpy as np
import pandas as pd

PATH = os.path.abspath("data/arseniodataset.csv")

df = pd.read_csv(PATH)

# Treat Data
df["Uso_Cozinhar"] = df["Uso_Cozinhar"].apply(
    lambda x: 0.25 if x <= 2 else 0.50 if x == 3 else 0.75
)
df["Uso_Beber"] = df["Uso_Beber"].apply(
    lambda x: 0.25 if x <= 2 else 0.50 if x == 3 else 0.75
)


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
