import numpy as np

CSV = "../data/arseniodataset.csv"  

def ols_com_intercepto(X, y):
    X1 = np.column_stack([np.ones(X.shape[0]), X])
    beta = np.linalg.pinv(X1) @ y
    yhat = X1 @ beta
    return beta, yhat

def ols_sem_intercepto(X, y):
    beta = np.linalg.pinv(X) @ y
    yhat = X @ beta
    return beta, yhat

def r2_com_intercepto(y, yhat):
    ss_res = np.sum((y - yhat)**2)
    ss_tot = np.sum((y - y.mean())**2)
    return 1 - ss_res/ss_tot if ss_tot != 0 else 0.0

def r2_sem_intercepto(y, yhat):
    ss_res = np.sum((y - yhat)**2)
    ss_tot0 = np.sum(y**2)  
    return 1 - ss_res/ss_tot0 if ss_tot0 != 0 else 0.0

def mse(y, yhat):  return np.mean((y - yhat)**2)
def rmse(y, yhat): return np.sqrt(mse(y, yhat))
def mae(y, yhat):  return np.mean(np.abs(y - yhat))

d = np.loadtxt(CSV, delimiter=",", skiprows=1)
X_full = np.column_stack([d[:,0], d[:,2], d[:,3], d[:,4]])  
y      = d[:,5]
X_alt  = d[:,4].reshape(-1,1)                              

_, y_full_com = ols_com_intercepto(X_full, y)
_, y_full_sem = ols_sem_intercepto(X_full, y)

r2_com  = r2_com_intercepto(y, y_full_com)
r2_sem  = r2_sem_intercepto(y, y_full_sem)
rmse_com = rmse(y, y_full_com)
rmse_sem = rmse(y, y_full_sem)

print("=== (g) Intercepto forçado a zero vs com intercepto ===")
print(f"Com intercepto:  R2={r2_com:.6f}  RMSE={rmse_com:.6f}")
print(f"Sem intercepto:  R2(sem-int)={r2_sem:.6f}  RMSE={rmse_sem:.6f}")

if rmse_com < rmse_sem or (np.isclose(rmse_com, rmse_sem) and r2_com >= r2_sem):
    escolha_g = "COM intercepto"
else:
    escolha_g = "SEM intercepto"
print(f"Escolha sugerida (g): {escolha_g}\n")

# ---------- (h) ----------
_, y_alt_com = ols_com_intercepto(X_alt, y)

print("=== (h) Completo vs Alternativo ===")
print(f"Completo:     R2={r2_com_intercepto(y,y_full_com):.6f}  MSE={mse(y,y_full_com):.6f}  RMSE={rmse(y,y_full_com):.6f}  MAE={mae(y,y_full_com):.6f}")
print(f"Alternativo:  R2={r2_com_intercepto(y,y_alt_com):.6f}  MSE={mse(y,y_alt_com):.6f}  RMSE={rmse(y,y_alt_com):.6f}  MAE={mae(y,y_alt_com):.6f}")

if rmse(y,y_full_com) < rmse(y,y_alt_com) or (
    np.isclose(rmse(y,y_full_com), rmse(y,y_alt_com)) and
    r2_com_intercepto(y,y_full_com) >= r2_com_intercepto(y,y_alt_com)
):
    escolha_h = "Completo"
else:
    escolha_h = "Alternativo"
print(f"\nEscolha sugerida (h): {escolha_h}")
