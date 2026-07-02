import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import os

from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error

os.makedirs("outputs/potability", exist_ok=True)

# sMAPE
def smape(y_true, y_pred):
    return np.mean(2 * np.abs(y_pred - y_true) / (np.abs(y_true) + np.abs(y_pred))) * 100

df = pd.read_csv("data/dataset1.csv")

df = df.dropna()

target = "Potability"

for col in df.columns:
    if col == target:
        continue

    print(f"Processing: {col} vs Potability")

    x = df[col].values.reshape(-1, 1)
    y = df[target].values

    model = LinearRegression()
    model.fit(x, y)
    y_pred = model.predict(x)

    rmse = np.sqrt(mean_squared_error(y, y_pred))
    smape_val = smape(y, y_pred)


    plt.figure(figsize=(8, 6))
    plt.scatter(x, y, alpha=0.4, label="Data")
    plt.plot(x, y_pred, color='red', label="Linear")

    plt.title(f"{col} vs Potability (Linear)\nRMSE: {rmse:.3f}, sMAPE: {smape_val:.2f}%")
    plt.xlabel(col)
    plt.ylabel("Potability")
    plt.legend()

    plt.savefig(f"outputs/potability/linear_{col}.png", dpi=600, bbox_inches='tight')
    plt.close()

    plt.figure(figsize=(8, 6))
    plt.scatter(x, y, alpha=0.4, label="Data")

    sns.regplot(
        x=x.flatten(),
        y=y,
        scatter=False,
        lowess=True,
        color='green',
        label="LOESS",
        line_kws={"linewidth": 2}
    )

    plt.title(f"{col} vs Potability (LOESS)")
    plt.xlabel(col)
    plt.ylabel("Potability")
    plt.legend()

    plt.savefig(f"outputs/potability/loess_{col}.png", dpi=600, bbox_inches='tight')
    plt.close()

print("DONE")