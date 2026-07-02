import pandas as pd
import os

from pgmpy.models import BayesianNetwork
from pgmpy.estimators import MaximumLikelihoodEstimator
from pgmpy.inference import VariableElimination

os.makedirs("outputs/classification", exist_ok=True)

df = pd.read_csv("data/dataset1.csv")
df = df.dropna()

attributes = ["ph", "Solids", "Sulfate", "Chloramines"]

df_cat = df.copy()
for col in attributes:
    df_cat[col] = pd.qcut(
        df[col],
        q=3,
        labels=["low", "medium", "high"]
    )

df_cat["Potability"] = df["Potability"]

for col in attributes:
    print(f"\nRozdelenie kategórií pre {col}:")
    print(df_cat[col].value_counts(normalize=True).round(3) * 100)

model = BayesianNetwork([
    ("ph", "Potability"),
    ("Solids", "Potability"),
    ("Sulfate", "Potability"),
    ("Chloramines", "Potability"),
])

model.fit(df_cat[attributes + ["Potability"]], estimator=MaximumLikelihoodEstimator)

with open("outputs/classification/cpt.txt", "w", encoding="utf-8") as f:
    for cpd in model.get_cpds():
        f.write(str(cpd))
        f.write("\n\n")

print("\nCPT tabuľky uložené do outputs/classification/cpt.txt")

inference = VariableElimination(model)

result_high = inference.query(
    variables=["Potability"],
    evidence={"ph": "high", "Solids": "high", "Sulfate": "high", "Chloramines": "high"}
)
print("\nPravdepodobnosť pitnosti pri vysokých hodnotách všetkých atribútov:")
print(result_high)

result_medium = inference.query(
    variables=["Potability"],
    evidence={"ph": "medium", "Solids": "medium", "Sulfate": "medium", "Chloramines": "medium"}
)
print("\nPravdepodobnosť pitnosti pri stredných hodnotách všetkých atribútov:")
print(result_medium)

print("\nDONE")