import pandas as pd
import os

from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier, plot_tree
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
import matplotlib.pyplot as plt

os.makedirs("outputs/classification", exist_ok=True)

df = pd.read_csv("data/dataset1.csv")
df = df.dropna()

X = df.drop("Potability", axis=1)
y = df["Potability"]

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

simple_model = DecisionTreeClassifier(
    max_depth=3,
    class_weight='balanced',
    random_state=42
)
simple_model.fit(X_train, y_train)
y_pred_simple = simple_model.predict(X_test)

print("\n Decision Tree")
print("Accuracy:", round(accuracy_score(y_test, y_pred_simple), 3))
print("Precision:", round(precision_score(y_test, y_pred_simple), 3))
print("Recall:", round(recall_score(y_test, y_pred_simple), 3))
print("F1-score:", round(f1_score(y_test, y_pred_simple), 3))

plt.figure(figsize=(16, 8))
plot_tree(
    simple_model,
    feature_names=X.columns,
    class_names=["Not potable", "Potable"],
    filled=True,
    fontsize=8
)
plt.savefig("outputs/classification/decision_tree_clean.png", dpi=600, bbox_inches='tight')
plt.close()

complex_model = DecisionTreeClassifier(
    class_weight='balanced',
    random_state=42
)
complex_model.fit(X_train, y_train)
y_pred_complex = complex_model.predict(X_test)

print("\n Decision Tree (komplexny)")
print("Accuracy:", round(accuracy_score(y_test, y_pred_complex), 3))
print("Precision:", round(precision_score(y_test, y_pred_complex), 3))
print("Recall:", round(recall_score(y_test, y_pred_complex), 3))
print("F1-score:", round(f1_score(y_test, y_pred_complex), 3))

plt.figure(figsize=(24, 12))
plot_tree(
    complex_model,
    feature_names=X.columns,
    class_names=["Not potable", "Potable"],
    filled=True,
    fontsize=6
)
plt.savefig("outputs/classification/decision_tree.png", dpi=600, bbox_inches='tight')
plt.close()

print("\nDONE")