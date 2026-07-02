import pandas as pd
import numpy as np
import shap
import matplotlib.pyplot as plt

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score

from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense

df = pd.read_csv("data/dataset1.csv")
df = df.dropna()

X = df.drop("Potability", axis=1)
y = df["Potability"]

scaler = StandardScaler()
X = scaler.fit_transform(X)

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

configs = [
    (32, 16),
    (64, 32),
    (128, 64),
    (256, 128)
]

print("\n--- Multilayer Neural Network ---")

for hidden1, hidden2 in configs:

    model = Sequential()

    # hidden layer 1
    model.add(Dense(
        hidden1,
        activation='relu',
        input_shape=(X_train.shape[1],)
    ))

    # hidden layer 2
    model.add(Dense(
        hidden2,
        activation='relu'
    ))

    # output layer
    model.add(Dense(1, activation='sigmoid'))

    model.compile(
        optimizer='adam',
        loss='binary_crossentropy',
        metrics=['accuracy']
    )

    model.fit(
        X_train,
        y_train,
        epochs=30,
        batch_size=32,
        verbose=0
    )

    y_pred = model.predict(X_test)

    y_pred = (y_pred > 0.5).astype(int)

    accuracy = accuracy_score(y_test, y_pred)
    precision = precision_score(y_test, y_pred)
    recall = recall_score(y_test, y_pred)
    f1 = f1_score(y_test, y_pred)

    print(f"\nConfiguration: {hidden1} - {hidden2}")

    print("Accuracy:", round(accuracy, 3))
    print("Precision:", round(precision, 3))
    print("Recall:", round(recall, 3))
    print("F1-score:", round(f1, 3))

    print("Architecture:")
    print(f"Input layer: {X_train.shape[1]} neurons")
    print(f"Hidden layer 1: {hidden1} neurons")
    print(f"Hidden layer 2: {hidden2} neurons")
    print("Output layer: 1 neuron")


print("\n--- SHAP ANALYSIS FOR BEST MODEL ---")

best_model = Sequential()

best_model.add(Dense(
    256,
    activation='relu',
    input_shape=(X_train.shape[1],)
))

best_model.add(Dense(
    128,
    activation='relu'
))

best_model.add(Dense(1, activation='sigmoid'))

best_model.compile(
    optimizer='adam',
    loss='binary_crossentropy',
    metrics=['accuracy']
)

best_model.fit(
    X_train,
    y_train,
    epochs=30,
    batch_size=32,
    verbose=0
)

# SHAP
explainer = shap.Explainer(best_model, X_train)

sample_data = X_test[:100]

shap_values = explainer(sample_data)

shap.summary_plot(
    shap_values,
    sample_data,
    feature_names=df.drop("Potability", axis=1).columns,
    show=False
)

plt.savefig("shap_summary.png", dpi=600, bbox_inches='tight')
plt.close()

from lime.lime_tabular import LimeTabularExplainer

# LIME

print("\n--- LIME ANALYSIS ---")

explainer = LimeTabularExplainer(
    training_data=X_train,
    feature_names=df.drop("Potability", axis=1).columns,
    class_names=["Not potable", "Potable"],
    mode='classification'
)

instance = X_test[0]

def predict_fn(x):
    predictions = best_model.predict(x)
    return np.hstack((1 - predictions, predictions))

exp = explainer.explain_instance(
    instance,
    predict_fn,
    num_features=5
)

fig = exp.as_pyplot_figure()

plt.tight_layout()

plt.savefig(
    "lime_plot.png",
    dpi=600,
    bbox_inches='tight'
)

plt.close()