import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import os

os.makedirs("outputs", exist_ok=True)

datasets = [
    "dataset1.csv",
    "dataset2.csv",
    "dataset3.xlsx"
]

for file in datasets:
    print(f"\nProcessing {file}...")

    if file.endswith(".csv"):
        df = pd.read_csv(f"data/{file}", encoding='latin1')
    elif file.endswith(".xlsx"):
        df = pd.read_excel(f"data/{file}")
    else:
        continue

    df = df.dropna(axis=1, how='all')

    if file == "dataset2.csv":
        needed_columns = [
            "latitude", "longitude",
            "sample_depth", "result", "reporting_limit"
        ]
        df = df[[col for col in needed_columns if col in df.columns]]

    elif file == "dataset3.xlsx":
        needed_columns = [
            "Upstream Basin Area", "Elevation", "Latitude", "Longitude",
            "River Width", "Discharge", "Max. Depth",
            "Lake Area", "Lake Volume", "Average Retention",
            "Area of Aquifer", "Depth of Impermeable Lining",
            "Production Zone", "Mean Abstraction Rate",
            "Mean Abstraction Level"
        ]
        df = df[[col for col in needed_columns if col in df.columns]]

    else:
        drop_columns = ["id", "station_id", "_id"]
        df = df.drop(columns=[col for col in drop_columns if col in df.columns])

    df = df.select_dtypes(include=['number'])

    df = df.dropna(thresh=3)

    df = df.loc[:, df.nunique() > 1]

    if df.shape[1] == 0:
        print(f"V  {file} nie su data")
        continue
    if file == "dataset2.csv":
        print(f"\nStats for {file}:")
        print(df[['latitude', 'longitude', 'sample_depth', 'reporting_limit']].agg(['min', 'mean', 'median', 'max']))
    corr_matrix = df.corr()

    corr_matrix.to_csv(f"outputs/{file}_correlation_matrix.csv")

    # heatmap
    plt.figure(figsize=(10, 8))
    sns.heatmap(
        corr_matrix,
        annot=True,
        cmap="coolwarm",
        fmt=".2f",
        linewidths=0.5,
        square=True,
        cbar_kws={"shrink": 0.8}
    )

    plt.title(f"Correlation Heatmap - {file}", fontsize=12)

    plt.savefig(f"outputs/{file}_heatmap.png", dpi=600, bbox_inches='tight')
    plt.close()

print("DONE")