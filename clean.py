import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from sklearn.preprocessing import StandardScaler

df = pd.read_csv("bienetre.csv")

print(df.shape)
print(df.head())
print(df.isnull().sum().sum(), "valeurs manquantes")
print(df.duplicated().sum(), "doublons")
print(df["target"].value_counts())

df["target"].value_counts().plot(kind="bar")
plt.title("Répartition des classes")
plt.show()

X = df.drop("target", axis=1)
y = df["target"]

scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

np.savez("data.npz", X=X_scaled, y=y.values)