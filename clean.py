import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

RANDOM_STATE = 42

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

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=RANDOM_STATE, stratify=y
)

scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

np.save("X_train_scaled.npy", X_train_scaled)
np.save("X_test_scaled.npy", X_test_scaled)
np.save("y_train.npy", y_train.values)
np.save("y_test.npy", y_test.values)