import numpy as np

from sklearn.model_selection import cross_val_score
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix

data = np.load("data.npz")
X, y = data["X"], data["y"]

k_values = range(1, 21)
cv_scores = []
for k in k_values:
    scores = cross_val_score(
        KNeighborsClassifier(n_neighbors=k), X, y, cv=5, scoring="accuracy"
    )
    cv_scores.append(scores.mean())

best_k = list(k_values)[int(np.argmax(cv_scores))]
print("Meilleur K :", best_k)

knn = KNeighborsClassifier(n_neighbors=best_k)
knn.fit(X, y)
y_pred = knn.predict(X)

print("Accuracy :", accuracy_score(y, y_pred))
print(classification_report(y, y_pred))
print(confusion_matrix(y, y_pred))