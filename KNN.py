import numpy as np

from sklearn.model_selection import cross_val_score
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix

X_train_scaled = np.load("X_train_scaled.npy")
X_test_scaled = np.load("X_test_scaled.npy")
y_train = np.load("y_train.npy")
y_test = np.load("y_test.npy")

k_values = range(1, 21)
cv_scores = []
for k in k_values:
    scores = cross_val_score(
        KNeighborsClassifier(n_neighbors=k),
        X_train_scaled, y_train, cv=5, scoring="accuracy"
    )
    cv_scores.append(scores.mean())

best_k = list(k_values)[int(np.argmax(cv_scores))]
print("Meilleur K :", best_k)

knn = KNeighborsClassifier(n_neighbors=best_k)
knn.fit(X_train_scaled, y_train)
y_pred = knn.predict(X_test_scaled)

print("Accuracy :", accuracy_score(y_test, y_pred))
print(classification_report(y_test, y_pred))
print(confusion_matrix(y_test, y_pred))