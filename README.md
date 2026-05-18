# TP KNN - Dataset Bien-être

Application d'un classifieur K-Nearest Neighbors sur le dataset `bienetre.csv` pour prédire la classe `target` (3 classes).

## Structure

```
.
├── bienetre.csv          # dataset
├── preprocessing.py      # exploration, split, normalisation
├── knn.py                # entraînement et évaluation du KNN
├── requirements.txt
└── README.md
```

## Installation

```bash
python -m venv venv
.\venv\Scripts\Activate.ps1
pip install -r requirements.txt
```

## Exécution

```bash
python preprocessing.py
python knn.py
```

`preprocessing.py` génère des fichiers `.npy` (données scalées) qui sont ensuite chargés par `knn.py`.

## Démarche

1. **Exploration** : vérification des NaN, doublons, distribution de la target
2. **Split** train/test stratifié (80/20)
3. **Normalisation** avec `StandardScaler` (le KNN est sensible aux échelles)
4. **Choix de K** par validation croisée 5-fold sur K = 1 à 20
5. **Évaluation** du modèle final : accuracy, classification report, matrice de confusion