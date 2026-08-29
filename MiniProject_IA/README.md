# Mini-Projet IA: Service Web pour la Data Science

Ce projet implémente une API REST pour la classification d'Iris et l'analyse statistique descriptive.

## Installation

1. Assurez-vous d'avoir Python installé (recommandé: Python 3.8+).
2. Installez les dépendances :
   ```bash
   pip install -r requirements.txt
   ```

## Utilisation

1. **Entraînement du modèle** (si nécessaire) :
   ```bash
   python train_iris.py
   ```
2. **Lancement de l'API** :
   ```bash
   python app.py
   ```
   L'API sera accessible sur `http://localhost:5000`.

## Endpoints

### 1. Classification d'Iris
- **POST `/predict`** : Prédit l'espèce d'une fleur.
  - Body JSON : `{"sepal_length": 5.1, "sepal_width": 3.5, "petal_length": 1.4, "petal_width": 0.2}`
- **GET `/model/info`** : Retourne les métadonnées du modèle.
- **GET `/plot/residuals`** : Retourne un graphique des importances des caractéristiques.

### 2. Statistiques Descriptives
- **POST `/stats`** : Calcule des statistiques sur une série de nombres.
  - Paramètres URL : `plot_type` (hist|box), `format` (image|json|combined)
  - Body JSON : `{"data": [10, 20, 30, 40, 100]}`

## Exemple de test (cURL)
```bash
curl -X POST http://localhost:5000/predict -H "Content-Type: application/json" -d "{\"sepal_length\": 5.1, \"sepal_width\": 3.5, \"petal_length\": 1.4, \"petal_width\": 0.2}"
```
