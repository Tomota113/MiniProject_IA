#!/bin/bash
# Script de lancement pour Mac et Linux

echo "--- Nettoyage du port 5000 ---"
# Tue tout ce qui tourne sur le port 5000 (Mac/Linux)
lsof -ti:5000 | xargs kill -9 2>/dev/null

echo "--- Installation des dépendances ---"
pip install -r requirements.txt --quiet

echo "--- Entraînement du modèle ---"
python3 train_iris.py

echo "--- Lancement de l'API ---"
python3 app.py
