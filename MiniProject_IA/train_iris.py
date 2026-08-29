import os
import joblib
import numpy as np
import pandas as pd
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score

def train_and_save_model():
    # Load dataset
    iris = load_iris()
    X = iris.data
    y = iris.target
    feature_names = iris.feature_names
    target_names = iris.target_names

    # Split data
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    # Train model
    model = LogisticRegression(max_iter=200)
    model.fit(X_train, y_train)

    # Evaluate
    y_pred = model.predict(X_test)
    accuracy = accuracy_score(y_test, y_pred)
    print(f"Model Accuracy: {accuracy:.4f}")

    # Prepare metadata
    metadata = {
        "model_type": "LogisticRegression",
        "accuracy": accuracy,
        "features": [f.replace(" (cm)", "").replace(" ", "_") for f in feature_names],
        "target_names": target_names.tolist(),
        "feature_importances": dict(zip([f.replace(" (cm)", "").replace(" ", "_") for f in feature_names], 
                                         np.abs(model.coef_).mean(axis=0).tolist()))
    }

    # Save model and metadata
    os.makedirs("model", exist_ok=True)
    joblib.dump(model, "model/iris_model.joblib")
    joblib.dump(metadata, "model/model_metadata.joblib")
    print("Model and metadata saved to model/ directory.")

if __name__ == "__main__":
    train_and_save_model()
