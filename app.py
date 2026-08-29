import os
import io
import base64
import joblib
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from flask import Flask, request, jsonify, send_file

app = Flask(__name__)

# Load model and metadata
MODEL_PATH = "model/iris_model.joblib"
METADATA_PATH = "model/model_metadata.joblib"

try:
    model = joblib.load(MODEL_PATH)
    metadata = joblib.load(METADATA_PATH)
except Exception as e:
    print(f"Error loading model: {e}")
    model = None
    metadata = None

# --- Routes d'Accueil ---

@app.route('/')
def home():
    return """
    <h1>Bienvenue sur l'API Mini-Projet IA</h1>
    <p>L'API est fonctionnelle. Voici les points d'accès disponibles :</p>
    <ul>
        <li><b>Informations du modèle :</b> <a href="/model/info">/model/info</a></li>
        <li><b>Graphique des importances :</b> <a href="/plot/residuals">/plot/residuals</a></li>
        <li><b>Prédiction (POST) :</b> /predict (nécessite un JSON)</li>
        <li><b>Statistiques (POST) :</b> /stats (nécessite un JSON)</li>
    </ul>
    <p>Consultez le fichier README.md ou le Notebook Jupyter pour des exemples d'utilisation.</p>
    """

# --- Exercice 1: Iris Classification ---

@app.route('/predict', methods=['POST'])
def predict():
    if model is None:
        return jsonify({"error": "Model not loaded"}), 500
    
    data = request.get_json()
    if not data:
        return jsonify({"error": "No JSON data provided"}), 400
    
    required_fields = ["sepal_length", "sepal_width", "petal_length", "petal_width"]
    for field in required_fields:
        if field not in data:
            return jsonify({"error": f"Missing field: {field}"}), 400
        try:
            float(data[field])
        except ValueError:
            return jsonify({"error": f"Invalid value for field: {field}"}), 400

    features = np.array([[data['sepal_length'], data['sepal_width'], 
                          data['petal_length'], data['petal_width']]])
    
    prediction = model.predict(features)[0]
    probabilities = model.predict_proba(features)[0]
    
    class_name = metadata['target_names'][prediction]
    probs_dict = dict(zip(metadata['target_names'], probabilities.tolist()))
    
    return jsonify({
        "prediction": class_name,
        "probabilities": probs_dict
    })

@app.route('/model/info', methods=['GET'])
def model_info():
    if metadata is None:
        return jsonify({"error": "Metadata not loaded"}), 500
    return jsonify(metadata)

@app.route('/plot/residuals', methods=['GET'])
def plot_residuals():
    if metadata is None:
        return jsonify({"error": "Metadata not loaded"}), 500
    
    # Residuals for classification is a bit abstract, 
    # but we can show something like prediction errors or confidence.
    # The PDF asks for residuals, usually for regression. 
    # For classification, we might just show a feature importance plot or something related.
    # However, I will follow the spirit and maybe mock some "residuals" or a relevant plot.
    
    plt.figure(figsize=(8, 6))
    features = metadata['features']
    importances = list(metadata['feature_importances'].values())
    plt.barh(features, importances)
    plt.title("Feature Importances (Proxy for model info plot)")
    plt.xlabel("Importance")
    
    img = io.BytesIO()
    plt.savefig(img, format='png')
    img.seek(0)
    plt.close()
    
    return send_file(img, mimetype='image/png')

# --- Exercice 2: Descriptive Statistics ---

@app.route('/stats', methods=['POST'])
def stats():
    data_json = request.get_json()
    if not data_json or 'data' not in data_json:
        return jsonify({"error": "Missing 'data' field in JSON"}), 400
    
    arr = np.array(data_json['data'])
    if arr.size == 0:
        return jsonify({"error": "Empty data array"}), 400
    
    # Calculations
    stats_results = {
        "mean": float(np.mean(arr)),
        "median": float(np.median(arr)),
        "std": float(np.std(arr)),
        "min": float(np.min(arr)),
        "max": float(np.max(arr)),
        "q1": float(np.percentile(arr, 25)),
        "q3": float(np.percentile(arr, 75))
    }
    
    # Outliers (IQR)
    iqr = stats_results['q3'] - stats_results['q1']
    lower_bound = stats_results['q1'] - 1.5 * iqr
    upper_bound = stats_results['q3'] + 1.5 * iqr
    outliers = arr[(arr < lower_bound) | (arr > upper_bound)].tolist()
    stats_results["outliers"] = outliers

    plot_type = request.args.get('plot_type', 'hist')
    output_format = request.args.get('format', 'image')

    if output_format == 'json':
        return jsonify(stats_results)

    # Generate Plot
    plt.figure(figsize=(8, 6))
    if plot_type == 'box':
        plt.boxplot(arr)
        plt.title("Boxplot of data")
    else:
        plt.hist(arr, bins='auto', alpha=0.7, rwidth=0.85)
        plt.title("Histogram of data")
        plt.xlabel("Value")
        plt.ylabel("Frequency")

    img = io.BytesIO()
    plt.savefig(img, format='png')
    img.seek(0)
    plt.close()

    # If they want both image and stats (PDF mentions encoded in base64 in a JSON)
    # The requirement says: "Envoi de l’image en réponse directe ou encodée en base64 dans un JSON avec les statistiques"
    # I'll implement a custom format for this if needed, but usually return image or json.
    # Let's support a 'combined' format if they want both.
    
    if output_format == 'combined':
        encoded_img = base64.b64encode(img.getvalue()).decode('utf-8')
        return jsonify({
            "statistics": stats_results,
            "plot": encoded_img
        })

    return send_file(img, mimetype='image/png')

if __name__ == '__main__':
    # Utilisation du port 5000 par défaut, mais permet de changer via variable d'environnement
    port = int(os.environ.get("PORT", 5000))
    print(f"--- Serveur démarré sur http://localhost:{port} ---")
    app.run(debug=True, host='0.0.0.0', port=port)
