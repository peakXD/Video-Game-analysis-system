"""
Flask Backend
=============
File    : C:\\videogames\\backend\\app.py
Run     : python app.py
Open    : http://localhost:5000

Folder structure expected:
C:\\videogames\\
├── backend\\
│   └── app.py          ← this file
├── frontend\\
│   └── index.html
└── models\\
    ├── all_models.pkl   ← copied from Airflow output
    ├── scaler.pkl       ← copied from Airflow output
    └── model_meta.json  ← copied from Airflow output
"""

from flask import Flask, jsonify, request, send_from_directory
from flask_cors import CORS
import os, json, pickle
import numpy as np

# Paths — go up one level from backend/ to find frontend/ and models/
BASE     = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
FRONTEND = os.path.join(BASE, "frontend")
MODELS   = os.path.join(BASE, "models")

app = Flask(__name__, static_folder=FRONTEND, static_url_path="")
CORS(app)

# Loaded model state
_models = {}
_scaler = None
_meta   = None


def load():
    global _models, _scaler, _meta

    p_models = os.path.join(MODELS, "all_models.pkl")
    p_scaler = os.path.join(MODELS, "scaler.pkl")
    p_meta   = os.path.join(MODELS, "model_meta.json")

    if os.path.exists(p_models):
        with open(p_models, "rb") as f: _models = pickle.load(f)
        print(f"Loaded models: {list(_models.keys())}")
    else:
        print(f"WARNING: all_models.pkl not found in {MODELS}")

    if os.path.exists(p_scaler):
        with open(p_scaler, "rb") as f: _scaler = pickle.load(f)
        print("Loaded scaler")

    if os.path.exists(p_meta):
        with open(p_meta, encoding="utf-8") as f: _meta = json.load(f)
        print("Loaded model_meta.json")
    else:
        print(f"WARNING: model_meta.json not found in {MODELS}")


load()


@app.route("/")
def index():
    return send_from_directory(FRONTEND, "index.html")


@app.route("/api/status")
def status():
    return jsonify({
        "models_ready": len(_models) > 0,
        "models":       list(_models.keys()),
        "meta_ready":   _meta is not None,
    })


@app.route("/api/info")
def info():
    """Return everything the frontend needs to build the inference form."""
    if not _meta:
        return jsonify({"error": "model_meta.json not found. Copy it to the models/ folder."}), 404
    return jsonify({
        "models":             list(_models.keys()),
        "best_model":         _meta["best_model"],
        "results":            _meta["results"],
        "features":           _meta["features"],
        "label_maps":         _meta["label_maps"],
        "feature_ranges":     _meta["feature_ranges"],
        "feature_importance": _meta["feature_importance"],
        "best_r2":            _meta["best_r2"],
    })


@app.route("/api/predict", methods=["POST"])
def predict():
    """
    POST body:
    {
      "model": "Random Forest",
      "features": { "Year": 2022, "Genre_enc": 3, "Plays": 1500, ... }
    }
    """
    if not _models:
        return jsonify({"error": "No models loaded. Copy pkl files to models/ folder."}), 503

    body       = request.get_json(force=True)
    model_name = body.get("model", list(_models.keys())[0])
    feat_vals  = body.get("features", {})

    if model_name not in _models:
        return jsonify({"error": f"Model '{model_name}' not found.",
                        "available": list(_models.keys())}), 400

    features = _meta["features"] if _meta else list(feat_vals.keys())
    X = np.array([[float(feat_vals.get(f, 0)) for f in features]])

    # Linear Regression needs StandardScaler
    X_input = _scaler.transform(X) if (model_name == "Linear Regression" and _scaler) else X

    raw  = float(_models[model_name].predict(X_input)[0])
    pred = round(max(1.0, min(10.0, raw)), 2)

    tier = "Poor" if pred < 4 else "Average" if pred < 6 else "Good" if pred < 8 else "Excellent"
    mae  = _meta["results"][model_name]["MAE"] if _meta else 1.2

    return jsonify({
        "prediction":      pred,
        "tier":            tier,
        "model":           model_name,
        "confidence_low":  round(max(1.0,  pred - mae), 2),
        "confidence_high": round(min(10.0, pred + mae), 2),
    })


@app.route("/api/reload")
def reload():
    """Call this after copying new model files, to reload without restarting."""
    load()
    return jsonify({"ok": True, "models": list(_models.keys())})


if __name__ == "__main__":
    print("\n" + "="*45)
    print("  Video Games Dashboard — Flask Backend")
    print("="*45)
    print(f"  Models folder : {MODELS}")
    print(f"  Frontend      : {FRONTEND}")
    print(f"  URL           : http://localhost:5000")
    print("="*45 + "\n")
    app.run(host="0.0.0.0", port=5000, debug=True)
