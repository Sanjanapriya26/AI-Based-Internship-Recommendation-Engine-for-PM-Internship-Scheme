"""
app.py — Flask REST API for PM Internship ML Recommendation Engine
"""

import os
import sys
import logging
from flask import Flask, request, jsonify, render_template

# Ensure project root is in path
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, BASE_DIR)

from ml_model import recommend, load_model, train_model
from data.internships import INTERNSHIPS

# ─────────────────────────────────────────────
# Flask Setup
# ─────────────────────────────────────────────

app = Flask(__name__, template_folder="templates")
app.config["JSON_SORT_KEYS"] = False

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s"
)

logger = logging.getLogger(__name__)

MODEL_PATH = os.path.join("models", "recommendation_model.pkl")
artifacts = None


# ─────────────────────────────────────────────
# Model Loader
# ─────────────────────────────────────────────

def get_artifacts():
    global artifacts

    if artifacts is None:
        if os.path.exists(MODEL_PATH):
            logger.info("Loading saved model...")
            artifacts = load_model(MODEL_PATH)
        else:
            logger.info("Training new model...")
            artifacts = train_model()

    return artifacts


# ─────────────────────────────────────────────
# CORS (manual)
# ─────────────────────────────────────────────

@app.after_request
def add_cors_headers(response):
    response.headers["Access-Control-Allow-Origin"] = "*"
    response.headers["Access-Control-Allow-Headers"] = "Content-Type"
    response.headers["Access-Control-Allow-Methods"] = "GET, POST, OPTIONS"
    return response


@app.route("/api/recommend", methods=["OPTIONS"])
def options_handler():
    return "", 200


# ─────────────────────────────────────────────
# ROUTES
# ─────────────────────────────────────────────

@app.route("/")
def home():
    return render_template("index.html")


@app.route("/api/health")
def health():
    return jsonify({
        "status": "ok",
        "internships_count": len(INTERNSHIPS),
        "model_loaded": os.path.exists(MODEL_PATH)
    })


@app.route("/api/internships")
def list_internships():
    return jsonify({
        "count": len(INTERNSHIPS),
        "internships": INTERNSHIPS
    })


@app.route("/api/recommend", methods=["POST"])
def recommend_endpoint():
    try:
        data = request.get_json()

        if not data:
            return jsonify({"success": False, "error": "JSON body required"}), 400

        if "education" not in data:
            return jsonify({"success": False, "error": "Missing required field: education"}), 400

        candidate = {
            "name": data.get("name", "Candidate"),
            "education": data.get("education", "10th"),
            "field": data.get("field", ""),
            "skills": data.get("skills", []),
            "interests": data.get("interests", []),
            "location": data.get("location", "Anywhere"),
            "state": data.get("state", "")
        }

        top_n = min(int(data.get("top_n", 5)), 10)

        logger.info(f"Recommendation request for {candidate['name']}")

        arts = get_artifacts()
        recommendations = recommend(candidate, arts, top_n=top_n)

        return jsonify({
            "success": True,
            "candidate": candidate,
            "recommendations": recommendations
        })

    except Exception as e:
        logger.error(str(e), exc_info=True)
        return jsonify({"success": False, "error": str(e)}), 500


@app.route("/api/train", methods=["POST"])
def retrain():
    global artifacts
    artifacts = train_model()
    return jsonify({"success": True, "message": "Model retrained successfully"})


# ─────────────────────────────────────────────
# MAIN
# ─────────────────────────────────────────────

if __name__ == "__main__":
    logger.info("Starting PM Internship ML API Server...")
    get_artifacts()
    app.run(host="0.0.0.0", port=5000, debug=True)
