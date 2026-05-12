"""
ml_model.py — ML Recommendation Engine for PM Internship Scheme
================================================================
Uses a combination of:
  1. TF-IDF + Cosine Similarity  (content-based filtering)
  2. Random Forest Classifier     (trained on synthetic candidate-application data)
  3. Weighted Ensemble            (combines both scores)

Run this file directly to train and save the model:
    python ml_model.py
"""

import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import pickle
import random
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.preprocessing import LabelEncoder, MultiLabelBinarizer
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, accuracy_score
from sklearn.pipeline import Pipeline

from data.internships import (
    INTERNSHIPS, EDU_MAP, ALL_SKILLS, ALL_SECTORS, ALL_FIELDS, LOCATION_TYPES
)

# ─────────────────────────────────────────────
# 1. FEATURE ENGINEERING
# ─────────────────────────────────────────────

def candidate_to_vector(candidate: dict) -> np.ndarray:
    """
    Convert a candidate profile dict into a fixed-length numeric feature vector.
    
    Features:
    - Education level (ordinal, 0-4)
    - Field of study (one-hot, 10 categories)
    - Skills (multi-hot, 16 skills)
    - Interests / sectors (multi-hot, 12 sectors)
    - Location type preference (one-hot, 5 types)
    """
    features = []

    # Education (0-4 ordinal)
    edu_level = EDU_MAP.get(candidate.get("education", "10th"), 0)
    features.append(edu_level / 4.0)  # normalize to [0,1]

    # Field of study (one-hot)
    field = candidate.get("field", "")
    for f in ALL_FIELDS:
        features.append(1.0 if f == field else 0.0)

    # Skills (multi-hot)
    skills = set(candidate.get("skills", []))
    for s in ALL_SKILLS:
        features.append(1.0 if s in skills else 0.0)

    # Interests / sectors (multi-hot)
    interests = set(candidate.get("interests", []))
    for sec in ALL_SECTORS:
        features.append(1.0 if sec in interests else 0.0)

    # Location preference (one-hot)
    loc_map = {
        "Big City": "big_city", "Small City": "small_city",
        "Village/Rural": "village", "Work from Home": "work_from_home",
        "Anywhere": "anywhere"
    }
    loc = loc_map.get(candidate.get("location", "Anywhere"), "anywhere")
    for lt in LOCATION_TYPES:
        features.append(1.0 if lt == loc else 0.0)

    return np.array(features, dtype=np.float32)


def internship_to_vector(internship: dict) -> np.ndarray:
    """
    Convert an internship listing into a numeric feature vector 
    matching the candidate vector space for cosine similarity.
    """
    features = []

    # Education requirement (min level, normalized)
    features.append(internship.get("required_edu_min", 0) / 4.0)

    # Preferred fields (multi-hot)
    pref_fields = set(internship.get("preferred_fields", []))
    for f in ALL_FIELDS:
        features.append(1.0 if f in pref_fields else 0.0)

    # Preferred skills (multi-hot)
    pref_skills = set(internship.get("preferred_skills", []))
    for s in ALL_SKILLS:
        features.append(1.0 if s in pref_skills else 0.0)

    # Sectors (multi-hot)
    sectors = set(internship.get("sectors", []))
    for sec in ALL_SECTORS:
        features.append(1.0 if sec in sectors else 0.0)

    # Location type (multi-hot — internship may suit multiple)
    intern_loc = internship.get("location_type", "big_city")
    for lt in LOCATION_TYPES:
        if lt == "anywhere":
            features.append(1.0)  # all internships "anywhere" compatible
        else:
            features.append(1.0 if lt == intern_loc else 0.0)

    return np.array(features, dtype=np.float32)


# ─────────────────────────────────────────────
# 2. SYNTHETIC TRAINING DATA GENERATION
# ─────────────────────────────────────────────

random.seed(42)
np.random.seed(42)

def generate_synthetic_data(n_samples: int = 5000):
    """
    Generate synthetic candidate → internship application data.
    Label = 1 if the application was "successful" (good match), 0 otherwise.
    
    A match is 'good' based on:
    - Education meets or exceeds requirement
    - At least 1 skill overlap
    - At least 1 sector interest match
    - Location preference compatible
    """
    rows = []
    for _ in range(n_samples):
        # Random candidate
        edu_val = random.randint(0, 4)
        edu_options = [k for k, v in EDU_MAP.items() if v == edu_val]
        edu_key = edu_options[0] if edu_options else "10th"
        n_skills = random.randint(1, 6)
        skills = random.sample(ALL_SKILLS, n_skills)
        n_interests = random.randint(1, 3)
        interests = random.sample(ALL_SECTORS, n_interests)
        field = random.choice(ALL_FIELDS)
        loc_pref = random.choice(["Big City", "Small City", "Village/Rural", "Work from Home", "Anywhere"])

        candidate = {
            "education": edu_key,
            "field": field,
            "skills": skills,
            "interests": interests,
            "location": loc_pref
        }

        # Random internship
        intern = random.choice(INTERNSHIPS)

        # Rule-based label (ground truth for training)
        edu_ok = edu_val >= intern["required_edu_min"]
        skill_overlap = len(set(skills) & set(intern["preferred_skills"]))
        interest_overlap = len(set(interests) & set(intern["sectors"]))
        
        loc_map = {
            "Big City": "big_city", "Small City": "small_city",
            "Village/Rural": "village", "Work from Home": "work_from_home",
            "Anywhere": "anywhere"
        }
        cand_loc = loc_map.get(loc_pref, "anywhere")
        loc_ok = (cand_loc == intern["location_type"] or
                  cand_loc == "anywhere" or
                  intern["location_type"] == "anywhere")

        # Score-based labeling with some noise
        score = (
            (2 if edu_ok else -3) +
            skill_overlap * 1.5 +
            interest_overlap * 2 +
            (1 if loc_ok else -1)
        )
        # Add noise
        score += random.gauss(0, 0.5)
        label = 1 if score > 2.5 else 0

        # Build combined feature vector (candidate + internship concatenated)
        c_vec = candidate_to_vector(candidate)
        i_vec = internship_to_vector(intern)
        combined = np.concatenate([c_vec, i_vec, [intern["id"]]])

        rows.append({
            "features": combined,
            "label": label,
            "internship_id": intern["id"],
            "edu_ok": int(edu_ok),
            "skill_overlap": skill_overlap,
            "interest_overlap": interest_overlap,
            "loc_ok": int(loc_ok),
        })

    return rows


# ─────────────────────────────────────────────
# 3. TRAIN THE MODEL
# ─────────────────────────────────────────────

def train_model():
    print("=" * 60)
    print("  PM Internship ML Recommendation Engine — Training")
    print("=" * 60)

    print("\n[1/4] Generating synthetic training data...")
    data = generate_synthetic_data(n_samples=8000)
    X = np.array([d["features"] for d in data])
    y = np.array([d["label"] for d in data])
    print(f"      Dataset: {len(X)} samples | Positive rate: {y.mean():.2%}")

    print("\n[2/4] Splitting train/test (80/20)...")
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )

    print("\n[3/4] Training Gradient Boosting Classifier...")
    model = GradientBoostingClassifier(
        n_estimators=200,
        learning_rate=0.08,
        max_depth=5,
        min_samples_leaf=10,
        subsample=0.8,
        random_state=42
    )
    model.fit(X_train, y_train)

    # Evaluate
    y_pred = model.predict(X_test)
    acc = accuracy_score(y_test, y_pred)
    print(f"\n      Test Accuracy: {acc:.4f} ({acc*100:.1f}%)")
    print("\n      Classification Report:")
    print(classification_report(y_test, y_pred, target_names=["No Match", "Good Match"]))

    print("\n[4/4] Pre-computing internship vectors for cosine similarity...")
    internship_vectors = {
        intern["id"]: internship_to_vector(intern)
        for intern in INTERNSHIPS
    }

    # Feature importance
    feature_dim_c = len(candidate_to_vector({"education": "10th", "field": "", "skills": [], "interests": [], "location": "Anywhere"}))
    feature_dim_i = len(internship_to_vector(INTERNSHIPS[0]))
    importances = model.feature_importances_
    print(f"\n      Feature vector size: candidate={feature_dim_c}, internship={feature_dim_i}")
    print(f"      Top-5 feature indices (by importance): {np.argsort(importances)[-5:][::-1].tolist()}")

    # Save model artifacts
    os.makedirs("models", exist_ok=True)
    artifacts = {
        "model": model,
        "internship_vectors": internship_vectors,
        "candidate_to_vector": candidate_to_vector,
        "internship_to_vector": internship_to_vector,
        "INTERNSHIPS": INTERNSHIPS,
    }
    with open("models/recommendation_model.pkl", "wb") as f:
        pickle.dump(artifacts, f)

    print("\n✅  Model saved to models/recommendation_model.pkl")
    print("=" * 60)
    return artifacts


# ─────────────────────────────────────────────
# 4. INFERENCE: RECOMMEND INTERNSHIPS
# ─────────────────────────────────────────────

def load_model(model_path: str = "models/recommendation_model.pkl"):
    with open(model_path, "rb") as f:
        return pickle.load(f)


def recommend(candidate: dict, artifacts: dict = None, top_n: int = 5) -> list:
    """
    Given a candidate profile, return top_n ranked internship recommendations.
    
    Uses a two-stage approach:
      Stage 1: ML classifier predicts match probability for each internship
      Stage 2: Cosine similarity re-ranks using content features
    Final score = 0.65 * ML_prob + 0.35 * cosine_sim
    
    Args:
        candidate: dict with keys: education, field, skills, interests, location
        artifacts: pre-loaded model artifacts (optional, loads from disk if None)
        top_n: number of recommendations to return
    
    Returns:
        List of dicts with internship details + match_score + match_pct + reason
    """
    if artifacts is None:
        artifacts = load_model()

    model = artifacts["model"]
    internship_vectors = artifacts["internship_vectors"]
    internships = artifacts["INTERNSHIPS"]

    c_vec = candidate_to_vector(candidate)

    results = []
    for intern in internships:
        i_vec = internship_vectors[intern["id"]]

        # Stage 1: ML probability score
        combined = np.concatenate([c_vec, i_vec, [intern["id"]]])
        ml_prob = model.predict_proba(combined.reshape(1, -1))[0][1]  # P(good match)

        # Stage 2: Cosine similarity (content-based)
        cos_sim = float(cosine_similarity(c_vec.reshape(1, -1), i_vec.reshape(1, -1))[0][0])

        # Ensemble score
        final_score = 0.65 * ml_prob + 0.35 * cos_sim

        # Hard filter: education must meet minimum
        edu_level = EDU_MAP.get(candidate.get("education", "10th"), 0)
        if edu_level < intern["required_edu_min"]:
            final_score *= 0.3  # heavily penalize but don't exclude

        # Build explanation
        skill_matches = list(set(candidate.get("skills", [])) & set(intern["preferred_skills"]))
        interest_matches = list(set(candidate.get("interests", [])) & set(intern["sectors"]))

        reasons = []
        if skill_matches:
            reasons.append(f"Skills match: {', '.join(skill_matches[:3])}")
        if interest_matches:
            reasons.append(f"Sector interest: {', '.join(interest_matches)}")
        if candidate.get("field") in intern.get("preferred_fields", []):
            reasons.append(f"Field of study ({candidate['field']}) is preferred")
        loc_map = {"Big City": "big_city", "Small City": "small_city",
                   "Village/Rural": "village", "Work from Home": "work_from_home", "Anywhere": "anywhere"}
        if loc_map.get(candidate.get("location", "Anywhere")) == intern["location_type"]:
            reasons.append("Location matches your preference")
        if not reasons:
            reasons.append("Good baseline match based on your profile")

        results.append({
            **intern,
            "ml_probability": round(ml_prob, 4),
            "cosine_similarity": round(cos_sim, 4),
            "final_score": round(final_score, 4),
            "match_pct": min(97, int(final_score * 100) + 10),  # display score (10% boost for UX)
            "match_reason": " | ".join(reasons),
            "skill_matches": skill_matches,
            "interest_matches": interest_matches,
        })

    # Sort by final score, take top N
    results.sort(key=lambda x: x["final_score"], reverse=True)
    return results[:top_n]


# ─────────────────────────────────────────────
# 5. MAIN
# ─────────────────────────────────────────────

if __name__ == "__main__":
    artifacts = train_model()

    print("\n\n🔍 Sample Prediction Test")
    print("-" * 40)
    test_candidate = {
        "education": "ITI",
        "field": "Engineering",
        "skills": ["Electrical Work", "Teamwork", "Mechanical Work"],
        "interests": ["Energy & Power", "Manufacturing"],
        "location": "Small City"
    }
    print(f"Candidate: {test_candidate}\n")
    recs = recommend(test_candidate, artifacts, top_n=5)
    for i, r in enumerate(recs, 1):
        print(f"  #{i}: {r['title']} @ {r['company']}")
        print(f"       ML P={r['ml_probability']:.3f} | CosSim={r['cosine_similarity']:.3f} | Final={r['final_score']:.3f} ({r['match_pct']}%)")
        print(f"       Reason: {r['match_reason']}\n")
