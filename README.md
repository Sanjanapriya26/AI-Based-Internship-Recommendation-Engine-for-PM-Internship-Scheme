# PM Internship — ML Recommendation Engine
### AI-Powered Internship Matching for the PM Internship Scheme

---

## 🧠 ML Architecture

```
Candidate Profile → Feature Vector (44 dims)
                           ↓
          ┌────────────────┴────────────────┐
          │                                 │
  GradientBoosting             Cosine Similarity
  Classifier (sklearn)      (Content-Based Filtering)
  Trained on 8,000            Candidate vector vs
  synthetic samples           Internship vectors
          │                                 │
          └──────── Weighted Ensemble ──────┘
                  65% ML + 35% Content
                           ↓
               Ranked Recommendations (Top-5)
```

**Model Performance:** `88.6% Test Accuracy` | F1-Score: 0.89

---

## 📁 Project Structure

```
pm_internship/
├── app.py                    # Flask REST API server
├── ml_model.py               # ML model (training + inference)
├── requirements.txt          # Python dependencies
├── data/
│   ├── __init__.py
│   └── internships.py        # Dataset: 15 internship listings + feature maps
├── models/
│   └── recommendation_model.pkl   # Saved trained model
└── static/
    └── index.html            # Frontend UI (connects to Flask API)
```

---

## ⚡ Quick Start

### 1. Install dependencies
```bash
pip install -r requirements.txt
```

### 2. Train the ML model (first time only)
```bash
python ml_model.py
```
This will:
- Generate 8,000 synthetic candidate-application training samples
- Train a Gradient Boosting Classifier
- Pre-compute TF-IDF cosine similarity vectors for all internships
- Save the model to `models/recommendation_model.pkl`
- Print accuracy report and a sample prediction

### 3. Start the Flask server
```bash
python app.py
```
Open your browser: **http://localhost:5000**

---

## 🔌 API Reference

### `POST /api/recommend`
Get ML-powered internship recommendations for a candidate.

**Request Body:**
```json
{
  "name": "Rahul Kumar",
  "education": "ITI",
  "field": "Engineering",
  "skills": ["Electrical Work", "Teamwork"],
  "interests": ["Energy & Power", "Manufacturing"],
  "location": "Small City",
  "state": "Maharashtra",
  "top_n": 5
}
```

**Education values:** `10th` | `12th` | `ITI` | `Graduate` | `PostGraduate`
**Location values:** `Big City` | `Small City` | `Village/Rural` | `Work from Home` | `Anywhere`

**Response:**
```json
{
  "success": true,
  "candidate": { ... },
  "recommendations": [
    {
      "rank": 1,
      "title": "Solar Panel Technician Trainee",
      "company": "Adani Green Energy",
      "match_pct": 97,
      "ml_probability": 0.998,
      "cosine_similarity": 0.885,
      "final_score": 0.958,
      "match_reason": "Skills match: Electrical Work, Teamwork | Sector: Energy & Power",
      "skill_matches": ["Electrical Work", "Teamwork"],
      "interest_matches": ["Energy & Power"],
      ...
    }
  ],
  "model_info": {
    "type": "GradientBoosting + CosineSimilarity Ensemble",
    "weights": { "ml_model": 0.65, "content_similarity": 0.35 }
  }
}
```

### `GET /api/health`
```json
{ "status": "ok", "model_loaded": true, "internships_count": 15 }
```

### `GET /api/internships`
Returns all 15 internship listings.

### `POST /api/train`
Retrains the model on demand (no body required).

---

## 🧪 Feature Engineering

Each candidate and internship is encoded into a **44-dimensional feature vector**:

| Feature Group        | Dimensions | Description                          |
|---------------------|-----------|--------------------------------------|
| Education Level      | 1          | Ordinal (0=10th, 4=PostGrad)         |
| Field of Study       | 10         | One-hot (Arts, Engineering, etc.)    |
| Skills               | 16         | Multi-hot (Computer Basics, etc.)    |
| Sector Interests     | 12         | Multi-hot (IT, Healthcare, etc.)     |
| Location Preference  | 5          | One-hot (Big City, Rural, etc.)      |
| **Total**           | **44**     |                                      |

---

## 🔄 How the ML Model Works

1. **Data Generation:** 8,000 synthetic candidate × internship pairs are created with rule-based success labels (education fit + skill overlap + interest match + location compatibility)

2. **Model Training:** `GradientBoostingClassifier` (200 estimators, lr=0.08, max_depth=5) is trained to predict P(good match)

3. **Inference:**
   - For each of the 15 internships, compute:
     - `ml_prob` = model.predict_proba(candidate_vec + internship_vec)[1]
     - `cos_sim` = cosine_similarity(candidate_vec, internship_vec)
     - `final_score` = 0.65 × ml_prob + 0.35 × cos_sim
   - Apply education hard-filter penalty
   - Return top-5 sorted by final_score

---

## 🌐 Tech Stack

| Layer      | Technology                            |
|-----------|---------------------------------------|
| Backend   | Python 3.x + Flask                    |
| ML Model  | scikit-learn (GradientBoosting)       |
| Similarity| NumPy cosine similarity               |
| Frontend  | Vanilla HTML/CSS/JS                   |
| Data      | Pandas + NumPy                        |
| Serialization | Pickle                           |

---

## 📈 Extending the System

- **More data:** Replace synthetic data with real application logs from the PM Internship portal
- **More internships:** Add entries to `data/internships.py`
- **Better model:** Switch to XGBoost, LightGBM, or a neural network
- **Language support:** Add Hindi NLP preprocessing for regional language inputs
- **Authentication:** Add user login to save profiles and track applications
