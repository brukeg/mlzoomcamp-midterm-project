import pickle
from flask import Flask, request, jsonify


MODEL_FILE = "model_decision_tree.bin"

with open(MODEL_FILE, "rb") as f_in:
    dv, model = pickle.load(f_in)

app = Flask(__name__)

REQUIRED_KEYS = [
    "age",
    "gender",
    "education_level",
    "job_title",
    "years_of_experience",
]

def _extract_allowed_categories(dv, prefix):
    """helper function to derive allowed categories from DictVectorizer"""
    feats = dv.get_feature_names_out()
    values = set()
    for f in feats:
        if f.startswith(prefix + "="):
            values.add(f.split("=", 1)[1])
    return values

ALLOWED = {
    "gender": _extract_allowed_categories(dv, "gender"),
    "education_level": {"bachelors", "masters", "phd"},
    "job_title": _extract_allowed_categories(dv, "job_title"),
}

def _valid_age(x):
    """age must be within the observed observed range of the data"""
    return isinstance(x, (int, float)) and 23 <= x <= 53

def _valid_yoe(x):
    """years of experience must be within the observed range of the data"""
    return isinstance(x, (int, float)) and 0 <= x <= 25

@app.route("/predict", methods=["POST"])
def predict():
    """Validate input JSON, transform with DictVectorizer, and return a prediction."""
    payload = request.get_json()

    if payload is None or not isinstance(payload, dict):
        return jsonify({"error": "Invalid or missing JSON"}), 400

    missing = [k for k in REQUIRED_KEYS if k not in payload]
    if missing:
        return jsonify({"error": f"Missing required keys: {missing}"}), 400

    # normalize strings
    record = {}
    for k in REQUIRED_KEYS:
        v = payload[k]
        if isinstance(v, str):
            v = v.lower().replace(" ", "_")
        record[k] = v

    # numeric validation
    if not _valid_age(record["age"]):
        return jsonify({"error": "age must be between 23 and 53"}), 422
    if not _valid_yoe(record["years_of_experience"]):
        return jsonify({"error": "years_of_experience must be between 0 and 25"}), 422
    if record["years_of_experience"] > record["age"] - 18:
        return jsonify({"error": "years_of_experience is inconsistent with age (too high for given age)"}), 422

    # categorical validation
    cat_errors = {}

    # strict fixed validation for education_level
    if record["education_level"] not in ALLOWED["education_level"]:
        cat_errors["education_level"] = {
            "received": record["education_level"],
            "allowed_values": sorted(list(ALLOWED["education_level"])),
            "message": "education_level must be one of: bachelors, masters, or phd",
        }

    # validation for gender and job_title based on DictVectorizer
    for field in ("gender", "job_title"):
        allowed = ALLOWED[field]
        if record[field] not in allowed:
            cat_errors[field] = {
                "received": record[field],
                "allowed_sample": sorted(list(allowed))[:30],
                "message": f"{field} value not seen in training data",
            }

    if cat_errors:
        return jsonify({"error": "invalid categorical values", "details": cat_errors}), 422

    # transform & predict
    X = dv.transform([record])
    pred = float(model.predict(X)[0])

    return jsonify({"salary_prediction": pred})
