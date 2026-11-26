from pathlib import Path
import re
import pandas as pd
from flask import Flask, request, jsonify
from flask_cors import CORS
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline

ROOT = Path(__file__).resolve().parent
CSV_PATH = ROOT / "pds_nlp_1000_records_cleaned (1).csv"
app = Flask(__name__)
CORS(app)
STOPWORDS = set("a an the is are was were am be been being to of and or in on at for from with my i we our it this that through by as after before when while could would should have has had do did does not no very only but about there".split())

def preprocess(text):
    text = re.sub(r"[^a-z\s]", " ", str(text).lower())
    return " ".join(w for w in text.split() if w not in STOPWORDS and len(w) > 1)

def load_data():
    if not CSV_PATH.exists():
        raise FileNotFoundError(f"Missing {CSV_PATH}")
    df = pd.read_csv(CSV_PATH)
    cols = {str(c).strip().lower(): c for c in df.columns}
    aliases = {
        "original_text": ["original_text", "complaint", "text", "complaint_text"],
        "category": ["category", "issue_category", "label"],
        "sentiment": ["sentiment", "sentiment_label", "polarity"]
    }
    rename = {}
    for target, choices in aliases.items():
        source = next((cols[x] for x in choices if x in cols), None)
        if source is None:
            raise ValueError(f"CSV must include a column for {target}. Current columns: {list(df.columns)}")
        rename[source] = target
    df = df.rename(columns=rename)[["original_text", "category", "sentiment"]].dropna()
    for c in ["original_text", "category", "sentiment"]:
        df[c] = df[c].astype(str).str.strip()
    df = df[(df.original_text != "") & (df.category != "") & (df.sentiment != "")]
    df = df.drop_duplicates(subset=["original_text", "category", "sentiment"]).reset_index(drop=True)
    df["processed_text"] = df.original_text.map(preprocess)
    df = df[df.processed_text.str.strip() != ""].reset_index(drop=True)
    if len(df) < 4:
        raise ValueError("Please provide at least 4 labeled rows in the CSV.")
    return df

df = load_data()
def make_model():
    return Pipeline([("tfidf", TfidfVectorizer(ngram_range=(1,2), sublinear_tf=True)),
                     ("model", LogisticRegression(max_iter=2000, class_weight="balanced"))])
category_model, sentiment_model = make_model(), make_model()
category_model.fit(df.processed_text, df.category)
sentiment_model.fit(df.processed_text, df.sentiment)

def confidence(model, text):
    return round(float(max(model.predict_proba([text])[0])) * 100, 2)

@app.get("/api/health")
def health():
    return jsonify({"status":"ok","message":"PDS NLP backend is running","records":int(len(df))})

@app.get("/api/stats")
def stats():
    return jsonify({"total_records":int(len(df)),
                    "categories":{str(k):int(v) for k,v in df.category.value_counts().items()},
                    "sentiments":{str(k):int(v) for k,v in df.sentiment.value_counts().items()}})

@app.post("/api/predict")
def predict():
    body = request.get_json(silent=True) or {}
    original = str(body.get("text","")).strip()
    if not original:
        return jsonify({"error":"Please enter a complaint."}), 400
    processed = preprocess(original)
    if not processed:
        return jsonify({"error":"Please enter a complaint containing meaningful words."}), 400
    cat = category_model.predict([processed])[0]
    sent = sentiment_model.predict([processed])[0]
    return jsonify({"original_text":original,"processed_text":processed,"category":str(cat),
                    "sentiment":str(sent),"category_confidence":confidence(category_model,processed),
                    "sentiment_confidence":confidence(sentiment_model,processed)})

if __name__ == "__main__":
    print(f"PDS NLP API running at http://127.0.0.1:5000 with {len(df)} records")
    app.run(host="127.0.0.1", port=5000, debug=True)
