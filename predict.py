import re
import joblib
import nltk
from nltk.corpus import stopwords

nltk.download('stopwords')

model = joblib.load('model/model.pkl')
vectorizer = joblib.load('model/vec.pkl')

stop_words = set(stopwords.words('english'))

def clean_text(text):
    text = text.lower()
    text = re.sub(r'[^a-z\s]', '', text)
    return text

def preprocess(text):
    words = text.split()
    words = [word for word in words if word not in stop_words]
    return ' '.join(words)

def keyword_score(text):
    reliable_keywords = {
        "confirmed", "official", "research", "report", "study", "government", "investigation",
        "statement", "authority", "scientific", "data", "evidence", "analysis", "court",
        "agency", "inquiry", "published", "verified", "statistics", "documentation",
        "testimony", "witness", "disclosed", "update", "breaking", "journal", "peer-reviewed",
        "source", "newswire", "announcement", "white house", "nasa", "cdc", "fbi",
        "department", "institute", "organization", "academic", "conference", "transcript",
        "health authority", "press release", "officials", "UN", "WHO", "reporter", "reuters", "bbc"
    }

    suspect_keywords = {
        "shocking", "secret", "banned", "miracle", "hoax", "exposed", "unbelievable", "you won't believe",
        "click here", "breaking news", "explosive", "outrageous", "hidden truth", "alien", "cancer cure",
        "fake", "conspiracy", "agenda", "mainstream media", "urgent", "alert", "this is why", "no one told you",
        "deep state", "plandemic", "rigged", "stolen", "leaked", "cover up", "scandal", "vaccine danger",
        "woke", "soros", "illuminati", "truther", "massive", "must watch", "spreads like wildfire", "5g",
        "new world order", "mind control", "wake up", "exposing", "truth bomb", "they don't want you to know"
    }

    text = text.lower()
    reliable_count = sum(1 for word in reliable_keywords if word in text)
    suspect_count = sum(1 for word in suspect_keywords if word in text)
    return reliable_count - suspect_count

def predict_news(text):
    cleaned = clean_text(text)
    preprocessed = preprocess(cleaned)
    vec = vectorizer.transform([preprocessed])
    model_prediction = model.predict(vec)[0]
    score = keyword_score(text)

    label = "Real" if model_prediction == 1 else "Fake"
    confidence = f"Keyword score: {score}"

    return {
        "prediction": label,
        "confidence": confidence,
        "text": text
    }
