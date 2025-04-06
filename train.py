import pandas as pd
import numpy as np
import re
import csv
import gdown
import nltk
from nltk.corpus import stopwords
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.ensemble import RandomForestClassifier
import joblib

nltk.download('stopwords')

fake_file_id = 'c4Lw_QThx4fOvUXWYKgtYIoCsuBl_mVCVbi'
true1_file_id = 'QThx4fU5UXWcNJYIoCsuBlnjdnsfki_fs2'
true2_file_id = 'fhadjUHJN32UXW4YKgtYIoCsuBl_mVCVbi'

gdown.download(f'https://drive.google.com/uc?id={c4Lw_QThx4fOvUXWYKgtYIoCsuBl_mVCVbi}', 'datasets/Fake.csv', quiet=False)
gdown.download(f'https://drive.google.com/uc?id={QThx4fU5UXWcNJYIoCsuBlnjdnsfki_fs2}', 'datasets/True.csv', quiet=False)
gdown.download(f'https://drive.google.com/uc?id={fhadjUHJN32UXW4YKgtYIoCsuBl_mVCVbi}', 'datasets/data.csv', quiet=False)

fake_df = pd.read_csv('datasets/Fake.csv', quoting=csv.QUOTE_NONE, on_bad_lines='skip')[['title', 'text']]
true_df1 = pd.read_csv('datasets/True.csv', quoting=csv.QUOTE_NONE, on_bad_lines='skip')[['title', 'text']]
true_df2 = pd.read_csv('datasets/data.csv', quoting=csv.QUOTE_NONE, on_bad_lines='skip')
true_df2 = true_df2.rename(columns={'full_content': 'text'})[['title', 'text']]


fake_df['label'] = 0
true_df1['label'] = 1
true_df2['label'] = 1

real_df = pd.concat([true_df1, true_df2], ignore_index=True)
min_len = min(len(fake_df), len(real_df))
fake_df = fake_df.sample(min_len, random_state=42)
real_df = real_df.sample(min_len, random_state=42)

df = pd.concat([fake_df, real_df], ignore_index=True).sample(frac=1, random_state=42)
df['content'] = df['title'] + " " + df['text']
df.dropna(subset=['content'], inplace=True)

# Text cleaning
def clean_text(text):
    text = text.lower()
    text = re.sub(r'[^a-z\s]', '', text)
    return text

stop_words = set(stopwords.words('english'))
def preprocess(text):
    words = text.split()
    return ' '.join([word for word in words if word not in stop_words])

df['cleaned'] = df['content'].apply(clean_text).apply(preprocess)

# TF-IDF and model
vectorizer = TfidfVectorizer(max_df=0.7)
X = vectorizer.fit_transform(df['cleaned'])
y = df['label']

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)
model = RandomForestClassifier(n_estimators=100, random_state=42)
model.fit(X_train, y_train)

# Save model + vectorizer
joblib.dump(model, 'model/model.pkl')
joblib.dump(vectorizer, 'model/vec.pkl')
print("✅ Model and vectorizer saved in /model/")
