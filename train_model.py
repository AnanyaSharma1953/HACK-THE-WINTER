import pandas as pd
import pickle
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split

# ---------------- Load Kaggle Dataset ----------------
fake = pd.read_csv("data/Fake.csv")
real = pd.read_csv("data/True.csv")


fake["label"] = "FAKE"
real["label"] = "REAL"

# Combine text + title
fake["content"] = fake["title"] + " " + fake["text"]
real["content"] = real["title"] + " " + real["text"]

data = pd.concat([fake, real])
data = data.sample(frac=1).reset_index(drop=True)  # shuffle

X = data["content"]
y = data["label"]

# ---------------- Train-Test Split ----------------
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

# ---------------- Vectorization ----------------
vectorizer = TfidfVectorizer(
    stop_words="english",
    max_df=0.7
)

X_train_vec = vectorizer.fit_transform(X_train)
X_test_vec = vectorizer.transform(X_test)

# ---------------- Model ----------------
model = LogisticRegression(max_iter=1000)
model.fit(X_train_vec, y_train)

# ---------------- Accuracy ----------------
accuracy = model.score(X_test_vec, y_test)
print(f"✅ Model trained with accuracy: {accuracy:.2f}")

# ---------------- Save ----------------
pickle.dump(model, open("model.pkl", "wb"))
pickle.dump(vectorizer, open("vectorizer.pkl", "wb"))

print("✅ model.pkl & vectorizer.pkl saved")

