import streamlit as st
import pickle
import re
import pandas as pd

# Page config
st.set_page_config(
    page_title="AI Fake News Detector",
    page_icon="🧠",
    layout="wide"
)

# Load model
model = pickle.load(open("model.pkl", "rb"))
vectorizer = pickle.load(open("vectorizer.pkl", "rb"))

# ---------------- CSS ----------------
st.markdown("""
<style>
.big-title {
    font-size: 42px;
    font-weight: 700;
    color: #2E86C1;
}
.sub-title {
    font-size: 18px;
    color: #555;
}
.card {
    padding: 20px;
    border-radius: 15px;
    background-color: #F4F6F7;
    box-shadow: 0px 4px 10px rgba(0,0,0,0.1);
}
</style>
""", unsafe_allow_html=True)

# ---------------- Header ----------------
st.markdown('<div class="big-title">📰 AI Fake News & Bot Detection</div>', unsafe_allow_html=True)
st.markdown('<div class="sub-title">Hackathon Prototype | NLP + Machine Learning</div>', unsafe_allow_html=True)
st.write("")

# ---------------- Layout ----------------
col1, col2 = st.columns([2, 1])

# ---------------- Input ----------------
with col1:
    st.markdown("### ✍️ Enter News / Comment")
    text_input = st.text_area(
        "Type or paste text below:",
        height=160,
        placeholder="Example: Drinking salt water cures cancer..."
    )

    st.markdown("### 📎 Or Upload File")
    uploaded_file = st.file_uploader(
        "Upload CSV or TXT file",
        type=["csv", "txt"]
    )

# ---------------- Info Panel ----------------
with col2:
    st.markdown("### ℹ️ How it Works")
    st.info("""
    • TF-IDF + Logistic Regression  
    • Fake vs Real News Detection  
    • Bot vs Human Detection  
    • Confidence Score
    """)

# ---------------- Helpers ----------------
def clean_text(text):
    text = re.sub(r"http\S+", "", text)
    text = re.sub(r"[^a-zA-Z ]", "", text)
    return text.lower()

def analyze_text(text):
    clean = clean_text(text)
    vec = vectorizer.transform([clean])
    prediction = model.predict(vec)[0]
    confidence = model.predict_proba(vec).max() * 100
    return prediction, confidence

# ---------------- Analyze ----------------
st.write("")
if st.button("🔍 Analyze Content", use_container_width=True):

    final_text = ""

    # ---- If file uploaded ----
    if uploaded_file is not None:
        if uploaded_file.name.endswith(".txt"):
            final_text = uploaded_file.read().decode("utf-8")

        elif uploaded_file.name.endswith(".csv"):
            df = pd.read_csv(uploaded_file)
            st.write("📄 Uploaded CSV Preview:")
            st.dataframe(df.head())

            final_text = df.iloc[0, 0]  # first cell text

    # ---- Else manual input ----
    else:
        final_text = text_input

    if final_text.strip() == "":
        st.warning("⚠️ Please provide text or upload a file.")
    else:
        prediction, confidence = analyze_text(final_text)

        st.markdown("## 📊 Analysis Result")
        r1, r2 = st.columns(2)

        # -------- Fake News --------
        with r1:
            st.markdown('<div class="card">', unsafe_allow_html=True)
            st.markdown("### 🧠 News Authenticity")

            if prediction == "FAKE":
                st.error("❌ **FAKE NEWS DETECTED**")
            else:
                st.success("✅ **REAL NEWS DETECTED**")

            st.write(f"**Confidence:** {confidence:.2f}%")
            st.progress(int(confidence))
            st.markdown('</div>', unsafe_allow_html=True)

        # -------- Bot Detection --------
        with r2:
            st.markdown('<div class="card">', unsafe_allow_html=True)
            st.markdown("### 🤖 Bot Detection")

            if len(final_text.split()) < 5 or final_text.isupper():
                st.warning("⚠️ Likely Bot-Generated Content")
                st.write("Reason: Very short text or excessive capitalization.")
            else:
                st.success("👤 Likely Human-Generated Content")
                st.write("Reason: Natural sentence structure.")

            st.markdown('</div>', unsafe_allow_html=True)

# ---------------- Footer ----------------
st.markdown("---")
st.caption("🚀 DEVELOPED BY TEAM CodingWithBiriyani ")
