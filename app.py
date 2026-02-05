import streamlit as st
import joblib
import re
import nltk
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer

# ===============================
# PAGE CONFIG
# ===============================
st.set_page_config(
    page_title="Flipkart Sentiment Analyzer",
    page_icon="🛒",
    layout="centered"
)

# ===============================
# DOWNLOAD NLTK (ONCE)
# ===============================
@st.cache_resource
def load_nltk():
    nltk.download("stopwords")
    nltk.download("wordnet")

load_nltk()

# ===============================
# LOAD MODEL + VECTORIZER
# ===============================
@st.cache_resource
def load_model():
    model = joblib.load("sentiment_model.pkl")
    vectorizer = joblib.load("tfidf_vectorizer.pkl")
    return model, vectorizer

model, vectorizer = load_model()

# ===============================
# TEXT PREPROCESSING
# ===============================
stop_words = set(stopwords.words("english"))
lemmatizer = WordNetLemmatizer()

def clean_text(text):
    text = str(text).lower()
    text = re.sub(r"[^a-zA-Z]", " ", text)
    words = text.split()
    words = [w for w in words if w not in stop_words]
    words = [lemmatizer.lemmatize(w) for w in words]
    return " ".join(words)

# ===============================
# CUSTOM CSS (PROFESSIONAL DARK GLASS)
# ===============================
st.markdown("""
<style>
body {
    background: linear-gradient(135deg, #0f2027, #203a43, #2c5364);
}

/* CENTER */
.main .block-container {
    max-width: 850px;
    margin: auto;
    padding-top: 60px;
}

/* GLASS CARD */
.glass-card {
    background: rgba(18, 25, 40, 0.92);
    backdrop-filter: blur(18px);
    border-radius: 26px;
    padding: 40px;
    box-shadow: 0 25px 60px rgba(0,0,0,0.45);
}

/* TITLE */
.title {
    text-align: center;
    font-size: 42px;
    font-weight: 900;
    color: #38bdf8;
}

/* SUBTITLE */
.subtitle {
    text-align: center;
    font-size: 17px;
    color: #94a3b8;
    margin-bottom: 30px;
}

/* TEXT AREA */
.stTextArea textarea {
    background: #020617;
    color: #e5e7eb;
    border-radius: 16px;
    border: 2px solid #334155;
    padding: 14px;
}

.stTextArea textarea:focus {
    border-color: #38bdf8;
    box-shadow: 0 0 0 3px rgba(56,189,248,0.3);
}

/* BUTTON */
.stButton button {
    width: 100%;
    border-radius: 16px;
    padding: 14px;
    font-size: 18px;
    font-weight: 700;
    background: linear-gradient(90deg, #38bdf8, #22d3ee);
    color: #020617;
    border: none;
}

.stButton button:hover {
    transform: scale(1.04);
    box-shadow: 0 0 30px rgba(34,211,238,0.6);
}

/* RESULT */
.result {
    margin-top: 25px;
    padding: 22px;
    border-radius: 18px;
    text-align: center;
    font-size: 26px;
    font-weight: 800;
}

/* POSITIVE */
.positive {
    background: rgba(16, 185, 129, 0.15);
    color: #10b981;
    border-left: 6px solid #10b981;
}

/* NEGATIVE */
.negative {
    background: rgba(239, 68, 68, 0.15);
    color: #ef4444;
    border-left: 6px solid #ef4444;
}

/* CONFIDENCE */
.confidence {
    text-align: center;
    margin-top: 10px;
    font-weight: 600;
    color: #7dd3fc;
}

/* FOOTER */
.footer {
    text-align: center;
    margin-top: 40px;
    font-size: 14px;
    color: #94a3b8;
}
</style>
""", unsafe_allow_html=True)

# ===============================
# UI
# ===============================
st.markdown("<div class='glass-card'>", unsafe_allow_html=True)

st.markdown("<div class='title'>🛒 Flipkart Review Sentiment Analysis</div>", unsafe_allow_html=True)
st.markdown("<div class='subtitle'>Analyze customer emotions using Machine Learning</div>", unsafe_allow_html=True)

review = st.text_area(
    "✍️ Enter Product Review",
    height=150,
    placeholder="Example: The product quality exceeded my expectations!"
)

predict = st.button("⚡ Analyze Sentiment")

if predict:
    if review.strip() == "":
        st.warning("⚠️ Please enter a review")
    else:
        with st.spinner("Analyzing sentiment..."):
            cleaned = clean_text(review)
            vec = vectorizer.transform([cleaned])
            prediction = model.predict(vec)[0]
            

        if prediction == 1:
            label = "😊 Positive Review"
            css_class = "positive"
        else:
            label = "😞 Negative Review"
            css_class = "negative"

        st.markdown(
            f"<div class='result {css_class}'>{label}</div>",
            unsafe_allow_html=True
        )

        st.progress(int(confidence))
        st.markdown(
            f"<div class='confidence'>Confidence: {confidence:.2f}%</div>",
            unsafe_allow_html=True
        )

st.markdown("</div>", unsafe_allow_html=True)

st.markdown(
    "<div class='footer'>Built with 💙 using Machine Learning & Streamlit</div>",
    unsafe_allow_html=True
)

