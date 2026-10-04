import streamlit as st
import nltk
import re
from collections import Counter
import heapq

# Download required NLTK data
nltk.download('punkt', quiet=True)
nltk.download('stopwords', quiet=True)
nltk.download('punkt_tab', quiet=True)

from nltk.tokenize import sent_tokenize, word_tokenize
from nltk.corpus import stopwords

# ─── Page config ──────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="AI-Based Text Summarization — NLP Summarizer",
    page_icon="📄",
    layout="centered"
)

# ─── Custom CSS ───────────────────────────────────────────────────────────────
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=DM+Serif+Display:ital@0;1&family=Outfit:wght@300;400;500;600&family=DM+Mono:wght@400;500&display=swap');

html, body, [class*="css"] {
    font-family: 'Outfit', sans-serif;
}

.stApp {
    background-color: #0e0f11;
    color: #e8e9eb;
}

/* Header */
.main-title {
    font-family: 'DM Serif Display', serif;
    font-size: 54px;
    color: #e8e9eb;
    line-height: 1.1;
    margin-bottom: 6px;
}

.main-title em {
    font-style: italic;
    color: #c8f060;
}

.chip {
    display: inline-block;
    background: rgba(200,240,96,0.08);
    border: 1px solid rgba(200,240,96,0.25);
    border-radius: 100px;
    padding: 4px 14px;
    font-size: 11px;
    font-family: 'DM Mono', monospace;
    color: #c8f060;
    letter-spacing: 0.1em;
    text-transform: uppercase;
    margin-bottom: 16px;
}

.subtitle {
    color: #6b6f7a;
    font-size: 15px;
    font-weight: 300;
    margin-bottom: 36px;
}

/* Summary output box */
.summary-box {
    background: #16181c;
    border: 1px solid #2a2d35;
    border-left: 3px solid #c8f060;
    border-radius: 14px;
    padding: 22px 24px;
    font-size: 15px;
    font-weight: 300;
    line-height: 1.85;
    color: #e8e9eb;
    margin-top: 8px;
}

/* Stats cards */
.stats-row {
    display: flex;
    gap: 12px;
    margin-top: 16px;
    flex-wrap: wrap;
}

.stat-card {
    background: #16181c;
    border: 1px solid #2a2d35;
    border-radius: 12px;
    padding: 14px 18px;
    flex: 1;
    min-width: 100px;
    text-align: center;
}

.stat-val {
    font-family: 'DM Serif Display', serif;
    font-size: 28px;
    color: #c8f060;
    line-height: 1;
    margin-bottom: 4px;
}

.stat-key {
    font-size: 11px;
    color: #6b6f7a;
    text-transform: uppercase;
    letter-spacing: 0.08em;
    font-family: 'DM Mono', monospace;
}

/* Section label */
.section-label {
    font-family: 'DM Mono', monospace;
    font-size: 11px;
    text-transform: uppercase;
    letter-spacing: 0.1em;
    color: #c8f060;
    margin-bottom: 8px;
}

/* Divider */
.divider {
    border: none;
    border-top: 1px solid #2a2d35;
    margin: 28px 0;
}

/* Footer */
.footer {
    text-align: center;
    font-family: 'DM Mono', monospace;
    font-size: 12px;
    color: #6b6f7a;
    margin-top: 48px;
}

.footer span { color: #c8f060; }

/* Streamlit overrides */
.stTextArea textarea {
    background-color: #16181c !important;
    border: 1px solid #2a2d35 !important;
    border-radius: 14px !important;
    color: #e8e9eb !important;
    font-family: 'Outfit', sans-serif !important;
    font-size: 15px !important;
    font-weight: 300 !important;
    line-height: 1.75 !important;
    caret-color: #c8f060 !important;
}

.stTextArea textarea:focus {
    border-color: rgba(200,240,96,0.4) !important;
    box-shadow: 0 0 0 1px rgba(200,240,96,0.15) !important;
}

.stButton > button {
    background-color: #c8f060 !important;
    color: #0e0f11 !important;
    border: none !important;
    border-radius: 100px !important;
    font-family: 'Outfit', sans-serif !important;
    font-weight: 500 !important;
    font-size: 15px !important;
    padding: 10px 32px !important;
    transition: opacity 0.2s !important;
    width: 100%;
}

.stButton > button:hover {
    opacity: 0.85 !important;
    color: #0e0f11 !important;
}

.stSlider > div > div {
    color: #c8f060 !important;
}

[data-testid="stSlider"] label {
    color: #6b6f7a !important;
    font-family: 'DM Mono', monospace !important;
    font-size: 13px !important;
}

.stAlert {
    background-color: rgba(255,95,87,0.08) !important;
    border: 1px solid rgba(255,95,87,0.2) !important;
    border-radius: 12px !important;
    color: #ff5f57 !important;
}
</style>
""", unsafe_allow_html=True)


# ─── Summarizer logic ─────────────────────────────────────────────────────────
def summarize_text(text: str, num_sentences: int) -> dict:

    stop_words = set(stopwords.words('english'))

    # Tokenize into sentences
    sentences = sent_tokenize(text)
    if len(sentences) < 2:
        return None

    num_sentences = min(num_sentences, len(sentences))

    # Clean and tokenize words
    clean_words = []
    for sent in sentences:
        words = word_tokenize(sent.lower())
        words = [w for w in words if w.isalpha() and w not in stop_words]
        clean_words.extend(words)

    # Word frequency count (normalized)
    word_freq = Counter(clean_words)
    max_freq = max(word_freq.values(), default=1)
    word_freq = {w: freq / max_freq for w, freq in word_freq.items()}

    # Score each sentence
    sentence_scores = {}
    for i, sent in enumerate(sentences):
        words = word_tokenize(sent.lower())
        words = [w for w in words if w.isalpha() and w not in stop_words]
        if words:
            score = sum(word_freq.get(w, 0) for w in words) / len(words)
            # Small bonus for the first sentence (often carries context)
            if i == 0:
                score *= 1.2
            sentence_scores[i] = score

    # Get top N sentence indices, sorted back to original order
    top_indices = heapq.nlargest(num_sentences, sentence_scores, key=sentence_scores.get)
    top_indices = sorted(top_indices)

    summary = ' '.join(sentences[i] for i in top_indices)

    orig_words  = len(text.split())
    summ_words  = len(summary.split())
    reduction   = round((1 - summ_words / orig_words) * 100) if orig_words else 0

    return {
        'summary':    summary,
        'sentences':  len(sentences),
        'orig_words': orig_words,
        'summ_words': summ_words,
        'reduction':  reduction,
        'kept':       len(top_indices),
    }


# ─── UI Layout ────────────────────────────────────────────────────────────────
st.markdown('<div class="chip">⬡ NLP · Extractive</div>', unsafe_allow_html=True)
st.markdown('<div class="main-title">AI-Based<em>Text Summarizer</em></div>', unsafe_allow_html=True)
st.markdown('<div class="subtitle">Paste any long text and get a clean, concise summary — powered by NLTK word frequency analysis.</div>', unsafe_allow_html=True)

# Input
input_text = st.text_area(
    label="Input text",
    placeholder="Paste your article, essay, or any long text here…",
    height=240,
    label_visibility="collapsed"
)

# Live word / sentence count
if input_text.strip():
    word_cnt = len(input_text.split())
    sent_cnt = len(sent_tokenize(input_text))
    col1, col2, col3 = st.columns(3)
    col1.metric("Words", word_cnt)
    col2.metric("Sentences", sent_cnt)
    col3.metric("Characters", len(input_text))
else:
    st.caption("Start typing or paste text above to see live stats.")

st.markdown('<hr class="divider">', unsafe_allow_html=True)

# Controls
num_sentences = st.slider(
    "Number of sentences in summary",
    min_value=1,
    max_value=10,
    value=3,
    help="How many sentences should the summary contain?"
)

summarize_btn = st.button("Summarize →")

# ─── Run summarizer ───────────────────────────────────────────────────────────
if summarize_btn:
    if not input_text.strip():
        st.error("⚠ Please paste some text first.")
    else:
        with st.spinner("Analyzing text…"):
            result = summarize_text(input_text.strip(), num_sentences)

        if result is None:
            st.error("⚠ Text is too short. Please paste a longer paragraph or article.")
        else:
            st.markdown('<hr class="divider">', unsafe_allow_html=True)
            st.markdown('<div class="section-label">⬡ Summary</div>', unsafe_allow_html=True)
            st.markdown(f'<div class="summary-box">{result["summary"]}</div>', unsafe_allow_html=True)

            # Stats
            st.markdown(f"""
            <div class="stats-row">
                <div class="stat-card">
                    <div class="stat-val">{result['kept']}</div>
                    <div class="stat-key">Sentences kept</div>
                </div>
                <div class="stat-card">
                    <div class="stat-val">{result['summ_words']}</div>
                    <div class="stat-key">Summary words</div>
                </div>
                <div class="stat-card">
                    <div class="stat-val">{result['orig_words']}</div>
                    <div class="stat-key">Original words</div>
                </div>
                <div class="stat-card">
                    <div class="stat-val">{result['reduction']}%</div>
                    <div class="stat-key">Reduction</div>
                </div>
            </div>
            """, unsafe_allow_html=True)

            # Copy button (Streamlit native)
            st.code(result['summary'], language=None)
            st.caption("👆 Click the copy icon on the top-right of the box above to copy your summary.")

st.markdown("""
<div class="footer">
    Built with <span>Python · NLTK · Streamlit</span> — Minor Project · NLP Text Summarizer
</div>
""", unsafe_allow_html=True)
