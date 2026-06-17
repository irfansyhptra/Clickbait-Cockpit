import streamlit as st
import pandas as pd
import numpy as np
import re
import os
import json
import joblib
import matplotlib.pyplot as plt
from wordcloud import WordCloud
from lime.lime_text import LimeTextExplainer
import plotly.express as px

# Konfigurasi Halaman Widescreen Streamlit
st.set_page_config(
    page_title="Detektor Clickbait Bahasa Indonesia",
    page_icon="📰",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom Premium Styling (Glassmorphism & Neon Glow Themes)
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Outfit:wght@300;400;500;600;700;800&family=JetBrains+Mono:wght@400;500;600&display=swap');

html, body, [class*="css"] {
    font-family: 'Outfit', 'Inter', sans-serif;
}

/* Background gradient */
.stApp {
    background: linear-gradient(135deg, #05020c 0%, #0c071d 50%, #030107 100%);
    color: #f1f5f9;
}

/* Widescreen full-bleed container styling */
.block-container {
    max-width: 96% !important;
    padding-left: 2.5rem !important;
    padding-right: 2.5rem !important;
    padding-top: 1.5rem !important;
    padding-bottom: 2rem !important;
}

/* Hide default streamlit branding */
#MainMenu {visibility: hidden;}
footer {visibility: hidden;}
header {visibility: hidden;}

/* Custom Scrollbar */
::-webkit-scrollbar {
    width: 8px;
    height: 8px;
}
::-webkit-scrollbar-track {
    background: rgba(0, 0, 0, 0.2);
}
::-webkit-scrollbar-thumb {
    background: rgba(255, 255, 255, 0.1);
    border-radius: 4px;
}
::-webkit-scrollbar-thumb:hover {
    background: rgba(255, 255, 255, 0.2);
}

/* Glassmorphism card wrapper */
.glass-card {
    background: rgba(15, 10, 30, 0.35);
    backdrop-filter: blur(20px);
    -webkit-backdrop-filter: blur(20px);
    border: 1px solid rgba(255, 255, 255, 0.04);
    border-radius: 20px;
    padding: 24px;
    box-shadow: 0 15px 35px 0 rgba(0, 0, 0, 0.5);
    margin-bottom: 20px;
    transition: all 0.3s cubic-bezier(0.4, 0, 0.2, 1);
}

.glass-card:hover {
    border-color: rgba(0, 242, 254, 0.15);
    box-shadow: 0 20px 45px 0 rgba(0, 242, 254, 0.05);
    background: rgba(15, 10, 30, 0.45);
}

/* Title gradient */
.glow-title {
    font-size: 2.8rem;
    font-weight: 800;
    background: linear-gradient(90deg, #ff2a5f, #ff7b39, #00f2fe);
    -webkit-background-clip: text;
    -webkit-text-fill-color: transparent;
    text-shadow: 0px 0px 40px rgba(255, 42, 95, 0.15);
    margin-bottom: 5px;
    letter-spacing: -0.5px;
}

.sub-title {
    font-size: 1.05rem;
    color: #94a3b8;
    margin-bottom: 25px;
    border-left: 3px solid #ff2a5f;
    padding-left: 12px;
}

/* Prediction Badges */
.badge-container {
    text-align: center;
    margin: 15px 0;
}

.badge-new {
    padding: 10px 24px;
    border-radius: 30px;
    font-weight: 700;
    font-size: 1.2rem;
    text-transform: uppercase;
    letter-spacing: 1.5px;
    display: inline-block;
    box-shadow: 0 4px 15px rgba(0,0,0,0.3);
}

.badge-new.clickbait {
    background: linear-gradient(90deg, #ff2a5f 0%, #ff7b39 100%);
    color: white;
    box-shadow: 0 0 25px rgba(255, 42, 95, 0.4);
    border: 1px solid rgba(255, 42, 95, 0.3);
}

.badge-new.non-clickbait {
    background: linear-gradient(90deg, #00f2fe 0%, #4facfe 100%);
    color: white;
    box-shadow: 0 0 25px rgba(0, 242, 254, 0.4);
    border: 1px solid rgba(0, 242, 254, 0.3);
}

/* LIME Highlight Token container */
.highlight-container {
    margin-top: 25px;
}

.highlight-box {
    line-height: 2.8rem;
    font-family: 'JetBrains Mono', 'Consolas', monospace;
    font-size: 1.15rem;
    padding: 24px;
    background: rgba(0, 0, 0, 0.4);
    border-radius: 16px;
    border: 1px solid rgba(255, 255, 255, 0.05);
    box-shadow: inset 0 3px 15px rgba(0,0,0,0.6);
}

.token-word {
    padding: 4px 10px;
    border-radius: 8px;
    margin: 0 4px;
    display: inline-block;
    font-weight: 600;
    transition: all 0.25s cubic-bezier(0.4, 0, 0.2, 1);
}

.token-word:hover {
    transform: scale(1.08) translateY(-1px);
}

/* Legend items */
.legend-box {
    display: flex;
    gap: 20px;
    margin-top: 20px;
    font-size: 0.95rem;
    color: #94a3b8;
    background: rgba(255, 255, 255, 0.01);
    padding: 12px 20px;
    border-radius: 10px;
    border: 1px solid rgba(255,255,255,0.02);
}

.legend-item {
    display: flex;
    align-items: center;
    gap: 8px;
}

.legend-color {
    width: 14px;
    height: 14px;
    border-radius: 4px;
}

/* Styled metric card */
.metric-container {
    background: rgba(0, 0, 0, 0.25);
    padding: 15px 25px;
    border-radius: 16px;
    border: 1px solid rgba(255, 255, 255, 0.03);
}

/* Customizing Streamlit widgets style */
div[data-baseweb="select"] {
    background-color: rgba(255, 255, 255, 0.03) !important;
}

/* Styled text area */
div[data-testid="stForm"] {
    background: transparent !important;
    border: none !important;
    padding: 0 !important;
}

/* Custom styled sidebar */
section[data-testid="stSidebar"] {
    background-color: #04020a !important;
    border-right: 1px solid rgba(255,255,255,0.04);
}

/* KPI metric cards */
.kpi-card {
    background: rgba(255, 255, 255, 0.015);
    backdrop-filter: blur(12px);
    -webkit-backdrop-filter: blur(12px);
    border: 1px solid rgba(255, 255, 255, 0.04);
    border-radius: 16px;
    padding: 16px 20px;
    text-align: center;
    box-shadow: 0 10px 25px rgba(0, 0, 0, 0.3);
    transition: all 0.3s ease;
    height: 100%;
}
.kpi-card:hover {
    border-color: rgba(255, 255, 255, 0.08);
    transform: translateY(-2px);
    background: rgba(255, 255, 255, 0.03);
}
.kpi-label {
    font-size: 0.75rem;
    color: #94a3b8;
    font-weight: 600;
    text-transform: uppercase;
    letter-spacing: 1px;
    margin-bottom: 6px;
}
.kpi-value {
    font-size: 1.6rem;
    font-weight: 800;
    margin-bottom: 4px;
}
.kpi-sub {
    font-size: 0.7rem;
    color: #64748b;
}

/* Form Submit Button */
button[kind="secondaryFormSubmit"] {
    background: linear-gradient(90deg, #ff2a5f 0%, #ff7b39 100%) !important;
    color: white !important;
    border: none !important;
    border-radius: 12px !important;
    font-weight: 700 !important;
    font-size: 1rem !important;
    padding: 12px 24px !important;
    box-shadow: 0 4px 20px rgba(255, 42, 95, 0.3) !important;
    transition: all 0.3s cubic-bezier(0.4, 0, 0.2, 1) !important;
    width: 100% !important;
}
button[kind="secondaryFormSubmit"]:hover {
    transform: translateY(-2px) !important;
    box-shadow: 0 8px 25px rgba(255, 42, 95, 0.5) !important;
    background: linear-gradient(90deg, #ff3366 0%, #ff8c42 100%) !important;
}

/* Section titles */
.section-title {
    font-size: 1.35rem;
    font-weight: 700;
    color: #ffffff;
    margin-top: 0;
    margin-bottom: 15px;
    display: flex;
    align-items: center;
    gap: 10px;
}
</style>
""", unsafe_allow_html=True)

# Helper Classifier untuk pipeline manual (Model dengan handcrafted features)
class ClickbaitClassifierPipeline:
    def __init__(self, vectorizer, scaler, model):
        self.vectorizer = vectorizer
        self.scaler = scaler
        self.model = model

    def predict_proba(self, texts):
        from scipy.sparse import hstack, csr_matrix
        
        QUESTION_WORDS = {
            "apa", "siapa", "kapan", "dimana", "di mana", "bagaimana", "mengapa",
            "kenapa", "berapa", "apakah", "siapakah"
        }
        
        def extract_handcrafted_features(texts_list):
            s = pd.Series(texts_list).astype(str)
            title_len = s.str.len().values.reshape(-1, 1)
            word_count = s.str.split().apply(len).values.reshape(-1, 1)
            has_question = s.str.lower().apply(
                lambda t: int(any(qw in t for qw in QUESTION_WORDS))
            ).values.reshape(-1, 1)
            excess_punct = s.apply(
                lambda t: len(re.findall(r'[!?]{1,}|\.\.\.+', t))
            ).values.reshape(-1, 1)
            return np.hstack([title_len, word_count, has_question, excess_punct]).astype(float)
            
        Xt = self.vectorizer.transform(texts)
        Xh = self.scaler.transform(extract_handcrafted_features(texts))
        Xc = hstack([Xt, csr_matrix(Xh)]).tocsr()
        return self.model.predict_proba(Xc)

# ----------------- CACHING ASSETS -----------------
@st.cache_resource
def load_model():
    """Memuat model, scaler, dan vectorizer dari folder data"""
    try:
        vectorizer = joblib.load("data/tfidf.pkl")
        scaler = joblib.load("data/handcrafted_scaler.pkl")
        model = joblib.load("data/model_lr.pkl")
        pipeline = ClickbaitClassifierPipeline(vectorizer, scaler, model)
        return pipeline
    except Exception as e:
        st.error(f"Gagal memuat model. Pastikan file model_lr.pkl, tfidf.pkl, dan handcrafted_scaler.pkl ada di folder data. Error: {e}")
        return None

@st.cache_data
def load_analytic_words():
    """Memuat kata clickbait teratas per portal"""
    try:
        with open("data/portal_clickbait_words.json", "r", encoding="utf-8") as f:
            return json.load(f)
    except Exception as e:
        st.warning(f"Gagal memuat file portal_clickbait_words.json. Detail: {e}")
        return {}

# Load Assets
pipeline = load_model()
portal_words_dict = load_analytic_words()

# Explainer LIME (Global)
explainer = LimeTextExplainer(class_names=["NON-CLICKBAIT", "CLICKBAIT"])

# Mappings untuk nama Portal Berita
PORTAL_MAP = {
    "all": "Semua Portal",
    "detikNews": "detikNews",
    "fimela": "Fimela",
    "kapanlagi": "Kapanlagi",
    "kompas": "Kompas",
    "liputan6": "Liputan6",
    "okezone": "Okezone",
    "pos_metro": "Posmetro Medan",
    "republika": "Republika",
    "sindonews": "Sindonews",
    "tempo": "Tempo",
    "tribunnews": "Tribunnews",
    "wowkeren": "Wowkeren"
}
REVERSE_PORTAL_MAP = {v: k for k, v in PORTAL_MAP.items()}

# ----------------- SIDEBAR -----------------
with st.sidebar:
    st.image("https://img.icons8.com/nolan/96/news.png", width=70)
    st.markdown("<h3 style='margin-bottom:0; color:#ffffff;'>Clickbait Cockpit</h3>", unsafe_allow_html=True)
    st.markdown("<p style='color:#64748b; font-size:0.85rem; margin-top:0;'>NLP dashboard based on Indonesian CLICK-ID Dataset</p>", unsafe_allow_html=True)
    st.markdown("---")
    
    st.markdown("##### Spek Dashboard:")
    st.info("""
    - **Model Utama**: TF-IDF + Logistic Regression (+ Fitur Linguistik)
    - **Dataset**: 15.000 Annotated Samples
    - **Akurasi Baseline**: 80% (F1-macro: 79%)
    - **Interpretabilitas**: LIME Text Explainer
    """)
    st.markdown("---")
    st.markdown("<p style='color:#475569; font-size:0.8rem; text-align:center;'>Developer: Irfan NLP Portfolio</p>", unsafe_allow_html=True)

# ----------------- HEADER & TITLE -----------------
st.markdown("<div class='glow-title'>Cockpit Analitis Deteksi Clickbait</div>", unsafe_allow_html=True)
st.markdown("<div class='sub-title'>Dashboard terintegrasi untuk mendeteksi clickbait secara real-time dan menganalisis sebaran kata di 12 portal berita nasional.</div>", unsafe_allow_html=True)

# ----------------- KPI METRICS GRID -----------------
kpi_col1, kpi_col2, kpi_col3, kpi_col4 = st.columns(4)

with kpi_col1:
    st.markdown("""
    <div class="kpi-card">
        <div class="kpi-label">Total Dataset Anotasi</div>
        <div class="kpi-value" style="background: linear-gradient(90deg, #00f2fe, #4facfe); -webkit-background-clip: text; -webkit-text-fill-color: transparent;">15.000</div>
        <div class="kpi-sub">CLICK-ID Headlines</div>
    </div>
    """, unsafe_allow_html=True)

with kpi_col2:
    st.markdown("""
    <div class="kpi-card">
        <div class="kpi-label">Model Detektor</div>
        <div class="kpi-value" style="background: linear-gradient(90deg, #ff2a5f, #ff7b39); -webkit-background-clip: text; -webkit-text-fill-color: transparent;">TF-IDF + LR</div>
        <div class="kpi-sub">Linguistic Features</div>
    </div>
    """, unsafe_allow_html=True)

with kpi_col3:
    st.markdown("""
    <div class="kpi-card">
        <div class="kpi-label">Akurasi Model</div>
        <div class="kpi-value" style="background: linear-gradient(90deg, #00ff87, #60efff); -webkit-background-clip: text; -webkit-text-fill-color: transparent;">80,20%</div>
        <div class="kpi-sub">F1-macro: 79,10%</div>
    </div>
    """, unsafe_allow_html=True)

with kpi_col4:
    st.markdown("""
    <div class="kpi-card">
        <div class="kpi-label">Portal Terintegrasi</div>
        <div class="kpi-value" style="background: linear-gradient(90deg, #b185ff, #ff8dec); -webkit-background-clip: text; -webkit-text-fill-color: transparent;">12 Portal</div>
        <div class="kpi-sub">Media Online Nasional</div>
    </div>
    """, unsafe_allow_html=True)

st.markdown("<div style='margin-bottom: 25px;'></div>", unsafe_allow_html=True)

# ================= MODERN GRID LAYOUT (FULL WIDESCREEN GRID) =================
col_left, col_right = st.columns([1, 1.25], gap="large")

# ----------------- LEFT COLUMN: DETECTOR & LIME -----------------
with col_left:
    st.markdown("<div class='glass-card'>", unsafe_allow_html=True)
    st.markdown("<div class='section-title'>🔍 Uji Judul Real-time</div>", unsafe_allow_html=True)
    st.markdown("Tulis judul berita di bawah untuk menguji kadar clickbait secara instan. Atau gunakan contoh cepat di bawah:")
    
    # Quick Sample Headlines Selector
    SAMPLES = [
        {"title": "Viral! Pria Ini Temukan Benda Tak Terduga di Kamar, Netizen Melongo!", "label": "Clickbait"},
        {"title": "Mengejutkan! Ternyata Ini Rahasia Mudah Menurunkan Berat Badan Tanpa Olahraga", "label": "Clickbait"},
        {"title": "Bikin Merinding! Detik-detik Penyelamatan Kucing Terjebak di Reruntuhan Gedung", "label": "Clickbait"},
        {"title": "Kementerian Keuangan Resmi Naikkan Tarif Cukai Rokok Mulai Tahun Depan", "label": "Non-Clickbait"},
        {"title": "Presiden Jokowi Meresmikan Proyek Jalan Tol Baru di Jawa Tengah Hari Ini", "label": "Non-Clickbait"},
        {"title": "Polisi Menangkap Pelaku Pencurian Kendaraan Bermotor di Jakarta Timur", "label": "Non-Clickbait"}
    ]
    
    # Track selection in session state
    if "news_title_input" not in st.session_state:
        st.session_state.news_title_input = "Viral! Pria Ini Temukan Benda Tak Terduga di Dalam Kamar, Netizen Dibuat Melongo!"
        st.session_state.analyze_triggered = True  # Run automatically on startup
        
    st.markdown("<p style='font-size:0.85rem; color:#94a3b8; font-weight:600; margin-bottom:8px;'>Pilih Contoh Cepat:</p>", unsafe_allow_html=True)
    
    # Display samples in a grid
    sample_cols = st.columns(2)
    for idx, sample in enumerate(SAMPLES):
        col_idx = idx % 2
        badge_type = "🔴" if sample["label"] == "Clickbait" else "🔵"
        if sample_cols[col_idx].button(f"{badge_type} {sample['title'][:40]}...", key=f"sample_{idx}", help=sample["title"]):
            st.session_state.news_title_input = sample["title"]
            st.session_state.analyze_triggered = True
            
    st.markdown("<div style='margin-bottom:15px;'></div>", unsafe_allow_html=True)
    
    # Form for entering news title to prevent typing lag
    with st.form(key="detector_form"):
        input_title = st.text_area(
            label="Masukkan Judul Berita",
            value=st.session_state.news_title_input,
            height=100,
            placeholder="Ketik judul berita di sini...",
            key="news_title_textarea",
            label_visibility="collapsed"
        )
        btn_analyze = st.form_submit_button("Mulai Analisis ✨")
        
    st.markdown("</div>", unsafe_allow_html=True)
    
    # Logic to check trigger
    should_analyze = btn_analyze or st.session_state.get("analyze_triggered", False)
    if st.session_state.get("analyze_triggered", False):
        st.session_state.analyze_triggered = False
        
    target_text = input_title if btn_analyze else st.session_state.news_title_input
    st.session_state.news_title_input = target_text
    
    if should_analyze and target_text:
        if not target_text.strip():
            st.warning("Silakan masukkan judul berita terlebih dahulu.")
        elif pipeline is None:
            st.error("Model tidak tersedia untuk dijalankan.")
        else:
            with st.spinner("Mengevaluasi judul berita..."):
                # Prediksi
                probs = pipeline.predict_proba([target_text])[0]
                clickbait_prob = probs[1]
                non_clickbait_prob = probs[0]
                prediction = "CLICKBAIT" if clickbait_prob >= 0.5 else "NON-CLICKBAIT"
                
                st.markdown("<div class='glass-card'>", unsafe_allow_html=True)
                st.markdown("<div class='section-title'>📊 Hasil Prediksi</div>", unsafe_allow_html=True)
                
                col_badge, col_score = st.columns([1, 1])
                with col_badge:
                    if prediction == "CLICKBAIT":
                        st.markdown(f'<div class="badge-container"><span class="badge-new clickbait">⚠️ {prediction}</span></div>', unsafe_allow_html=True)
                    else:
                        st.markdown(f'<div class="badge-container"><span class="badge-new non-clickbait">✅ {prediction}</span></div>', unsafe_allow_html=True)
                        
                with col_score:
                    confidence = clickbait_prob if prediction == "CLICKBAIT" else non_clickbait_prob
                    st.markdown(f"""
                    <div class="metric-container">
                        <div style="font-size:0.8rem; color:#94a3b8; font-weight:600; text-transform:uppercase; letter-spacing:1px;">CONFIDENCE SCORE</div>
                        <div style="font-size:1.8rem; font-weight:800; color:#ffffff; margin:5px 0;">{confidence:.2%}</div>
                    </div>
                    """, unsafe_allow_html=True)
                    st.progress(int(confidence * 100))
                    
                st.markdown("---")
                
                # LIME Token Highlight
                st.markdown("<div class='highlight-container'>", unsafe_allow_html=True)
                st.markdown("<div class='section-title'>🧩 Highlight Token LIME (Kata Pemicu Prediksi)</div>", unsafe_allow_html=True)
                st.markdown("<p style='font-size:0.85rem; color:#94a3b8;'>Arahkan kursor pada kata untuk melihat kontribusi numeriknya terhadap prediksi:</p>", unsafe_allow_html=True)
                
                exp = explainer.explain_instance(
                    target_text,
                    pipeline.predict_proba,
                    num_features=8
                )
                
                word_weights = dict(exp.as_list())
                tokens = re.split(r'(\s+)', target_text)
                highlighted_html = ""
                
                weights_abs = [abs(w) for w in word_weights.values()]
                max_w = max(weights_abs) if weights_abs else 1.0
                
                for token in tokens:
                    if token.isspace():
                        highlighted_html += token
                        continue
                    
                    word_clean = re.sub(r'[^\w]', '', token).lower()
                    
                    if word_clean in word_weights:
                        w = word_weights[word_clean]
                        opacity = min(abs(w) / max_w, 1.0) * 0.7 + 0.15
                        
                        if w > 0:
                            bg_color = f"rgba(255, 42, 95, {opacity})"
                            border_color = f"rgba(255, 42, 95, {opacity + 0.2})"
                            title_tip = f"Memicu Clickbait (Bobot: +{w:.4f})"
                        else:
                            bg_color = f"rgba(0, 242, 254, {opacity})"
                            border_color = f"rgba(0, 242, 254, {opacity + 0.2})"
                            title_tip = f"Memicu Non-Clickbait (Bobot: {w:.4f})"
                            
                        highlighted_html += f'<span class="token-word" style="background-color: {bg_color}; border: 1px solid {border_color}; color: white; cursor: help;" title="{title_tip}">{token}</span>'
                    else:
                        highlighted_html += f'<span class="token-word" style="color: #94a3b8; border: 1px solid rgba(255,255,255,0.03); background:rgba(0,0,0,0.15);">{token}</span>'
                
                st.markdown(f'<div class="highlight-box">{highlighted_html}</div>', unsafe_allow_html=True)
                
                # Legend LIME
                st.markdown("""
                <div class="legend-box">
                    <div class="legend-item">
                        <div class="legend-color" style="background-color: rgba(255, 42, 95, 0.75);"></div>
                        <span>Memicu Clickbait (Positif)</span>
                    </div>
                    <div class="legend-item">
                        <div class="legend-color" style="background-color: rgba(0, 242, 254, 0.75);"></div>
                        <span>Memicu Non-Clickbait (Negatif)</span>
                    </div>
                </div>
                """, unsafe_allow_html=True)
                st.markdown("</div>", unsafe_allow_html=True)
                st.markdown("</div>", unsafe_allow_html=True)

# ----------------- RIGHT COLUMN: PORTAL ANALYTICS -----------------
with col_right:
    st.markdown("<div class='glass-card'>", unsafe_allow_html=True)
    st.markdown("<div class='section-title'>📈 Analitik Tren Lintas Portal</div>", unsafe_allow_html=True)
    st.markdown("Pilih portal berita di bawah ini untuk melihat sebaran kata clickbait yang paling dominan:")
    
    # Dropdown portal selector
    portal_names = list(REVERSE_PORTAL_MAP.keys())
    portal_names.remove("Semua Portal")
    portal_names = ["Semua Portal"] + sorted(portal_names)
    
    selected_portal_name = st.selectbox(
        "Pilih Portal Berita:",
        portal_names,
        key="portal_selector",
        label_visibility="collapsed"
    )
    
    selected_key = REVERSE_PORTAL_MAP[selected_portal_name]
    st.markdown("</div>", unsafe_allow_html=True)
    
    if not portal_words_dict:
        st.error("Data kata portal tidak ditemukan. Pastikan data/portal_clickbait_words.json sudah digenerate.")
    else:
        words_data = portal_words_dict.get(selected_key, [])
        
        if not words_data:
            st.warning(f"Data clickbait kata untuk portal {selected_portal_name} kosong.")
        else:
            # ----------------- PORTAL SPECIFIC DYNAMIC METRICS SUB-GRID -----------------
            top_word = words_data[0][0]
            top_freq = words_data[0][1]
            total_vocab = len(words_data)
            
            sub_col1, sub_col2, sub_col3 = st.columns(3)
            
            with sub_col1:
                st.markdown(f"""
                <div class="metric-container" style="text-align: center; height:100%;">
                    <div style="font-size:0.75rem; color:#94a3b8; font-weight:600; text-transform:uppercase; letter-spacing:0.5px;">Top Clickbait Word</div>
                    <div style="font-size:1.4rem; font-weight:800; color:#ff2a5f; margin:5px 0;">"{top_word}"</div>
                    <div style="font-size:0.7rem; color:#64748b;">Kata paling dominan</div>
                </div>
                """, unsafe_allow_html=True)
                
            with sub_col2:
                st.markdown(f"""
                <div class="metric-container" style="text-align: center; height:100%;">
                    <div style="font-size:0.75rem; color:#94a3b8; font-weight:600; text-transform:uppercase; letter-spacing:0.5px;">Frekuensi Kata</div>
                    <div style="font-size:1.4rem; font-weight:800; color:#ff7b39; margin:5px 0;">{top_freq}</div>
                    <div style="font-size:0.7rem; color:#64748b;">Kemunculan di judul</div>
                </div>
                """, unsafe_allow_html=True)
                
            with sub_col3:
                st.markdown(f"""
                <div class="metric-container" style="text-align: center; height:100%;">
                    <div style="font-size:0.75rem; color:#94a3b8; font-weight:600; text-transform:uppercase; letter-spacing:0.5px;">Kosa Kata Unik</div>
                    <div style="font-size:1.4rem; font-weight:800; color:#00f2fe; margin:5px 0;">{total_vocab}</div>
                    <div style="font-size:0.7rem; color:#64748b;">Kosakata teridentifikasi</div>
                </div>
                """, unsafe_allow_html=True)
                
            st.markdown("<div style='margin-bottom: 20px;'></div>", unsafe_allow_html=True)
            
            # Word Cloud Card
            st.markdown("<div class='glass-card'>", unsafe_allow_html=True)
            st.markdown(f"<div class='section-title'>☁️ Word Cloud Clickbait - {selected_portal_name}</div>", unsafe_allow_html=True)
            
            word_freq = dict(words_data)
            wc = WordCloud(
                width=800,
                height=320,
                background_color="rgba(0,0,0,0)",
                mode="RGBA",
                colormap="magma",
                max_words=60,
                random_state=42
            ).generate_from_frequencies(word_freq)
            
            fig, ax = plt.subplots(figsize=(10, 4), facecolor='none')
            ax.imshow(wc, interpolation="bilinear")
            ax.axis("off")
            st.pyplot(fig, clear_figure=True, width="stretch")
            st.markdown("</div>", unsafe_allow_html=True)
            
            # Chart Card
            st.markdown("<div class='glass-card'>", unsafe_allow_html=True)
            st.markdown(f"<div class='section-title'>📊 Top 20 Kata Teratas - {selected_portal_name}</div>", unsafe_allow_html=True)
            
            top_20 = words_data[:20]
            words, counts = zip(*top_20)
            
            df_plotly = pd.DataFrame({
                "Kata": words,
                "Frekuensi": counts
            }).sort_values(by="Frekuensi", ascending=True)
            
            fig_bar = px.bar(
                df_plotly,
                x="Frekuensi",
                y="Kata",
                orientation="h",
                color="Frekuensi",
                color_continuous_scale=["#ff7b39", "#ff2a5f", "#9d0042"],
                template="plotly_dark",
            )
            
            fig_bar.update_layout(
                paper_bgcolor="rgba(0,0,0,0)",
                plot_bgcolor="rgba(0,0,0,0)",
                font=dict(family="Outfit, sans-serif", color="#cbd5e1"),
                margin=dict(l=0, r=0, t=10, b=0),
                height=450,
                coloraxis_showscale=False,
                xaxis=dict(
                    showgrid=True, 
                    gridcolor="rgba(255,255,255,0.05)",
                    tickfont=dict(family="Outfit", size=11, color="#94a3b8"),
                    title=dict(font=dict(family="Outfit", size=12, color="#94a3b8"))
                ),
                yaxis=dict(
                    showgrid=False,
                    tickfont=dict(family="Outfit", size=12, color="#f1f5f9"),
                )
            )
            
            st.plotly_chart(fig_bar, width="stretch", config={'displayModeBar': False})
            st.markdown("</div>", unsafe_allow_html=True)

