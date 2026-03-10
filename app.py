import streamlit as st
import numpy as np
import json
import cv2
from PIL import Image
import tensorflow as tf
from tensorflow import keras
import matplotlib.pyplot as plt
import matplotlib
matplotlib.use('Agg')

# ─── Page Config ───────────────────────────────────────────────────────────────
st.set_page_config(
    page_title='RiceGuard AI',
    page_icon='🌾',
    layout='wide'
)

# ─── Custom CSS ────────────────────────────────────────────────────────────────
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Playfair+Display:wght@700;900&family=DM+Sans:wght@300;400;500;600&display=swap');

/* ── Global Reset ── */
*, *::before, *::after { box-sizing: border-box; margin: 0; padding: 0; }

html, body, [data-testid="stAppViewContainer"] {
    background: #0a0f0a !important;
    font-family: 'DM Sans', sans-serif;
    color: #e8f0e8;
}

[data-testid="stAppViewContainer"] {
    background: linear-gradient(135deg, #0a0f0a 0%, #0d1a0d 50%, #0a1210 100%) !important;
}

/* ── Hide Streamlit Branding ── */
#MainMenu, footer, header { visibility: hidden; }
[data-testid="stToolbar"] { display: none; }

/* ── Hero Section ── */
.hero {
    text-align: center;
    padding: 3rem 1rem 2rem;
    position: relative;
}

.hero-badge {
    display: inline-block;
    background: rgba(74, 222, 128, 0.1);
    border: 1px solid rgba(74, 222, 128, 0.3);
    color: #4ade80;
    font-size: 0.75rem;
    font-weight: 600;
    letter-spacing: 0.15em;
    text-transform: uppercase;
    padding: 0.4rem 1.2rem;
    border-radius: 100px;
    margin-bottom: 1.5rem;
}

.hero-title {
    font-family: 'Playfair Display', serif;
    font-size: clamp(2.5rem, 6vw, 4.5rem);
    font-weight: 900;
    line-height: 1.1;
    margin-bottom: 1rem;
    background: linear-gradient(135deg, #ffffff 0%, #86efac 50%, #4ade80 100%);
    -webkit-background-clip: text;
    -webkit-text-fill-color: transparent;
    background-clip: text;
}

.hero-divider {
    width: 60px;
    height: 3px;
    background: linear-gradient(90deg, #4ade80, #22d3ee);
    border-radius: 2px;
    margin: 0 auto 3rem;
}

/* ── Stats Bar ── */
.stats-bar {
    display: flex;
    justify-content: center;
    gap: 3rem;
    padding: 1.5rem 2rem;
    background: rgba(255,255,255,0.03);
    border: 1px solid rgba(255,255,255,0.06);
    border-radius: 16px;
    margin: 0 auto 3rem;
    max-width: 600px;
}

.stat-item { text-align: center; }

.stat-value {
    font-family: 'Playfair Display', serif;
    font-size: 1.8rem;
    font-weight: 700;
    color: #4ade80;
}

.stat-label {
    font-size: 0.75rem;
    color: #4a5e4a;
    text-transform: uppercase;
    letter-spacing: 0.1em;
    margin-top: 0.2rem;
}

/* ── Result Card ── */
.result-card {
    background: rgba(255,255,255,0.03);
    border: 1px solid rgba(255,255,255,0.08);
    border-radius: 20px;
    padding: 2rem;
    margin-bottom: 1.5rem;
}

.result-card.success {
    border-color: rgba(74, 222, 128, 0.3);
    background: rgba(74, 222, 128, 0.05);
}

.result-card.warning {
    border-color: rgba(251, 191, 36, 0.3);
    background: rgba(251, 191, 36, 0.05);
}

.result-card.danger {
    border-color: rgba(239, 68, 68, 0.3);
    background: rgba(239, 68, 68, 0.05);
}

/* ── Disease Tag ── */
.disease-tag {
    display: inline-flex;
    align-items: center;
    gap: 0.5rem;
    padding: 0.5rem 1.2rem;
    border-radius: 100px;
    font-size: 0.8rem;
    font-weight: 600;
    letter-spacing: 0.08em;
    text-transform: uppercase;
    margin-bottom: 1rem;
}

.disease-tag.blight {
    background: rgba(251, 191, 36, 0.15);
    color: #fbbf24;
    border: 1px solid rgba(251, 191, 36, 0.3);
}

.disease-tag.brown {
    background: rgba(180, 83, 9, 0.15);
    color: #fb923c;
    border: 1px solid rgba(180, 83, 9, 0.3);
}

.disease-tag.smut {
    background: rgba(107, 114, 128, 0.15);
    color: #9ca3af;
    border: 1px solid rgba(107, 114, 128, 0.3);
}

/* ── Confidence Bar ── */
.conf-row { margin-bottom: 1rem; }

.conf-label {
    display: flex;
    justify-content: space-between;
    margin-bottom: 0.4rem;
    font-size: 0.9rem;
}

.conf-name { color: #c8d8c8; font-weight: 500; }
.conf-pct  { color: #4ade80; font-weight: 600; font-family: 'Playfair Display', serif; }

.conf-track {
    height: 6px;
    background: rgba(255,255,255,0.06);
    border-radius: 100px;
    overflow: hidden;
}

.conf-fill {
    height: 100%;
    border-radius: 100px;
}

.conf-fill.primary   { background: linear-gradient(90deg, #4ade80, #22d3ee); }
.conf-fill.secondary { background: rgba(255,255,255,0.15); }

/* ── Info Cards ── */
.info-grid {
    display: grid;
    grid-template-columns: 1fr 1fr;
    gap: 1rem;
    margin-top: 1.5rem;
}

.info-card {
    background: rgba(255,255,255,0.03);
    border: 1px solid rgba(255,255,255,0.07);
    border-radius: 14px;
    padding: 1.2rem;
}

.info-card-title {
    font-size: 0.75rem;
    text-transform: uppercase;
    letter-spacing: 0.1em;
    color: #4a5e4a;
    margin-bottom: 0.6rem;
}

.info-card-text {
    font-size: 0.9rem;
    color: #9ab09a;
    line-height: 1.6;
}

/* ── Disease Guide Cards ── */
.guide-grid {
    display: grid;
    grid-template-columns: repeat(3, 1fr);
    gap: 1rem;
    margin-top: 1rem;
}

.guide-card {
    background: rgba(255,255,255,0.02);
    border: 1px solid rgba(255,255,255,0.06);
    border-radius: 16px;
    padding: 1.5rem;
    text-align: center;
}

.guide-emoji   { font-size: 2rem; margin-bottom: 0.8rem; }
.guide-name    { font-weight: 600; color: #ffffff; margin-bottom: 0.4rem; font-size: 0.95rem; }
.guide-caption { font-size: 0.8rem; color: #4a5e4a; line-height: 1.5; }

/* ── Section Title ── */
.section-title {
    font-family: 'Playfair Display', serif;
    font-size: 1.3rem;
    color: #ffffff;
    margin-bottom: 1.2rem;
    display: flex;
    align-items: center;
    gap: 0.6rem;
}

/* ── Streamlit Overrides ── */
[data-testid="stFileUploader"] > div {
    background: rgba(255,255,255,0.02) !important;
    border: 2px dashed rgba(74, 222, 128, 0.25) !important;
    border-radius: 16px !important;
    padding: 2rem !important;
}

img { border-radius: 12px; }

/* ── Footer ── */
.footer {
    text-align: center;
    padding: 2rem;
    color: #2a3a2a;
    font-size: 0.8rem;
    letter-spacing: 0.05em;
    margin-top: 3rem;
    border-top: 1px solid rgba(255,255,255,0.04);
}
</style>
""", unsafe_allow_html=True)

# ─── Load Model ────────────────────────────────────────────────────────────────
@st.cache_resource
def load_model():
    model = keras.models.load_model(
        'saved_models/best_model_inceptionv3.keras'
    )
    with open('saved_models/class_labels.json') as f:
        class_indices = json.load(f)
    labels = {v: k for k, v in class_indices.items()}
    return model, labels

model, labels = load_model()

# ─── Disease Data ──────────────────────────────────────────────────────────────
disease_data = {
    'Bacterial leaf blight': {
        'emoji'      : '🟡',
        'tag_class'  : 'blight',
        'short'      : 'Bacterial Leaf Blight',
        'description': 'Caused by Xanthomonas oryzae pv. oryzae. '
                       'Produces yellow to white water-soaked lesions '
                       'that spread along leaf margins and veins.',
        'treatment'  : 'Apply copper-based bactericides. Use resistant '
                       'varieties. Improve field drainage and avoid '
                       'excessive nitrogen fertilization.',
        'severity'   : 'High'
    },
    'Brown spot': {
        'emoji'      : '🟤',
        'tag_class'  : 'brown',
        'short'      : 'Brown Spot',
        'description': 'Caused by Cochliobolus miyabeanus fungus. '
                       'Appears as oval to circular brown spots with '
                       'yellow halos scattered across the leaf blade.',
        'treatment'  : 'Apply propiconazole or tricyclazole fungicide. '
                       'Ensure balanced NPK fertilization. '
                       'Avoid drought stress during critical stages.',
        'severity'   : 'Medium'
    },
    'Leaf smut': {
        'emoji'      : '⚫',
        'tag_class'  : 'smut',
        'short'      : 'Leaf Smut',
        'description': 'Caused by Entyloma oryzae. Characterized by '
                       'angular dark black to grey raised spots on '
                       'both surfaces of the leaf blade.',
        'treatment'  : 'Use certified disease-free seeds. Apply '
                       'propiconazole fungicide. Practice crop rotation '
                       'and remove infected plant debris.',
        'severity'   : 'Low–Medium'
    }
}

# ─── HERO ──────────────────────────────────────────────────────────────────────
st.markdown("""
<div class="hero">
    <div class="hero-badge">🌾 AI-Powered Plant Pathology</div>
    <h1 class="hero-title">RiceGuard AI</h1>
    <p style="
        color: #6b7f6b;
        font-size: 1.1rem;
        font-weight: 300;
        max-width: 500px;
        width: 100%;
        margin: 0 auto 2.5rem auto;
        line-height: 1.7;
        text-align: center;
        display: block;
    ">
        Upload a rice leaf photo and get instant disease diagnosis
        powered by deep learning
    </p>
    <div class="hero-divider"></div>
</div>
""", unsafe_allow_html=True)

# ─── Stats Bar ─────────────────────────────────────────────────────────────────
st.markdown("""
<div class="stats-bar">
    <div class="stat-item">
        <div class="stat-value">94.44%</div>
        <div class="stat-label">Accuracy</div>
    </div>
    <div class="stat-item">
        <div class="stat-value">3</div>
        <div class="stat-label">Diseases</div>
    </div>
    <div class="stat-item">
        <div class="stat-value">119</div>
        <div class="stat-label">Images Trained</div>
    </div>
    <div class="stat-item">
        <div class="stat-value">V3</div>
        <div class="stat-label">InceptionV3</div>
    </div>
</div>
""", unsafe_allow_html=True)

# ─── Main Layout ───────────────────────────────────────────────────────────────
left_col, right_col = st.columns([1, 1], gap='large')

with left_col:
    st.markdown('<p class="section-title">📤 Upload Image</p>',
                unsafe_allow_html=True)

    uploaded = st.file_uploader(
        'Drop your rice leaf photo here',
        type=['jpg', 'jpeg', 'png'],
        label_visibility='collapsed'
    )

    if uploaded:
        image = Image.open(uploaded)
        st.image(image, use_container_width=True),
                 caption='Uploaded Image')

    st.markdown('<br>', unsafe_allow_html=True)
    st.markdown('<p class="section-title">🌿 Detectable Diseases</p>',
                unsafe_allow_html=True)
    st.markdown("""
    <div class="guide-grid">
        <div class="guide-card">
            <div class="guide-emoji">🟡</div>
            <div class="guide-name">Bacterial Blight</div>
            <div class="guide-caption">Yellow/white lesions on leaf edges</div>
        </div>
        <div class="guide-card">
            <div class="guide-emoji">🟤</div>
            <div class="guide-name">Brown Spot</div>
            <div class="guide-caption">Brown oval spots on leaf surface</div>
        </div>
        <div class="guide-card">
            <div class="guide-emoji">⚫</div>
            <div class="guide-name">Leaf Smut</div>
            <div class="guide-caption">Dark raised spots on leaf</div>
        </div>
    </div>
    """, unsafe_allow_html=True)

with right_col:
    if uploaded is None:
        st.markdown("""
        <div style="
            height: 400px;
            display: flex;
            flex-direction: column;
            align-items: center;
            justify-content: center;
            background: rgba(255,255,255,0.02);
            border: 1px solid rgba(255,255,255,0.06);
            border-radius: 20px;
            text-align: center;
            padding: 2rem;
        ">
            <div style="font-size:4rem; margin-bottom:1rem;">🔬</div>
            <div style="
                font-family: 'Playfair Display', serif;
                font-size: 1.3rem;
                color: #ffffff;
                margin-bottom: 0.5rem;
            ">Awaiting Analysis</div>
            <div style="color:#4a5e4a; font-size:0.9rem;
                        max-width:260px; line-height:1.6;">
                Upload a rice leaf image on the left
                to get instant AI diagnosis
            </div>
        </div>
        """, unsafe_allow_html=True)

    else:
        img_array = np.array(image)
        if img_array.shape[-1] == 4:
            img_array = img_array[:, :, :3]

        img_resized = cv2.resize(img_array, (299, 299))
        img_input   = img_resized / 255.0
        img_input   = np.expand_dims(img_input, axis=0)

        # ── Green Leaf Check ──
        def is_rice_leaf(arr):
            img = arr.astype(np.uint8)
            hsv = cv2.cvtColor(img, cv2.COLOR_RGB2HSV)
            lower_green = np.array([25, 40, 40])
            upper_green = np.array([90, 255, 255])
            mask = cv2.inRange(hsv, lower_green, upper_green)
            green_ratio = np.sum(mask > 0) / mask.size
            gray = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
            edges = cv2.Canny(gray, 50, 150)
            edge_ratio = np.sum(edges > 0) / edges.size
            is_leaf = (green_ratio > 0.08) and (edge_ratio > 0.02)
            return green_ratio, edge_ratio, is_leaf

        green_ratio, edge_ratio, is_leaf = is_rice_leaf(img_array)

        if not is_leaf:
            st.markdown(f"""
            <div style="
                background: rgba(239,68,68,0.08);
                border: 1px solid rgba(239,68,68,0.3);
                border-radius: 16px;
                padding: 2.5rem;
                text-align: center;
                margin-top: 1rem;
            ">
                <div style="font-size:3.5rem; margin-bottom:1rem;">🚫</div>
                <div style="
                    font-family: 'Playfair Display', serif;
                    font-size: 1.5rem;
                    color: #ffffff;
                    margin-bottom: 0.8rem;
                ">Not a Rice Leaf!</div>
                <div style="color:#9a6060; font-size:0.9rem; line-height:1.7;">
                    This image doesn't appear to be a rice leaf.<br>
                    Please upload a clear photo of a rice leaf.<br><br>
                    <span style="color:#4a5e4a; font-size:0.8rem;">
                        Green detected: {green_ratio*100:.1f}%
                        &nbsp;|&nbsp; Edge texture: {edge_ratio*100:.1f}%
                    </span>
                </div>
            </div>
            """, unsafe_allow_html=True)

        else:
            with st.spinner('🔍 Analyzing...'):
                preds      = model.predict(img_input, verbose=0)
                pred_idx   = np.argmax(preds[0])
                pred_label = labels[pred_idx]
                confidence = preds[0][pred_idx] * 100
                info       = disease_data[pred_label]

            card_class = ('success' if confidence >= 80
                          else 'warning' if confidence >= 60
                          else 'danger')

            st.markdown(f"""
            <div class="result-card {card_class}">
                <div class="disease-tag {info['tag_class']}">
                    {info['emoji']} {info['short']}
                </div>
                <div style="
                    font-family: 'Playfair Display', serif;
                    font-size: 2.2rem;
                    font-weight: 700;
                    color: #ffffff;
                    margin-bottom: 0.3rem;
                ">{confidence:.1f}%</div>
                <div style="color:#4a5e4a; font-size:0.85rem;">
                    Confidence Score
                </div>
            </div>
            """, unsafe_allow_html=True)

            st.markdown('<p class="section-title">📊 All Class Scores</p>',
                        unsafe_allow_html=True)

            class_names = [labels[i] for i in range(3)]
            for i in range(3):
                name   = class_names[i]
                conf   = preds[0][i] * 100
                is_top = (i == pred_idx)
                fill   = 'primary' if is_top else 'secondary'
                emoji  = disease_data[name]['emoji']

                st.markdown(f"""
                <div class="conf-row">
                    <div class="conf-label">
                        <span class="conf-name">{emoji} {name}</span>
                        <span class="conf-pct">{conf:.1f}%</span>
                    </div>
                    <div class="conf-track">
                        <div class="conf-fill {fill}"
                             style="width:{conf}%">
                        </div>
                    </div>
                </div>
                """, unsafe_allow_html=True)

            st.markdown('<br>', unsafe_allow_html=True)
            st.markdown('<p class="section-title">📋 Disease Details</p>',
                        unsafe_allow_html=True)

            st.markdown(f"""
            <div class="info-grid">
                <div class="info-card">
                    <div class="info-card-title">🔍 Description</div>
                    <div class="info-card-text">{info['description']}</div>
                </div>
                <div class="info-card">
                    <div class="info-card-title">💊 Treatment</div>
                    <div class="info-card-text">{info['treatment']}</div>
                </div>
            </div>
            <div class="info-card" style="margin-top:1rem;">
                <div class="info-card-title">⚠️ Severity Level</div>
                <div class="info-card-text">{info['severity']}</div>
            </div>
            """, unsafe_allow_html=True)

# ─── Footer ────────────────────────────────────────────────────────────────────
st.markdown("""
<div class="footer">
    PRCP-1001 · RiceGuard AI · InceptionV3 · 94.44% Accuracy
</div>
""", unsafe_allow_html=True)