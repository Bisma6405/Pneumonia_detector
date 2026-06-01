import streamlit as st
import pickle
import json
import numpy as np
from PIL import Image, ImageFilter, ImageEnhance, ImageOps
import torch
import torch.nn.functional as F
from torchvision import transforms
import os
import io
import csv
import time

# ── Page Config ───────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="PneumoScan AI",
    page_icon="🫁",
    layout="wide",
)

# ── CSS ───────────────────────────────────────────────────────────────────────
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Orbitron:wght@400;700;900&family=Exo+2:wght@300;400;600&display=swap');

:root {
    --cyan:   #00F5FF;
    --green:  #00FF88;
    --red:    #FF3860;
    --orange: #FF8C00;
    --yellow: #FFD700;
    --bg:     #020B18;
    --card:   #0A1F35;
    --border: #0D3B66;
    --text:   #CDD6F4;
    --dim:    #6C7086;
}

html, body, [class*="css"] {
    font-family: 'Exo 2', sans-serif;
    background-color: var(--bg) !important;
    color: var(--text);
}

/* ── Main title ── */
.main-title {
    font-family: 'Orbitron', monospace;
    font-size: 2.8rem;
    font-weight: 900;
    color: var(--cyan);
    text-align: center;
    letter-spacing: 6px;
    text-shadow: 0 0 30px rgba(0,245,255,0.5);
    margin-bottom: 0.2rem;
}
.main-sub {
    font-family: 'Exo 2', sans-serif;
    font-size: 0.85rem;
    color: var(--dim);
    text-align: center;
    letter-spacing: 4px;
    margin-bottom: 1.5rem;
}

/* ── Cards ── */
.scan-card {
    background: var(--card);
    border: 1px solid var(--border);
    border-top: 3px solid var(--cyan);
    border-radius: 8px;
    padding: 1.5rem;
    margin: 0.8rem 0;
}

/* ── Result banners ── */
.result-normal {
    background: linear-gradient(135deg, #003322, #004433);
    border: 2px solid var(--green);
    border-radius: 8px;
    padding: 1rem 1.5rem;
    text-align: center;
    box-shadow: 0 0 20px rgba(0,255,136,0.3);
}
.result-pneumonia {
    background: linear-gradient(135deg, #2D0014, #3D0020);
    border: 2px solid var(--red);
    border-radius: 8px;
    padding: 1rem 1.5rem;
    text-align: center;
    box-shadow: 0 0 20px rgba(255,56,96,0.4);
    animation: pulse-red 2s infinite;
}
@keyframes pulse-red {
    0%   { box-shadow: 0 0 20px rgba(255,56,96,0.4); }
    50%  { box-shadow: 0 0 40px rgba(255,56,96,0.7); }
    100% { box-shadow: 0 0 20px rgba(255,56,96,0.4); }
}
.result-label {
    font-family: 'Orbitron', monospace;
    font-size: 2rem;
    font-weight: 900;
    letter-spacing: 4px;
}

/* ── Metric badges ── */
.metric-row { display: flex; gap: 1rem; margin: 1rem 0; flex-wrap: wrap; }
.metric-badge {
    flex: 1;
    background: rgba(0,245,255,0.07);
    border: 1px solid rgba(0,245,255,0.25);
    border-radius: 6px;
    padding: 0.6rem 1rem;
    text-align: center;
    min-width: 100px;
}
.metric-label {
    font-family: 'Orbitron', monospace;
    font-size: 0.55rem;
    color: var(--dim);
    letter-spacing: 2px;
    display: block;
    margin-bottom: 0.3rem;
}
.metric-value {
    font-family: 'Orbitron', monospace;
    font-size: 1.1rem;
    font-weight: 700;
    color: var(--cyan);
}

/* ── Probability bars ── */
.prob-container { margin: 0.5rem 0; }
.prob-label {
    font-family: 'Exo 2', sans-serif;
    font-size: 0.8rem;
    color: var(--dim);
    letter-spacing: 2px;
    margin-bottom: 0.3rem;
}
.prob-track {
    width: 100%;
    background: rgba(255,255,255,0.07);
    border-radius: 4px;
    height: 14px;
    overflow: hidden;
}
.prob-fill-g {
    height: 100%;
    border-radius: 4px;
    background: linear-gradient(90deg, #00FF88, #00CC66);
    transition: width 0.8s ease;
}
.prob-fill-r {
    height: 100%;
    border-radius: 4px;
    background: linear-gradient(90deg, #FF3860, #CC0033);
    transition: width 0.8s ease;
}
.prob-pct {
    font-family: 'Orbitron', monospace;
    font-size: 0.75rem;
    text-align: right;
    margin-top: 2px;
}

/* ── Severity meter ── */
.sev-section { margin: 1rem 0; }
.sev-title {
    font-family: 'Orbitron', monospace;
    font-size: 0.6rem;
    color: var(--dim);
    letter-spacing: 3px;
    margin-bottom: 0.5rem;
}
.sev-track {
    width: 100%;
    height: 18px;
    border-radius: 9px;
    background: linear-gradient(90deg, #00FF88 0%, #FFD700 35%, #FF8C00 65%, #FF3860 100%);
    position: relative;
    box-shadow: 0 0 10px rgba(0,245,255,0.2);
}
.sev-needle {
    position: absolute;
    top: -5px;
    width: 4px;
    height: 28px;
    background: white;
    border-radius: 2px;
    box-shadow: 0 0 8px rgba(255,255,255,0.8);
    transform: translateX(-50%);
}
.sev-badge {
    display: inline-block;
    padding: 0.25rem 0.9rem;
    border-radius: 20px;
    font-family: 'Orbitron', monospace;
    font-size: 0.75rem;
    font-weight: 700;
    letter-spacing: 2px;
    margin-top: 0.5rem;
}

/* ── Sidebar ── */
section[data-testid="stSidebar"] {
    background: #050F1C !important;
    border-right: 1px solid var(--border);
}
.sidebar-title {
    font-family: 'Orbitron', monospace;
    font-size: 0.7rem;
    color: var(--cyan);
    letter-spacing: 3px;
    margin-bottom: 0.8rem;
    padding-bottom: 0.3rem;
    border-bottom: 1px solid var(--border);
}
.status-dot {
    display: inline-block;
    width: 8px; height: 8px;
    border-radius: 50%;
    margin-right: 6px;
}
.status-ok  { background: var(--green); box-shadow: 0 0 6px var(--green); }
.status-err { background: var(--red);   box-shadow: 0 0 6px var(--red);   }
.meta-row {
    display: flex;
    justify-content: space-between;
    padding: 0.3rem 0;
    border-bottom: 1px solid rgba(13,59,102,0.5);
    font-size: 0.82rem;
}
.meta-key { color: var(--dim); font-family: 'Exo 2', sans-serif; }
.meta-val { color: var(--cyan); font-family: 'Orbitron', monospace; font-size: 0.75rem; }

/* ── Batch cards ── */
.batch-card {
    background: var(--card);
    border: 1px solid var(--border);
    border-radius: 6px;
    padding: 0.6rem;
    margin-bottom: 0.5rem;
    text-align: center;
}
.batch-normal   { border-top: 3px solid var(--green); }
.batch-pneumonia{ border-top: 3px solid var(--red); }

/* ── Summary badges ── */
.summary-row { display: flex; gap: 0.8rem; margin: 1rem 0; flex-wrap: wrap; }
.summary-badge {
    flex: 1;
    background: var(--card);
    border: 1px solid var(--border);
    border-radius: 6px;
    padding: 0.8rem;
    text-align: center;
    min-width: 90px;
}
.summary-num {
    font-family: 'Orbitron', monospace;
    font-size: 1.6rem;
    font-weight: 900;
}
.summary-lbl {
    font-size: 0.7rem;
    color: var(--dim);
    letter-spacing: 2px;
    font-family: 'Orbitron', monospace;
}

/* ── Streamlit tweaks ── */
.stTabs [data-baseweb="tab-list"] { background: transparent; gap: 8px; }
.stTabs [data-baseweb="tab"] {
    background: var(--card) !important;
    border: 1px solid var(--border) !important;
    border-radius: 6px !important;
    color: var(--dim) !important;
    font-family: 'Orbitron', monospace !important;
    font-size: 0.7rem !important;
    letter-spacing: 2px !important;
    padding: 8px 20px !important;
}
.stTabs [aria-selected="true"] {
    background: rgba(0,245,255,0.12) !important;
    border-color: var(--cyan) !important;
    color: var(--cyan) !important;
}
div[data-testid="stFileUploaderDropzone"] {
    background: rgba(0,245,255,0.03) !important;
    border: 1.5px dashed var(--cyan) !important;
    border-radius: 8px !important;
}
.stButton > button {
    background: linear-gradient(135deg, #0D3B66, #1565C0) !important;
    color: var(--cyan) !important;
    border: 1px solid var(--cyan) !important;
    font-family: 'Orbitron', monospace !important;
    font-size: 0.7rem !important;
    letter-spacing: 2px !important;
    border-radius: 6px !important;
}
.stSelectbox label, .stFileUploader label {
    color: var(--dim) !important;
    font-family: 'Orbitron', monospace !important;
    font-size: 0.65rem !important;
    letter-spacing: 2px !important;
}
.stExpander {
    background: var(--card) !important;
    border: 1px solid var(--border) !important;
    border-radius: 6px !important;
}
</style>
""", unsafe_allow_html=True)


# ── Helper functions ──────────────────────────────────────────────────────────

@st.cache_resource(show_spinner="Initializing neural network…")
def load_model_and_meta():
    """Load trained model and metadata. Cached so runs only once per session."""
    if not os.path.exists("pneumonia_model.pkl"):
        return None, None
    with open("pneumonia_model.pkl", "rb") as f:
        model = pickle.load(f)
    model.eval()
    metadata = {}
    if os.path.exists("model_metadata.json"):
        with open("model_metadata.json") as f:
            metadata = json.load(f)
    return model, metadata


def build_transform(input_size, mean, std):
    """Build preprocessing pipeline matching training conditions."""
    return transforms.Compose([
        transforms.Resize((input_size, input_size)),
        transforms.ToTensor(),
        transforms.Normalize(mean=mean, std=std),
    ])


def predict(model, image, transform, class_names):
    """Run inference on a single PIL Image. Returns label, confidence, all probs."""
    if image.mode != "RGB":
        image = image.convert("RGB")
    tensor = transform(image).unsqueeze(0)
    with torch.no_grad():
        logits = model(tensor)
        probs  = F.softmax(logits, dim=1)[0]
    pred_idx   = probs.argmax().item()
    pred_label = class_names[pred_idx]
    confidence = probs[pred_idx].item() * 100
    all_probs  = {cls: probs[i].item() * 100 for i, cls in enumerate(class_names)}
    return pred_label, confidence, all_probs


def get_severity(pneu_prob):
    """Map pneumonia probability to clinical severity level."""
    if pneu_prob < 30:   return "LOW",      "#00FF88"
    elif pneu_prob < 60: return "MODERATE",  "#FFD700"
    elif pneu_prob < 80: return "HIGH",      "#FF8C00"
    else:                return "CRITICAL",  "#FF3860"


def apply_processing(img, mode):
    """Apply selected image enhancement. Only affects display, not model input."""
    if mode == "Original":       return img
    if mode == "High Contrast":  return ImageEnhance.Contrast(img).enhance(2.2)
    if mode == "Sharpen":        return img.filter(ImageFilter.SHARPEN).filter(ImageFilter.SHARPEN)
    if mode == "Edge Highlight": return img.convert("L").filter(ImageFilter.FIND_EDGES).convert("RGB")
    if mode == "Equalized":      return ImageOps.equalize(img.convert("RGB"))
    return img


def prob_bar_html(label, pct, fill_class):
    return f"""
    <div class="prob-container">
        <div class="prob-label">{label}</div>
        <div class="prob-track">
            <div class="{fill_class}" style="width:{pct:.1f}%"></div>
        </div>
        <div class="prob-pct" style="color:{'#00FF88' if fill_class=='prob-fill-g' else '#FF3860'}">
            {pct:.1f}%
        </div>
    </div>"""


def severity_meter_html(pneu_prob, sev_label, sev_color):
    needle_pct = min(pneu_prob, 97)
    return f"""
    <div class="sev-section">
        <div class="sev-title">RISK ASSESSMENT METER</div>
        <div class="sev-track">
            <div class="sev-needle" style="left:{needle_pct}%"></div>
        </div>
        <div style="display:flex; justify-content:space-between; font-size:0.65rem;
                    color:var(--dim); font-family:'Orbitron',monospace; margin-top:4px;">
            <span>LOW</span><span>MODERATE</span><span>HIGH</span><span>CRITICAL</span>
        </div>
        <div>
            <span class="sev-badge"
                  style="background:rgba(0,0,0,0.4); border:2px solid {sev_color};
                         color:{sev_color}; margin-top:0.5rem;">
                SEVERITY: {sev_label}
            </span>
        </div>
    </div>"""


# ── Load model ────────────────────────────────────────────────────────────────
model, metadata = load_model_and_meta()

# Extract metadata values
input_size  = metadata.get("input_size",  224)     if metadata else 224
mean        = metadata.get("imagenet_mean", [0.485, 0.456, 0.406]) if metadata else [0.485, 0.456, 0.406]
std_norm    = metadata.get("imagenet_std",  [0.229, 0.224, 0.225]) if metadata else [0.229, 0.224, 0.225]
class_names = metadata.get("class_names",  ["NORMAL", "PNEUMONIA"]) if metadata else ["NORMAL", "PNEUMONIA"]
transform   = build_transform(input_size, mean, std_norm)


# ── SIDEBAR ───────────────────────────────────────────────────────────────────
with st.sidebar:
    st.markdown('<div class="sidebar-title">⚡ SYSTEM STATUS</div>', unsafe_allow_html=True)

    model_ok = model is not None
    meta_ok  = bool(metadata)

    st.markdown(f"""
    <div style="font-family:'Exo 2',sans-serif; font-size:0.85rem; margin-bottom:1rem;">
        <div style="margin-bottom:6px;">
            <span class="status-dot {'status-ok' if model_ok else 'status-err'}"></span>
            <span style="color:{'#00FF88' if model_ok else '#FF3860'}">
                {'MODEL LOADED' if model_ok else 'MODEL NOT FOUND'}
            </span>
        </div>
        <div>
            <span class="status-dot {'status-ok' if meta_ok else 'status-err'}"></span>
            <span style="color:{'#00FF88' if meta_ok else '#FF3860'}">
                {'METADATA OK' if meta_ok else 'METADATA MISSING'}
            </span>
        </div>
    </div>
    """, unsafe_allow_html=True)

    if not model_ok:
        st.warning("Place `pneumonia_model.pkl` in the project folder and restart.")

    # ── Model metrics ──
    if metadata:
        st.markdown('<div class="sidebar-title" style="margin-top:1rem;">📊 MODEL METRICS</div>',
                    unsafe_allow_html=True)
        metrics = [
            ("ARCHITECTURE", metadata.get("architecture", "ResNet18")),
            ("TEST ACCURACY", f"{metadata.get('test_accuracy', '-')}%"),
            ("AUC-ROC",       str(metadata.get("test_auc_roc", "-"))),
            ("EPOCHS",        str(metadata.get("epochs_trained", "-"))),
            ("VAL IMAGES",    str(metadata.get("val_set_size", "-"))),
            ("INPUT SIZE",    f"{input_size}×{input_size}"),
        ]
        rows_html = "".join(
            f'<div class="meta-row"><span class="meta-key">{k}</span><span class="meta-val">{v}</span></div>'
            for k, v in metrics
        )
        st.markdown(f'<div style="margin-bottom:1rem;">{rows_html}</div>', unsafe_allow_html=True)

    # ── Image processing mode ──
    st.markdown('<div class="sidebar-title">🔧 IMAGE PROCESSING</div>', unsafe_allow_html=True)
    processing_mode = st.selectbox(
        "Enhancement Mode",
        ["Original", "High Contrast", "Sharpen", "Edge Highlight", "Equalized"],
        label_visibility="collapsed",
    )

    st.markdown("""
    <div style="font-size:0.72rem; color:var(--dim); margin-top:0.5rem; line-height:1.6;">
        <b style="color:#00F5FF;">Original</b> — raw scan<br>
        <b style="color:#00F5FF;">High Contrast</b> — enhanced clarity<br>
        <b style="color:#00F5FF;">Sharpen</b> — edge sharpening<br>
        <b style="color:#00F5FF;">Edge Highlight</b> — structural edges<br>
        <b style="color:#00F5FF;">Equalized</b> — histogram eq.
    </div>
    """, unsafe_allow_html=True)

    st.markdown("---")
    st.markdown("""
    <div style="font-size:0.7rem; color:var(--dim); text-align:center; line-height:1.8;">
        <span style="color:#FF3860;">⚠</span> FOR RESEARCH USE ONLY<br>
        Not a certified medical device.<br>
        Always consult a physician.
    </div>
    """, unsafe_allow_html=True)


# ── HEADER ────────────────────────────────────────────────────────────────────
st.markdown('<div class="main-title">PNEUMOSCAN AI</div>', unsafe_allow_html=True)
st.markdown(
    '<div class="main-sub">CHEST X-RAY ANALYSIS SYSTEM &nbsp;·&nbsp; ResNet18 &nbsp;·&nbsp; TRANSFER LEARNING</div>',
    unsafe_allow_html=True,
)

# ── TABS ──────────────────────────────────────────────────────────────────────
tab_scan, tab_batch, tab_about = st.tabs(["🔬  SCAN", "📂  BATCH", "📋  ABOUT"])


# ════════════════════════════════════════════════════════════════════════════
# TAB 1 — SINGLE SCAN
# ════════════════════════════════════════════════════════════════════════════
with tab_scan:
    uploaded_files = st.file_uploader(
        "Drop chest X-ray(s) here or click to browse — select multiple images at once",
        type=["jpg", "jpeg", "png"],
        accept_multiple_files=True,
        key="single_upload",
    )

    if not uploaded_files:
        st.markdown("""
        <div style="text-align:center; padding:4rem 0; color:var(--dim);">
            <div style="font-size:3rem; margin-bottom:1rem;">🫁</div>
            <div style="font-family:'Orbitron',monospace; font-size:0.75rem;
                        letter-spacing:4px; color:var(--dim);">
                AWAITING SCAN INPUT
            </div>
        </div>
        """, unsafe_allow_html=True)
    else:
        if model is None:
            st.error("Model not loaded. Place `pneumonia_model.pkl` in the project folder.")
        else:
            for uploaded_file in uploaded_files:
                st.markdown(f"""
                <div style="font-family:'Orbitron',monospace; font-size:0.65rem;
                            color:var(--cyan); letter-spacing:3px; margin:1.2rem 0 0.4rem;">
                    ── SCAN: {uploaded_file.name} ──
                </div>
                """, unsafe_allow_html=True)

                raw_image     = Image.open(uploaded_file)
                display_image = apply_processing(raw_image, processing_mode)

                with st.spinner(f"Analyzing {uploaded_file.name}…"):
                    pred_label, confidence, all_probs = predict(model, raw_image, transform, class_names)
                    time.sleep(0.3)

                pneu_prob    = all_probs.get("PNEUMONIA", 0)
                norm_prob    = all_probs.get("NORMAL",    100)
                sev_label, sev_color = get_severity(pneu_prob)
                is_pneumonia = pred_label == "PNEUMONIA"

                img_col, res_col = st.columns([1, 1], gap="large")

                # ── Image column ──
                with img_col:
                    st.markdown('<div class="scan-card">', unsafe_allow_html=True)
                    st.image(display_image, use_container_width=True,
                             caption=f"Mode: {processing_mode}")
                    # File info
                    buf = io.BytesIO()
                    raw_image.save(buf, format="PNG")
                    size_kb = len(buf.getvalue()) / 1024
                    w, h = raw_image.size
                    st.markdown(f"""
                    <div style="font-size:0.75rem; color:var(--dim); margin-top:0.5rem;
                                font-family:'Exo 2',sans-serif;">
                        📄 {uploaded_file.name} &nbsp;|&nbsp;
                        {w}×{h}px &nbsp;|&nbsp; {size_kb:.1f} KB
                    </div>
                    """, unsafe_allow_html=True)
                    st.markdown("</div>", unsafe_allow_html=True)

                # ── Results column ──
                with res_col:
                    # Result banner
                    if is_pneumonia:
                        st.markdown(f"""
                        <div class="result-pneumonia">
                            <div class="result-label" style="color:#FF3860;">⚠ PNEUMONIA DETECTED</div>
                            <div style="font-size:0.8rem; color:#FF6B8A; margin-top:0.4rem;
                                        font-family:'Exo 2',sans-serif;">
                                Consult a radiologist immediately
                            </div>
                        </div>""", unsafe_allow_html=True)
                    else:
                        st.markdown(f"""
                        <div class="result-normal">
                            <div class="result-label" style="color:#00FF88;">✓ NORMAL</div>
                            <div style="font-size:0.8rem; color:#66FFB2; margin-top:0.4rem;
                                        font-family:'Exo 2',sans-serif;">
                                No pneumonia indicators detected
                            </div>
                        </div>""", unsafe_allow_html=True)

                    st.markdown("<br>", unsafe_allow_html=True)

                    # Metric badges
                    st.markdown(f"""
                    <div class="metric-row">
                        <div class="metric-badge">
                            <span class="metric-label">CONFIDENCE</span>
                            <span class="metric-value">{confidence:.1f}%</span>
                        </div>
                        <div class="metric-badge">
                            <span class="metric-label">PREDICTION</span>
                            <span class="metric-value" style="color:{'#FF3860' if is_pneumonia else '#00FF88'};">
                                {pred_label}
                            </span>
                        </div>
                        <div class="metric-badge">
                            <span class="metric-label">SEVERITY</span>
                            <span class="metric-value" style="color:{sev_color};">{sev_label}</span>
                        </div>
                    </div>
                    """, unsafe_allow_html=True)

                    # Probability bars
                    st.markdown(
                        prob_bar_html("NORMAL",    norm_prob, "prob-fill-g") +
                        prob_bar_html("PNEUMONIA", pneu_prob, "prob-fill-r"),
                        unsafe_allow_html=True,
                    )

                    # Severity meter
                    st.markdown(
                        severity_meter_html(pneu_prob, sev_label, sev_color),
                        unsafe_allow_html=True,
                    )

                    # Clinical notes
                    with st.expander(f"📋  CLINICAL NOTES — {uploaded_file.name}"):
                        if is_pneumonia:
                            st.markdown(f"""
                            <div style="font-family:'Exo 2',sans-serif; font-size:0.88rem;
                                        line-height:1.8; color:var(--text);">
                                <b style="color:#FF3860;">⚠ Pneumonia Indicators Present</b><br><br>
                                The model has detected radiological features consistent with pneumonia
                                (Confidence: <b>{confidence:.1f}%</b>, Severity: <b style="color:{sev_color};">{sev_label}</b>).<br><br>
                                <b style="color:#00F5FF;">Recommended Actions:</b><br>
                                • Urgent consultation with a qualified radiologist or pulmonologist.<br>
                                • Clinical correlation with patient symptoms (fever, cough, dyspnea).<br>
                                • Consider further imaging (CT scan) for severity assessment.<br>
                                • Laboratory investigations: CBC, CRP, sputum culture.<br><br>
                                <b style="color:#FFD700;">⚠ Disclaimer:</b> This AI tool is for research
                                purposes only and is NOT a substitute for professional medical diagnosis.
                            </div>
                            """, unsafe_allow_html=True)
                        else:
                            st.markdown(f"""
                            <div style="font-family:'Exo 2',sans-serif; font-size:0.88rem;
                                        line-height:1.8; color:var(--text);">
                                <b style="color:#00FF88;">✓ No Pneumonia Indicators Detected</b><br><br>
                                The model found no significant radiological features of pneumonia
                                (Confidence: <b>{confidence:.1f}%</b>).<br><br>
                                <b style="color:#00F5FF;">Note:</b><br>
                                • A normal AI result does not definitively rule out disease.<br>
                                • Clinical judgment and physician review remain essential.<br>
                                • If symptoms persist, please consult a healthcare professional.<br><br>
                                <b style="color:#FFD700;">⚠ Disclaimer:</b> This AI tool is for research
                                purposes only and is NOT a substitute for professional medical diagnosis.
                            </div>
                            """, unsafe_allow_html=True)


# ════════════════════════════════════════════════════════════════════════════
# TAB 2 — BATCH PROCESSING
# ════════════════════════════════════════════════════════════════════════════
with tab_batch:
    st.markdown('<div class="scan-card">', unsafe_allow_html=True)
    st.markdown("""
    <div style="font-family:'Orbitron',monospace; font-size:0.75rem; color:var(--cyan);
                letter-spacing:3px; margin-bottom:0.8rem;">
        📂 BATCH SCAN MODE
    </div>
    <div style="font-size:0.85rem; color:var(--dim); margin-bottom:1rem;">
        Upload multiple X-ray images for simultaneous analysis. Results can be exported as CSV.
    </div>
    """, unsafe_allow_html=True)

    # ── Upload mode toggle ──
    upload_mode = st.radio(
        "Upload Mode",
        ["📄 Select Individual Files", "📁 Load from Folder Path"],
        horizontal=True,
        label_visibility="collapsed",
    )
    st.markdown("</div>", unsafe_allow_html=True)

    batch_files = []   # list of PIL Images with .name attribute

    # ── Helper to wrap a PIL image so it looks like an UploadedFile ──
    class _ImgFile:
        def __init__(self, name, pil_img):
            self.name    = name
            self._img    = pil_img

    if upload_mode == "📄 Select Individual Files":
        _uploaded = st.file_uploader(
            "Upload X-ray images",
            type=["jpg", "jpeg", "png"],
            accept_multiple_files=True,
            key="batch_upload_files",
        )
        if _uploaded:
            batch_files = [_ImgFile(f.name, Image.open(f)) for f in _uploaded]

    else:
        # ── Folder path mode ──
        st.markdown("""
        <div style="font-size:0.8rem; color:var(--dim); margin-bottom:0.6rem; line-height:1.7;">
            Paste the full path to a folder on your computer. All
            <b style="color:#00F5FF;">JPG / PNG</b> images inside will be loaded automatically.<br>
            <span style="color:#6C7086; font-size:0.73rem;">
                Example: <code style="color:#00F5FF;">C:\\Users\\you\\Desktop\\xrays</code>
                &nbsp;or&nbsp;
                <code style="color:#00F5FF;">/home/you/xrays</code>
            </span>
        </div>
        """, unsafe_allow_html=True)

        folder_path = st.text_input(
            "Folder path",
            placeholder=r"C:\Users\you\Desktop\xrays",
            label_visibility="collapsed",
            key="folder_path_input",
        )

        if folder_path:
            folder_path = folder_path.strip().strip('"').strip("'")  # handle quoted paths
            if not os.path.isdir(folder_path):
                st.error(f"❌ Folder not found: `{folder_path}`")
            else:
                exts = {".jpg", ".jpeg", ".png"}
                found = sorted([
                    f for f in os.listdir(folder_path)
                    if os.path.splitext(f)[1].lower() in exts
                ])
                if not found:
                    st.warning("⚠ No JPG/PNG images found in that folder.")
                else:
                    st.markdown(f"""
                    <div style="font-size:0.8rem; color:#00FF88; margin:0.4rem 0 0.8rem;">
                        ✓ Found <b>{len(found)}</b> image(s) in
                        <code style="color:#00F5FF;">{folder_path}</code>
                    </div>
                    """, unsafe_allow_html=True)

                    load_errors = []
                    for fname in found:
                        fpath = os.path.join(folder_path, fname)
                        try:
                            img = Image.open(fpath)
                            img.load()          # force decode now to catch corrupt files
                            batch_files.append(_ImgFile(fname, img))
                        except Exception as e:
                            load_errors.append(fname)

                    if load_errors:
                        st.warning(f"Skipped {len(load_errors)} unreadable file(s): {', '.join(load_errors)}")

    if batch_files:
        if model is None:
            st.error("Model not loaded. Place `pneumonia_model.pkl` in the project folder.")
        else:
            results     = []
            prog_bar    = st.progress(0, text="Processing scans…")

            for i, f in enumerate(batch_files):
                img = f._img
                lbl, conf, probs = predict(model, img, transform, class_names)
                pneu = probs.get("PNEUMONIA", 0)
                sv, sc = get_severity(pneu)
                results.append({
                    "file":          f.name,
                    "prediction":    lbl,
                    "confidence":    round(conf, 2),
                    "pneumonia_prob":round(pneu, 2),
                    "normal_prob":   round(probs.get("NORMAL", 0), 2),
                    "severity":      sv,
                    "severity_color":sc,
                    "image":         img,
                })
                prog_bar.progress((i + 1) / len(batch_files),
                                  text=f"Processed {i+1}/{len(batch_files)} scans…")

            prog_bar.empty()

            # Summary dashboard
            n_pneu = sum(1 for r in results if r["prediction"] == "PNEUMONIA")
            n_norm = len(results) - n_pneu
            det_rate = (n_pneu / len(results) * 100) if results else 0

            st.markdown(f"""
            <div class="summary-row">
                <div class="summary-badge">
                    <div class="summary-num" style="color:var(--cyan);">{len(results)}</div>
                    <div class="summary-lbl">TOTAL</div>
                </div>
                <div class="summary-badge">
                    <div class="summary-num" style="color:#00FF88;">{n_norm}</div>
                    <div class="summary-lbl">NORMAL</div>
                </div>
                <div class="summary-badge">
                    <div class="summary-num" style="color:#FF3860;">{n_pneu}</div>
                    <div class="summary-lbl">PNEUMONIA</div>
                </div>
                <div class="summary-badge">
                    <div class="summary-num" style="color:#FFD700;">{det_rate:.0f}%</div>
                    <div class="summary-lbl">DETECTION RATE</div>
                </div>
            </div>
            """, unsafe_allow_html=True)

            # Image grid (3 columns)
            cols = st.columns(3)
            for idx, r in enumerate(results):
                with cols[idx % 3]:
                    border_color = "#FF3860" if r["prediction"] == "PNEUMONIA" else "#00FF88"
                    st.image(r["image"], use_container_width=True)
                    st.markdown(f"""
                    <div class="batch-card batch-{'pneumonia' if r['prediction']=='PNEUMONIA' else 'normal'}">
                        <div style="font-family:'Orbitron',monospace; font-size:0.65rem;
                                    color:{border_color}; letter-spacing:2px;">
                            {r['prediction']}
                        </div>
                        <div style="font-size:0.75rem; color:var(--dim); margin-top:3px;">
                            {r['confidence']:.1f}% &nbsp;|&nbsp;
                            <span style="color:{r['severity_color']};">{r['severity']}</span>
                        </div>
                        <div style="font-size:0.65rem; color:var(--dim); margin-top:2px;
                                    white-space:nowrap; overflow:hidden; text-overflow:ellipsis;">
                            {r['file']}
                        </div>
                    </div>
                    """, unsafe_allow_html=True)

            # CSV Export
            csv_buf = io.StringIO()
            writer  = csv.DictWriter(
                csv_buf,
                fieldnames=["file", "prediction", "confidence", "pneumonia_prob",
                            "normal_prob", "severity"],
            )
            writer.writeheader()
            for r in results:
                writer.writerow({k: r[k] for k in
                                 ["file", "prediction", "confidence",
                                  "pneumonia_prob", "normal_prob", "severity"]})

            st.download_button(
                label="⬇  Download CSV Report",
                data=csv_buf.getvalue(),
                file_name="pneumoscan_batch_results.csv",
                mime="text/csv",
            )


# ════════════════════════════════════════════════════════════════════════════
# TAB 3 — ABOUT
# ════════════════════════════════════════════════════════════════════════════
with tab_about:
    col1, col2 = st.columns(2, gap="large")

    with col1:
        st.markdown("""
        <div class="scan-card">
            <div style="font-family:'Orbitron',monospace; font-size:0.8rem; color:var(--cyan);
                        letter-spacing:3px; margin-bottom:1rem;">🧠 MODEL ARCHITECTURE</div>
            <div style="font-size:0.88rem; line-height:1.9; color:var(--text);">
                <b style="color:#00F5FF;">Base Model:</b> ResNet18 (ImageNet pretrained)<br>
                <b style="color:#00F5FF;">Strategy:</b> Transfer Learning — frozen backbone<br>
                <b style="color:#00F5FF;">Custom Head:</b> Dropout(0.4) → Linear(512→256) → ReLU → Dropout(0.3) → Linear(256→2)<br>
                <b style="color:#00F5FF;">Trainable Params:</b> ~66,000<br>
                <b style="color:#00F5FF;">Input:</b> 224×224 RGB (ImageNet normalization)<br>
                <b style="color:#00F5FF;">Output:</b> Softmax — NORMAL / PNEUMONIA<br>
            </div>
        </div>
        """, unsafe_allow_html=True)

    with col2:
        st.markdown("""
        <div class="scan-card">
            <div style="font-family:'Orbitron',monospace; font-size:0.8rem; color:var(--cyan);
                        letter-spacing:3px; margin-bottom:1rem;">⚙ TRAINING SETUP</div>
            <div style="font-size:0.88rem; line-height:1.9; color:var(--text);">
                <b style="color:#00F5FF;">Dataset:</b> Kaggle Chest X-Ray (5,863 images)<br>
                <b style="color:#00F5FF;">Optimizer:</b> Adam (lr=1e-3, wd=1e-4)<br>
                <b style="color:#00F5FF;">Scheduler:</b> ReduceLROnPlateau<br>
                <b style="color:#00F5FF;">Imbalance Fix:</b> WeightedRandomSampler<br>
                <b style="color:#00F5FF;">Early Stopping:</b> Patience = 4<br>
                <b style="color:#00F5FF;">Platform:</b> Google Colab T4 GPU<br>
            </div>
        </div>
        """, unsafe_allow_html=True)

    st.markdown("""
    <div class="scan-card" style="border-top-color:#FF3860; margin-top:1rem;">
        <div style="font-family:'Orbitron',monospace; font-size:0.8rem; color:#FF3860;
                    letter-spacing:3px; margin-bottom:0.8rem;">⚠ MEDICAL DISCLAIMER</div>
        <div style="font-size:0.88rem; line-height:1.8; color:var(--text);">
            PneumoScan AI is developed strictly for <b>research and educational purposes</b>.<br>
            It is <b>NOT</b> a certified medical device and must <b>NOT</b> be used for clinical
            diagnosis or patient care decisions.<br>
            Always consult a licensed physician or radiologist for medical evaluation.<br><br>
            <span style="color:var(--dim);">
                Group 9 &nbsp;|&nbsp; BSAI-212 &nbsp;|&nbsp; NUML Islamabad &nbsp;|&nbsp;
                <a href="https://github.com/Bisma6405/Pneumonia_detector"
                   style="color:#00F5FF;">GitHub Repository</a>
            </span>
        </div>
    </div>
    """, unsafe_allow_html=True)