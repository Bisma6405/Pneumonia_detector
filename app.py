import streamlit as st
import pickle
import json
import numpy as np
from PIL import Image, ImageFilter, ImageEnhance
import torch
import torch.nn.functional as F
from torchvision import transforms
import os
import io
import base64
import time
import csv

# ── Page config ──────────────────────────────────────────────────────────────
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
    --cyan:#00F5FF; --green:#00FF88; --red:#FF3860;
    --yellow:#FFD700; --bg:#020B18; --surface:#071525;
    --card:#0A1F35; --border:#0D3A5C; --text:#C8E6FA; --dim:#5A8CAA;
}
.stApp {
    background: var(--bg) !important;
    background-image:
        radial-gradient(ellipse 80% 50% at 50% -10%, rgba(0,245,255,0.07) 0%, transparent 60%),
        repeating-linear-gradient(0deg, transparent, transparent 39px, rgba(0,245,255,0.03) 40px),
        repeating-linear-gradient(90deg, transparent, transparent 39px, rgba(0,245,255,0.03) 40px) !important;
    font-family: 'Exo 2', sans-serif !important;
    color: var(--text) !important;
}
#MainMenu, footer, header { visibility: hidden; }
.block-container { padding-top: 1.5rem !important; max-width: 1300px !important; }
[data-testid="stSidebar"] { background: var(--surface) !important; border-right: 1px solid var(--border) !important; }
[data-testid="stSidebar"] * { color: var(--text) !important; }
h1,h2,h3 { font-family: 'Orbitron', monospace !important; }

.scan-card {
    background: var(--card); border: 1px solid var(--border);
    border-radius: 12px; padding: 1.5rem;
    box-shadow: 0 0 30px rgba(0,245,255,0.05), inset 0 1px 0 rgba(255,255,255,0.03);
    position: relative; overflow: hidden; margin-bottom: 1rem;
}
.scan-card::before {
    content:''; position:absolute; top:0; left:0; right:0; height:2px;
    background: linear-gradient(90deg, transparent, var(--cyan), transparent);
}
.card-title {
    font-family: 'Orbitron', monospace; font-size: 0.75rem;
    color: #00F5FF; letter-spacing: 0.12em; margin-bottom: 1rem;
}
.metric-row { display:flex; gap:1rem; margin-bottom:1.2rem; flex-wrap:wrap; }
.metric-badge {
    display:inline-flex; flex-direction:column; align-items:center;
    border-radius:10px; padding:0.7rem 1.2rem; min-width:110px;
    background: rgba(0,245,255,0.06); border: 1px solid rgba(0,245,255,0.2);
}
.metric-val {
    font-family:'Orbitron',monospace; font-size:1.4rem; font-weight:700;
    color:#00F5FF; text-shadow: 0 0 10px rgba(0,245,255,0.5);
}
.metric-lbl { font-size:0.65rem; letter-spacing:0.12em; color:var(--dim); text-transform:uppercase; margin-top:2px; }

.result-normal {
    background: linear-gradient(135deg, rgba(0,255,136,0.12), rgba(0,255,136,0.04));
    border: 1px solid rgba(0,255,136,0.4); border-radius:12px; padding:1.2rem 1.8rem;
    text-align:center; box-shadow: 0 0 30px rgba(0,255,136,0.15); margin-bottom:1rem;
}
.result-pneumonia {
    background: linear-gradient(135deg, rgba(255,56,96,0.14), rgba(255,56,96,0.04));
    border: 1px solid rgba(255,56,96,0.5); border-radius:12px; padding:1.2rem 1.8rem;
    text-align:center; box-shadow: 0 0 30px rgba(255,56,96,0.2); margin-bottom:1rem;
    animation: pulse-red 2s ease-in-out infinite;
}
@keyframes pulse-red {
    0%,100% { box-shadow: 0 0 30px rgba(255,56,96,0.2); }
    50%      { box-shadow: 0 0 50px rgba(255,56,96,0.4); }
}
.result-label { font-family:'Orbitron',monospace; font-size:2rem; font-weight:900; letter-spacing:0.1em; }
.label-green  { color:#00FF88; text-shadow: 0 0 20px rgba(0,255,136,0.6); }
.label-red    { color:#FF3860; text-shadow: 0 0 20px rgba(255,56,96,0.7); }

.prob-label-row { display:flex; justify-content:space-between; font-size:0.8rem;
    font-family:'Orbitron',monospace; letter-spacing:0.08em; margin-bottom:4px; }
.prob-track { background:rgba(255,255,255,0.05); border-radius:6px; height:10px; overflow:hidden; }
.prob-fill-g { height:100%; background:linear-gradient(90deg,#00C866,#00FF88);
    border-radius:6px; box-shadow:0 0 8px rgba(0,255,136,0.5); }
.prob-fill-r { height:100%; background:linear-gradient(90deg,#CC1133,#FF3860);
    border-radius:6px; box-shadow:0 0 8px rgba(255,56,96,0.5); }

.severity-box { background:rgba(0,0,0,0.3); border:1px solid var(--border);
    border-radius:10px; padding:1rem; margin-top:0.8rem; }
.sev-title { font-family:'Orbitron',monospace; font-size:0.65rem; color:#5A8CAA;
    letter-spacing:0.1em; margin-bottom:0.5rem; }
.sev-track { height:16px; border-radius:8px; position:relative;
    background:linear-gradient(90deg,#00FF88 0%,#FFD700 40%,#FF8C00 70%,#FF3860 100%); }
.sev-needle { position:absolute; top:-4px; width:4px; height:24px; background:white;
    border-radius:2px; box-shadow:0 0 8px rgba(255,255,255,0.8); transform:translateX(-50%); }
.sev-labels { display:flex; justify-content:space-between; font-size:0.6rem;
    color:#2A5A7A; font-family:'Orbitron',monospace; margin-top:2px; }

[data-testid="stFileUploader"] {
    background:rgba(0,245,255,0.03) !important;
    border:2px dashed rgba(0,245,255,0.25) !important; border-radius:12px !important;
}
.stButton > button {
    background:linear-gradient(135deg,rgba(0,245,255,0.15),rgba(0,245,255,0.05)) !important;
    border:1px solid #00F5FF !important; color:#00F5FF !important;
    font-family:'Orbitron',monospace !important; font-size:0.75rem !important;
    letter-spacing:0.1em !important; border-radius:8px !important;
}
[data-testid="stExpander"] { background:var(--card) !important; border:1px solid var(--border) !important; border-radius:10px !important; }
[data-baseweb="tab-list"] { background:transparent !important; border-bottom:1px solid var(--border) !important; }
[data-baseweb="tab"] { font-family:'Orbitron',monospace !important; font-size:0.7rem !important; letter-spacing:0.1em !important; color:var(--dim) !important; }
[aria-selected="true"] { color:var(--cyan) !important; border-bottom:2px solid var(--cyan) !important; }
::-webkit-scrollbar { width:6px; }
::-webkit-scrollbar-track { background:var(--bg); }
::-webkit-scrollbar-thumb { background:var(--border); border-radius:3px; }
</style>
""", unsafe_allow_html=True)


# ── Helpers ───────────────────────────────────────────────────────────────────
@st.cache_resource(show_spinner="Initializing neural network…")
def load_model_and_meta():
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
    return transforms.Compose([
        transforms.Resize((input_size, input_size)),
        transforms.ToTensor(),
        transforms.Normalize(mean=mean, std=std),
    ])

def predict(model, image, transform, class_names):
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
    if pneu_prob < 30:   return "LOW",      "#00FF88"
    elif pneu_prob < 60: return "MODERATE", "#FFD700"
    elif pneu_prob < 80: return "HIGH",     "#FF8C00"
    else:                return "CRITICAL", "#FF3860"

def apply_processing(img, mode):
    if mode == "Original":      return img
    if mode == "High Contrast": return ImageEnhance.Contrast(img).enhance(2.2)
    if mode == "Sharpen":       return img.filter(ImageFilter.SHARPEN).filter(ImageFilter.SHARPEN)
    if mode == "Edge Highlight":
        return img.convert("L").filter(ImageFilter.FIND_EDGES).convert("RGB")
    if mode == "Equalized":
        import PIL.ImageOps
        return PIL.ImageOps.equalize(img.convert("RGB"))
    return img

def html_metric_badge(val, lbl, val_color="#00F5FF", border_color="rgba(0,245,255,0.2)"):
    return (
        "<div class='metric-badge' style='border-color:" + border_color + ";'>"
        "<span class='metric-val' style='color:" + val_color + ";'>" + val + "</span>"
        "<span class='metric-lbl'>" + lbl + "</span>"
        "</div>"
    )

def html_prob_bar(label, pct, fill_class, label_color):
    width = str(round(pct, 1)) + "%"
    pct_str = str(round(pct, 1)) + "%"
    return (
        "<div style='margin-bottom:0.8rem;'>"
        "<div class='prob-label-row'>"
        "<span style='color:" + label_color + ";'>" + label + "</span>"
        "<span style='color:" + label_color + ";'>" + pct_str + "</span>"
        "</div>"
        "<div class='prob-track'>"
        "<div class='" + fill_class + "' style='width:" + width + ";'></div>"
        "</div>"
        "</div>"
    )


# ── Load ──────────────────────────────────────────────────────────────────────
model, metadata = load_model_and_meta()

# ── Header ────────────────────────────────────────────────────────────────────
st.markdown(
    "<div style='text-align:center;padding:1rem 0 0.5rem;'>"
    "<div style='font-family:Orbitron,monospace;font-size:2.4rem;font-weight:900;"
    "background:linear-gradient(90deg,#00F5FF,#00FF88);"
    "-webkit-background-clip:text;-webkit-text-fill-color:transparent;"
    "letter-spacing:0.12em;'>PNEUMOSCAN AI</div>"
    "<div style='color:#5A8CAA;font-size:0.8rem;letter-spacing:0.25em;"
    "font-family:Orbitron,monospace;margin-top:4px;'>"
    "CHEST X-RAY ANALYSIS SYSTEM &nbsp;·&nbsp; ResNet18 &nbsp;·&nbsp; TRANSFER LEARNING"
    "</div></div>",
    unsafe_allow_html=True
)

# ── Sidebar ───────────────────────────────────────────────────────────────────
with st.sidebar:
    st.markdown(
        "<div style='font-family:Orbitron,monospace;font-size:0.9rem;color:#00F5FF;"
        "letter-spacing:0.1em;padding:0.5rem 0;border-bottom:1px solid #0D3A5C;margin-bottom:1rem;'>"
        "&#9672; SYSTEM STATUS</div>",
        unsafe_allow_html=True
    )

    model_ok = model is not None
    meta_ok  = bool(metadata)

    nn_color  = "#00FF88" if model_ok else "#FF3860"
    nn_dot    = "&#9679;" if model_ok else "&#9675;"
    nn_status = "LOADED"  if model_ok else "MISSING"
    md_color  = "#00FF88" if meta_ok  else "#FFD700"
    md_dot    = "&#9679;" if meta_ok  else "&#9675;"
    md_status = "LOADED"  if meta_ok  else "NOT FOUND"

    st.markdown(
        "<div style='font-size:0.78rem;font-family:Exo 2,sans-serif;line-height:2.2;'>"
        "<span style='color:" + nn_color + ";'>" + nn_dot + "</span>"
        "&nbsp; Neural Network &nbsp;"
        "<span style='color:" + nn_color + ";font-family:Orbitron,monospace;font-size:0.7rem;'>" + nn_status + "</span><br>"
        "<span style='color:" + md_color + ";'>" + md_dot + "</span>"
        "&nbsp; Metadata &nbsp;"
        "<span style='color:" + md_color + ";font-family:Orbitron,monospace;font-size:0.7rem;'>" + md_status + "</span>"
        "</div>",
        unsafe_allow_html=True
    )

    st.markdown("<div style='height:1rem'></div>", unsafe_allow_html=True)

    if metadata:
        st.markdown(
            "<div style='font-family:Orbitron,monospace;font-size:0.75rem;"
            "color:#00F5FF;letter-spacing:0.1em;margin-bottom:0.8rem;'>&#9672; MODEL METRICS</div>",
            unsafe_allow_html=True
        )
        metrics_list = [
            ("ARCHITECTURE",  metadata.get("architecture", "ResNet18")),
            ("TEST ACCURACY", str(metadata.get("test_accuracy", "—")) + "%"),
            ("AUC-ROC",       str(metadata.get("test_auc_roc", "—"))),
            ("EPOCHS",        str(metadata.get("epochs_trained", "—"))),
            ("VAL IMAGES",    str(metadata.get("val_set_size", "—"))),
        ]
        rows_html = ""
        for lbl, val in metrics_list:
            rows_html += (
                "<div style='display:flex;justify-content:space-between;padding:4px 0;"
                "border-bottom:1px solid rgba(13,58,92,0.5);font-size:0.72rem;'>"
                "<span style='color:#5A8CAA;font-family:Orbitron,monospace;letter-spacing:0.06em;'>" + lbl + "</span>"
                "<span style='color:#00F5FF;font-family:Orbitron,monospace;font-weight:700;'>" + val + "</span>"
                "</div>"
            )
        st.markdown(rows_html, unsafe_allow_html=True)

    st.markdown("<div style='height:1rem'></div>", unsafe_allow_html=True)
    st.markdown(
        "<div style='font-family:Orbitron,monospace;font-size:0.75rem;"
        "color:#00F5FF;letter-spacing:0.1em;margin-bottom:0.8rem;'>&#9672; IMAGE PROCESSING</div>",
        unsafe_allow_html=True
    )
    processing_mode = st.selectbox(
        "Enhancement Mode",
        ["Original", "High Contrast", "Sharpen", "Edge Highlight", "Equalized"],
        label_visibility="collapsed"
    )

    st.markdown("<div style='height:1rem'></div>", unsafe_allow_html=True)
    st.markdown(
        "<div style='font-size:0.65rem;color:#2A5A7A;font-family:Exo 2,sans-serif;"
        "line-height:1.6;padding:0.8rem;background:rgba(0,0,0,0.3);"
        "border-radius:8px;border:1px solid #0D3A5C;'>"
        "&#9888; FOR RESEARCH USE ONLY<br>"
        "Not a substitute for professional medical diagnosis."
        "</div>",
        unsafe_allow_html=True
    )

# ── Model missing ─────────────────────────────────────────────────────────────
if model is None:
    st.markdown(
        "<div class='scan-card' style='text-align:center;padding:3rem;margin-top:2rem;'>"
        "<div style='font-size:3rem;'>&#9888;</div>"
        "<div style='font-family:Orbitron,monospace;color:#FF3860;font-size:1.1rem;"
        "letter-spacing:0.1em;margin:1rem 0;'>MODEL NOT FOUND</div>"
        "<div style='color:#5A8CAA;font-size:0.85rem;'>"
        "Place <code style='color:#00F5FF'>pneumonia_model.pkl</code> next to "
        "<code style='color:#00F5FF'>app.py</code> and restart.</div>"
        "</div>",
        unsafe_allow_html=True
    )
    st.stop()

# ── Tabs ──────────────────────────────────────────────────────────────────────
tab_scan, tab_batch, tab_about = st.tabs(["🔬  SCAN", "📂  BATCH", "📋  ABOUT"])

# ════════════════════════════════
# TAB 1 — SINGLE SCAN
# ════════════════════════════════
with tab_scan:
    st.markdown("<div style='height:0.5rem'></div>", unsafe_allow_html=True)
    upload_col, _ = st.columns([2, 1])
    with upload_col:
        uploaded_file = st.file_uploader(
            "Drop chest X-ray here or click to browse",
            type=["jpg", "jpeg", "png"],
        )

    if uploaded_file is None:
        st.markdown(
            "<div style='text-align:center;padding:3rem 0;'>"
            "<div style='font-size:4rem;margin-bottom:1rem;opacity:0.15;'>&#129753;</div>"
            "<div style='font-family:Orbitron,monospace;font-size:0.85rem;"
            "letter-spacing:0.15em;color:#1A4060;'>AWAITING SCAN INPUT</div>"
            "</div>",
            unsafe_allow_html=True
        )
    else:
        raw_image     = Image.open(uploaded_file)
        display_image = apply_processing(raw_image, processing_mode)

        input_size  = metadata.get("input_size", 224)
        mean        = metadata.get("imagenet_mean", [0.485, 0.456, 0.406])
        std_norm    = metadata.get("imagenet_std",  [0.229, 0.224, 0.225])
        class_names = metadata.get("class_names",  ["NORMAL", "PNEUMONIA"])
        transform   = build_transform(input_size, mean, std_norm)

        with st.spinner("Analyzing scan…"):
            pred_label, confidence, all_probs = predict(model, raw_image, transform, class_names)
            time.sleep(0.3)

        pneu_prob      = all_probs.get("PNEUMONIA", 0)
        norm_prob      = all_probs.get("NORMAL", 100)
        sev_label, sev_color = get_severity(pneu_prob)
        is_pneumonia   = (pred_label == "PNEUMONIA")
        needle_pct     = min(pneu_prob, 97)

        img_col, res_col = st.columns([1, 1], gap="large")

        # ── Image card ────────────────────────────────────────────────────────
        with img_col:
            proc_upper = processing_mode.upper()
            fname      = uploaded_file.name
            w_px       = str(raw_image.width)
            h_px       = str(raw_image.height)
            kb_size    = str(uploaded_file.size // 1024)

            st.markdown(
                "<div class='scan-card'>"
                "<div class='card-title'>&#9672; SCAN IMAGE &nbsp;·&nbsp; " + proc_upper + "</div>",
                unsafe_allow_html=True
            )
            st.image(display_image, use_container_width=True)
            st.markdown(
                "<div style='font-size:0.7rem;color:#2A5A7A;margin-top:0.5rem;"
                "font-family:Exo 2,sans-serif;'>"
                + fname + " &nbsp;&#183;&nbsp; " + w_px + "x" + h_px + "px &nbsp;&#183;&nbsp; " + kb_size + " KB"
                "</div></div>",
                unsafe_allow_html=True
            )

        # ── Result card ───────────────────────────────────────────────────────
        with res_col:
            # Banner
            if is_pneumonia:
                st.markdown(
                    "<div class='result-pneumonia'>"
                    "<div style='font-size:1.8rem;margin-bottom:0.3rem;'>&#128308;</div>"
                    "<div class='result-label label-red'>PNEUMONIA</div>"
                    "<div style='color:#FF8C8C;font-size:0.75rem;letter-spacing:0.15em;"
                    "margin-top:6px;font-family:Orbitron,monospace;'>"
                    "DETECTED &nbsp;&#183;&nbsp; SEVERITY: "
                    "<span style='color:" + sev_color + ";'>" + sev_label + "</span>"
                    "</div></div>",
                    unsafe_allow_html=True
                )
            else:
                st.markdown(
                    "<div class='result-normal'>"
                    "<div style='font-size:1.8rem;margin-bottom:0.3rem;'>&#128994;</div>"
                    "<div class='result-label label-green'>NORMAL</div>"
                    "<div style='color:#66FF99;font-size:0.75rem;letter-spacing:0.15em;"
                    "margin-top:6px;font-family:Orbitron,monospace;'>"
                    "NO PNEUMONIA DETECTED"
                    "</div></div>",
                    unsafe_allow_html=True
                )

            # Confidence card
            conf_str   = str(round(confidence)) + "%"
            pred_short = pred_label[:4]
            sev_short  = sev_label[:4]
            needle_str = str(round(needle_pct, 1)) + "%"

            badge1 = html_metric_badge(conf_str,   "Confidence")
            badge2 = html_metric_badge(pred_short, "Prediction",
                                       val_color=sev_color,
                                       border_color="rgba(255,56,96,0.3)" if is_pneumonia else "rgba(0,255,136,0.3)")
            badge3 = html_metric_badge(sev_short,  "Severity",
                                       val_color=sev_color,
                                       border_color="rgba(255,215,0,0.3)")

            bar_norm = html_prob_bar("NORMAL",    norm_prob, "prob-fill-g", "#00FF88")
            bar_pneu = html_prob_bar("PNEUMONIA", pneu_prob, "prob-fill-r", "#FF3860")

            st.markdown(
                "<div class='scan-card'>"
                "<div class='card-title'>&#9672; CONFIDENCE ANALYSIS</div>"
                "<div class='metric-row'>" + badge1 + badge2 + badge3 + "</div>"
                + bar_norm + bar_pneu +
                "<div class='severity-box'>"
                "<div class='sev-title'>RISK METER</div>"
                "<div class='sev-track'>"
                "<div class='sev-needle' style='left:" + needle_str + ";'></div>"
                "</div>"
                "<div class='sev-labels'>"
                "<span>LOW</span><span>MODERATE</span><span>HIGH</span><span>CRITICAL</span>"
                "</div></div></div>",
                unsafe_allow_html=True
            )

            # Clinical notes
            with st.expander("&#9672; CLINICAL NOTES & INTERPRETATION"):
                pneu_str = str(round(pneu_prob, 1)) + "%"
                norm_str = str(round(norm_prob, 1)) + "%"
                if is_pneumonia:
                    st.markdown(
                        "<div style='font-size:0.82rem;line-height:1.8;color:#C8E6FA;'>"
                        "The model flagged this scan with <b style='color:#FF3860;'>" + pneu_str + "</b> "
                        "probability of pneumonia. Severity: <b style='color:" + sev_color + ";'>" + sev_label + "</b>.<br><br>"
                        "<b style='color:#00F5FF;'>Recommended actions:</b><br>"
                        "&#8226; Correlate with patient symptoms (fever, cough, dyspnea)<br>"
                        "&#8226; Consider CBC, CRP, and procalcitonin<br>"
                        "&#8226; Consult a radiologist for confirmatory read<br>"
                        "&#8226; Do not delay treatment based on AI result alone"
                        "</div>",
                        unsafe_allow_html=True
                    )
                else:
                    st.markdown(
                        "<div style='font-size:0.82rem;line-height:1.8;color:#C8E6FA;'>"
                        "The model classified this scan as <b style='color:#00FF88;'>NORMAL</b> with "
                        "<b style='color:#00FF88;'>" + norm_str + "</b> confidence.<br><br>"
                        "<b style='color:#00F5FF;'>Notes:</b><br>"
                        "&#8226; No consolidations or opacities detected by the model<br>"
                        "&#8226; If symptoms persist, clinical judgment takes precedence<br>"
                        "&#8226; A normal AI result does not exclude early-stage disease"
                        "</div>",
                        unsafe_allow_html=True
                    )


# ════════════════════════════════
# TAB 2 — BATCH SCAN
# ════════════════════════════════
with tab_batch:
    st.markdown(
        "<div style='font-family:Orbitron,monospace;font-size:0.8rem;color:#5A8CAA;"
        "letter-spacing:0.12em;margin:0.5rem 0 1rem;'>UPLOAD MULTIPLE X-RAYS FOR BATCH ANALYSIS</div>",
        unsafe_allow_html=True
    )
    batch_files = st.file_uploader(
        "Upload multiple X-ray images",
        type=["jpg", "jpeg", "png"],
        accept_multiple_files=True,
    )

    if batch_files:
        input_size  = metadata.get("input_size", 224)
        mean        = metadata.get("imagenet_mean", [0.485, 0.456, 0.406])
        std_norm    = metadata.get("imagenet_std",  [0.229, 0.224, 0.225])
        class_names = metadata.get("class_names",  ["NORMAL", "PNEUMONIA"])
        transform   = build_transform(input_size, mean, std_norm)

        results  = []
        prog_bar = st.progress(0, text="Scanning images…")
        for i, f in enumerate(batch_files):
            img = Image.open(f)
            lbl, conf, probs = predict(model, img, transform, class_names)
            pneu = probs.get("PNEUMONIA", 0)
            sv, sc = get_severity(pneu)
            results.append({"file": f.name, "prediction": lbl, "confidence": conf,
                             "pneumonia_prob": pneu, "severity": sv,
                             "severity_color": sc, "image": img})
            prog_bar.progress((i + 1) / len(batch_files), text="Scanned " + str(i+1) + "/" + str(len(batch_files)))
        prog_bar.empty()

        n_pneu     = sum(1 for r in results if r["prediction"] == "PNEUMONIA")
        n_norm     = len(results) - n_pneu
        detect_pct = str(round(n_pneu / len(results) * 100)) + "%"

        b1 = html_metric_badge(str(len(results)), "Total Scans")
        b2 = html_metric_badge(str(n_norm),  "Normal",    val_color="#00FF88", border_color="rgba(0,255,136,0.3)")
        b3 = html_metric_badge(str(n_pneu),  "Pneumonia", val_color="#FF3860", border_color="rgba(255,56,96,0.3)")
        b4 = html_metric_badge(detect_pct,   "Detection", val_color="#FFD700", border_color="rgba(255,215,0,0.3)")

        st.markdown(
            "<div class='metric-row' style='margin:1rem 0;'>" + b1 + b2 + b3 + b4 + "</div>",
            unsafe_allow_html=True
        )

        cols = st.columns(3)
        for idx, r in enumerate(results):
            with cols[idx % 3]:
                clr      = "#FF3860" if r["prediction"] == "PNEUMONIA" else "#00FF88"
                conf_txt = str(round(r["confidence"], 1)) + "%"
                border   = clr + "33"
                st.image(r["image"], use_container_width=True)
                st.markdown(
                    "<div style='background:rgba(0,0,0,0.4);border:1px solid " + border + ";"
                    "border-radius:8px;padding:0.5rem 0.7rem;margin-bottom:0.8rem;'>"
                    "<div style='font-size:0.65rem;color:#5A8CAA;overflow:hidden;"
                    "text-overflow:ellipsis;white-space:nowrap;'>" + r["file"] + "</div>"
                    "<div style='font-family:Orbitron,monospace;font-size:0.85rem;"
                    "color:" + clr + ";font-weight:700;'>" + r["prediction"] + "</div>"
                    "<div style='font-size:0.7rem;color:#5A8CAA;'>" + conf_txt + " conf &nbsp;&#183;&nbsp; "
                    "<span style='color:" + r["severity_color"] + ";'>" + r["severity"] + "</span>"
                    "</div></div>",
                    unsafe_allow_html=True
                )

        csv_buf = io.StringIO()
        writer  = csv.DictWriter(csv_buf,
            fieldnames=["file", "prediction", "confidence", "pneumonia_prob", "severity"])
        writer.writeheader()
        for r in results:
            writer.writerow({k: r[k] for k in
                ["file", "prediction", "confidence", "pneumonia_prob", "severity"]})
        st.download_button(
            label="⬇  DOWNLOAD CSV REPORT",
            data=csv_buf.getvalue(),
            file_name="pneumoscan_batch_results.csv",
            mime="text/csv",
        )


# ════════════════════════════════
# TAB 3 — ABOUT
# ════════════════════════════════
with tab_about:
    st.markdown(
        "<div class='scan-card' style='margin-top:0.5rem;'>"
        "<div class='card-title'>&#9672; ABOUT PNEUMOSCAN AI</div>"
        "<div style='font-size:0.85rem;line-height:1.9;color:#C8E6FA;'>"
        "PneumoScan AI is a deep learning-based chest X-ray analysis tool built on "
        "<b style='color:#00F5FF;'>ResNet18</b> pretrained on ImageNet and fine-tuned on the "
        "Kaggle Chest X-Ray (Pneumonia) dataset.<br><br>"
        "<b style='color:#00F5FF;'>Architecture:</b> Frozen convolutional backbone + "
        "custom 2-layer head (Dropout 0.4 &rarr; Linear &rarr; ReLU &rarr; Dropout 0.3 &rarr; Linear).<br>"
        "Only ~66K parameters trained out of 11M total.<br><br>"
        "<b style='color:#00F5FF;'>Training:</b> Adam (lr=1e-3), CrossEntropyLoss, "
        "ReduceLROnPlateau, WeightedRandomSampler, Early stopping (patience=4).<br><br>"
        "<b style='color:#00F5FF;'>Input:</b> 224x224 RGB, ImageNet normalization<br>"
        "<b style='color:#00F5FF;'>Output:</b> Binary &mdash; NORMAL vs PNEUMONIA"
        "</div></div>",
        unsafe_allow_html=True
    )
    st.markdown(
        "<div class='scan-card'>"
        "<div class='card-title' style='color:#FFD700;'>&#9672; DISCLAIMER</div>"
        "<div style='font-size:0.82rem;line-height:1.8;color:#C8E6FA;'>"
        "This software is provided <b>strictly for research and educational purposes</b>. "
        "It is <b style='color:#FF3860;'>NOT</b> approved for clinical diagnosis, "
        "medical decision-making, or patient care. Always seek the advice of qualified "
        "medical professionals. AI predictions may be incorrect and should never replace "
        "expert radiological evaluation."
        "</div></div>",
        unsafe_allow_html=True
    )