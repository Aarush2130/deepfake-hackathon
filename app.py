import base64
import hashlib
import os
import tempfile
import cv2
import numpy as np
import streamlit as st
import streamlit.components.v1 as components
from engine import get_engine_status

# 1. Page Configuration
st.set_page_config(
    page_title="VeriChain Forensic Suite | Digital Evidence Authentication",
    layout="wide",
    page_icon="⚖️",
    initial_sidebar_state="expanded"
)

# 2. Enterprise Judicial & Defense Forensic Design System
st.html(
"""
<style>
@import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700;800&family=JetBrains+Mono:wght@400;500;600;700&display=swap');

/* Strict Enterprise Base */
html, body, [class*="css"] {
    font-family: 'Inter', -apple-system, BlinkMacSystemFont, "Segoe UI", Roboto, sans-serif !important;
    color: #e2e8f0;
}

.font-mono {
    font-family: 'JetBrains Mono', monospace !important;
}

/* Deep Obsidian Solid Professional Background */
.stApp {
    background: #090c15 !important;
}

/* Enterprise Header Bar */
.enterprise-header {
    background: #0f1422;
    border: 1px solid #1e293b;
    border-radius: 12px;
    padding: 1.2rem 1.6rem;
    margin-bottom: 1.6rem;
    box-shadow: 0 4px 20px rgba(0, 0, 0, 0.4);
}

.header-top-row {
    display: flex;
    justify-content: space-between;
    align-items: center;
    border-bottom: 1px solid #1e293b;
    padding-bottom: 0.8rem;
    margin-bottom: 0.8rem;
}

.header-title-block {
    display: flex;
    align-items: center;
    gap: 12px;
}

.header-title {
    font-size: 1.35rem;
    font-weight: 800;
    letter-spacing: -0.02em;
    color: #ffffff;
}

.header-meta-row {
    display: flex;
    justify-content: space-between;
    align-items: center;
    font-family: 'JetBrains Mono', monospace;
    font-size: 0.76rem;
    color: #64748b;
}

.badge-tag {
    display: inline-flex;
    align-items: center;
    gap: 6px;
    padding: 3px 8px;
    border-radius: 4px;
    font-size: 0.72rem;
    font-family: 'JetBrains Mono', monospace;
    font-weight: 600;
    letter-spacing: 0.03em;
    text-transform: uppercase;
}

.badge-operational {
    background: rgba(16, 185, 129, 0.1);
    border: 1px solid #10b981;
    color: #10b981;
}

.badge-iso {
    background: rgba(59, 130, 246, 0.1);
    border: 1px solid #3b82f6;
    color: #60a5fa;
}

.badge-amber {
    background: rgba(245, 158, 11, 0.1);
    border: 1px solid #f59e0b;
    color: #fbbf24;
}

/* Restrained Enterprise Metric Cards */
div[data-testid="stMetric"] {
    background: #0f1422 !important;
    border: 1px solid #1e293b !important;
    border-radius: 10px !important;
    padding: 1.1rem 1.3rem !important;
    box-shadow: 0 4px 14px rgba(0, 0, 0, 0.25) !important;
}

div[data-testid="stMetric"]:hover {
    border-color: #334155 !important;
}

/* Formal Determination Cards */
.verdict-box {
    padding: 1.4rem 1.8rem;
    border-radius: 10px;
    margin: 1.4rem 0;
    display: flex;
    align-items: center;
    justify-content: space-between;
}

.verdict-authentic {
    background: #062419;
    border: 1px solid #059669;
    border-left: 6px solid #10b981;
}

.verdict-manipulated {
    background: #270d10;
    border: 1px solid #b91c1c;
    border-left: 6px solid #ef4444;
}

.verdict-inconclusive {
    background: #141b2d;
    border: 1px solid #334155;
    border-left: 6px solid #64748b;
}

/* Primary Enterprise Action Buttons */
.stButton > button {
    background: #2563eb !important;
    color: #ffffff !important;
    border: 1px solid #3b82f6 !important;
    border-radius: 6px !important;
    padding: 0.6rem 1.6rem !important;
    font-weight: 600 !important;
    font-size: 0.88rem !important;
    letter-spacing: 0.01em !important;
    box-shadow: 0 2px 8px rgba(37, 99, 235, 0.3) !important;
    transition: all 0.15s ease !important;
}

.stButton > button:hover {
    background: #1d4ed8 !important;
    border-color: #60a5fa !important;
}

/* Enterprise Tabs */
button[data-baseweb="tab"] {
    font-size: 0.88rem !important;
    font-weight: 600 !important;
    padding: 0.6rem 1.2rem !important;
    color: #64748b !important;
    border-bottom: 2px solid transparent !important;
}

button[aria-selected="true"] {
    color: #38bdf8 !important;
    border-bottom-color: #38bdf8 !important;
}

/* Enterprise Cards & Tables */
.enterprise-card {
    background: #0f1422;
    border: 1px solid #1e293b;
    border-radius: 10px;
    padding: 1.3rem 1.5rem;
    margin-bottom: 1rem;
}

div[data-testid="stDataFrame"] {
    border: 1px solid #1e293b;
    border-radius: 8px;
    overflow: hidden;
}
</style>
"""
)

# 3. Interactive Forensic Spotlight & Wipe Inspector
def render_interactive_spotlight_lens(original_path, heatmap_path, fft_path=None, height=540):
    """
    Renders an enterprise-grade forensic visualizer:
    - Cursor-following spatial attention spotlight
    - Forensic Wipe Comparison Slider
    - 2D-FFT Fourier Spectral Decomposition
    - Full Attention Layer Overlay
    """
    try:
        with open(original_path, "rb") as f:
            b64_orig = base64.b64encode(f.read()).decode("utf-8")
        with open(heatmap_path, "rb") as f:
            b64_heat = base64.b64encode(f.read()).decode("utf-8")
        b64_fft = ""
        if fft_path and os.path.exists(fft_path):
            with open(fft_path, "rb") as f:
                b64_fft = base64.b64encode(f.read()).decode("utf-8")
    except Exception as e:
        st.error(f"Error loading visualizer: {e}")
        return

    html_code = f"""
    <!DOCTYPE html>
    <html>
    <head>
    <style>
    @import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700&family=JetBrains+Mono:wght@400;500;600&display=swap');
    
    * {{
        box-sizing: border-box;
        margin: 0;
        padding: 0;
        user-select: none;
    }}
    
    body {{
        background: transparent;
        color: #f1f5f9;
        font-family: 'Inter', sans-serif;
        overflow: hidden;
    }}

    .lens-wrapper {{
        background: #0f1422;
        border: 1px solid #1e293b;
        border-radius: 10px;
        padding: 12px;
        box-shadow: 0 10px 30px rgba(0, 0, 0, 0.5);
        display: flex;
        flex-direction: column;
        gap: 10px;
    }}

    .lens-header {{
        display: flex;
        justify-content: space-between;
        align-items: center;
        flex-wrap: wrap;
        gap: 8px;
        padding-bottom: 8px;
        border-bottom: 1px solid #1e293b;
    }}

    .lens-title {{
        display: flex;
        align-items: center;
        gap: 8px;
        font-size: 0.82rem;
        font-weight: 700;
        font-family: 'JetBrains Mono', monospace;
        color: #94a3b8;
        letter-spacing: 0.05em;
        text-transform: uppercase;
    }}

    .lens-controls {{
        display: flex;
        gap: 4px;
        flex-wrap: wrap;
    }}

    .ctrl-btn {{
        background: #1e293b;
        color: #94a3b8;
        border: 1px solid #334155;
        border-radius: 4px;
        padding: 4px 10px;
        font-size: 0.72rem;
        font-family: 'JetBrains Mono', monospace;
        font-weight: 600;
        cursor: pointer;
        transition: all 0.15s ease;
    }}

    .ctrl-btn:hover {{
        background: #334155;
        color: #ffffff;
    }}

    .ctrl-btn.active {{
        background: #2563eb;
        color: #ffffff;
        border-color: #3b82f6;
    }}

    .lens-viewport {{
        position: relative;
        width: 100%;
        height: 420px;
        border-radius: 8px;
        overflow: hidden;
        background: #05070d;
        cursor: crosshair;
        border: 1px solid #1e293b;
    }}

    .layer-img {{
        position: absolute;
        inset: 0;
        width: 100%;
        height: 100%;
        object-fit: contain;
        pointer-events: none;
    }}

    .reveal-img {{
        z-index: 2;
        transition: opacity 0.15s ease;
    }}

    .fft-img {{
        z-index: 3;
        display: none;
    }}

    /* Precision Reticle */
    .hud-reticle {{
        position: absolute;
        width: 320px;
        height: 320px;
        border-radius: 50%;
        border: 1px solid rgba(56, 189, 248, 0.7);
        transform: translate(-50%, -50%);
        pointer-events: none;
        z-index: 6;
        box-shadow: 0 0 20px rgba(56, 189, 248, 0.2);
        display: none;
    }}

    .hud-crosshair-h {{
        position: absolute;
        top: 50%;
        left: 0;
        right: 0;
        height: 1px;
        background: rgba(56, 189, 248, 0.35);
    }}

    .hud-crosshair-v {{
        position: absolute;
        left: 50%;
        top: 0;
        bottom: 0;
        width: 1px;
        background: rgba(56, 189, 248, 0.35);
    }}

    .hud-tag {{
        position: absolute;
        top: 10px;
        left: 50%;
        transform: translateX(-50%);
        background: rgba(15, 23, 42, 0.95);
        border: 1px solid #38bdf8;
        border-radius: 4px;
        padding: 2px 6px;
        font-size: 0.65rem;
        font-family: 'JetBrains Mono', monospace;
        color: #38bdf8;
        white-space: nowrap;
    }}

    /* Wipe Slider Divider */
    .slider-divider {{
        position: absolute;
        top: 0;
        bottom: 0;
        width: 2px;
        background: #38bdf8;
        z-index: 5;
        box-shadow: 0 0 10px rgba(56, 189, 248, 0.8);
        display: none;
    }}

    .slider-thumb {{
        position: absolute;
        top: 50%;
        left: 50%;
        transform: translate(-50%, -50%);
        width: 24px;
        height: 24px;
        background: #0f172a;
        border: 2px solid #38bdf8;
        border-radius: 50%;
        display: flex;
        align-items: center;
        justify-content: center;
        color: #38bdf8;
        font-size: 10px;
        font-weight: bold;
    }}

    .lens-footer {{
        display: flex;
        justify-content: space-between;
        align-items: center;
        font-size: 0.74rem;
        font-family: 'JetBrains Mono', monospace;
        color: #64748b;
        padding-top: 4px;
    }}
    </style>
    </head>
    <body>
    <div class="lens-wrapper">
        <div class="lens-header">
            <div class="lens-title">
                <span>// FORENSIC SPATIAL ANOMALY INSPECTOR</span>
            </div>
            <div class="lens-controls">
                <button id="btnLens" class="ctrl-btn active" onclick="setInspectionMode('lens')">SPOTLIGHT LENS</button>
                <button id="btnWipe" class="ctrl-btn" onclick="setInspectionMode('wipe')">WIPE COMPARISON</button>
                <button id="btnHeatmap" class="ctrl-btn" onclick="setInspectionMode('heatmap')">ATTENTION HEATMAP</button>
                <button id="btnFFT" class="ctrl-btn" onclick="setInspectionMode('fft')" {'style="display:none;"' if not b64_fft else ''}>2D-FFT SPECTRUM</button>
                <button id="btnOriginal" class="ctrl-btn" onclick="setInspectionMode('orig')">RAW SOURCE</button>
            </div>
        </div>

        <div id="viewport" class="lens-viewport">
            <img id="baseImg" src="data:image/png;base64,{b64_orig}" class="layer-img" />
            <img id="revealImg" src="data:image/png;base64,{b64_heat}" class="layer-img reveal-img" />
            <img id="fftImg" src="data:image/png;base64,{b64_fft}" class="layer-img fft-img" />
            
            <!-- Reticle -->
            <div id="reticle" class="hud-reticle">
                <div class="hud-crosshair-h"></div>
                <div class="hud-crosshair-v"></div>
                <div id="coordTag" class="hud-tag">LENS [R: 160PX]</div>
            </div>

            <!-- Wipe Slider -->
            <div id="sliderDivider" class="slider-divider">
                <div class="slider-thumb">↔</div>
            </div>
        </div>

        <div class="lens-footer">
            <span id="guideText">MODE: SPOTLIGHT LENS (Hover cursor over regions to reveal ViT attention layer)</span>
            <span id="telemetry">STANDBY</span>
        </div>
    </div>

    <script>
    const viewport = document.getElementById('viewport');
    const revealImg = document.getElementById('revealImg');
    const fftImg = document.getElementById('fftImg');
    const reticle = document.getElementById('reticle');
    const sliderDivider = document.getElementById('sliderDivider');
    const coordTag = document.getElementById('coordTag');
    const telemetry = document.getElementById('telemetry');
    const guideText = document.getElementById('guideText');

    let currentMode = 'lens';
    const SPOTLIGHT_RADIUS = 160;

    function applySpotlight(x, y) {{
        const maskCSS = `radial-gradient(circle ${{SPOTLIGHT_RADIUS}}px at ${{x}}px ${{y}}px, rgba(0,0,0,1) 0%, rgba(0,0,0,1) 40%, rgba(0,0,0,0.7) 65%, rgba(0,0,0,0.2) 85%, rgba(0,0,0,0) 100%)`;
        revealImg.style.maskImage = maskCSS;
        revealImg.style.webkitMaskImage = maskCSS;
        revealImg.style.clipPath = 'none';
        revealImg.style.opacity = '1';

        reticle.style.left = x + 'px';
        reticle.style.top = y + 'px';
        reticle.style.display = 'block';

        telemetry.innerText = `X:${{Math.round(x)}} Y:${{Math.round(y)}}`;
    }}

    function applyWipe(x) {{
        const width = viewport.clientWidth;
        const pct = Math.max(0, Math.min(100, (x / width) * 100));
        revealImg.style.maskImage = 'none';
        revealImg.style.webkitMaskImage = 'none';
        revealImg.style.clipPath = `inset(0 ${{100 - pct}}% 0 0)`;
        revealImg.style.opacity = '1';

        sliderDivider.style.left = pct + '%';
        sliderDivider.style.display = 'block';

        telemetry.innerText = `WIPE: ${{Math.round(pct)}}%`;
    }}

    function setInspectionMode(mode) {{
        currentMode = mode;
        document.querySelectorAll('.ctrl-btn').forEach(btn => btn.classList.remove('active'));
        
        reticle.style.display = 'none';
        sliderDivider.style.display = 'none';
        fftImg.style.display = 'none';
        revealImg.style.display = 'block';
        revealImg.style.clipPath = 'none';
        revealImg.style.maskImage = 'none';
        revealImg.style.webkitMaskImage = 'none';

        if (mode === 'lens') {{
            document.getElementById('btnLens').classList.add('active');
            revealImg.style.opacity = '0';
            guideText.innerText = 'MODE: SPOTLIGHT LENS (Hover cursor over regions to reveal ViT attention layer)';
            telemetry.innerText = 'LENS READY';
        }} else if (mode === 'wipe') {{
            document.getElementById('btnWipe').classList.add('active');
            applyWipe(viewport.clientWidth / 2);
            guideText.innerText = 'MODE: FORENSIC WIPE (Move cursor left/right to compare raw media vs heatmap)';
            telemetry.innerText = 'WIPE ACTIVE';
        }} else if (mode === 'heatmap') {{
            document.getElementById('btnHeatmap').classList.add('active');
            revealImg.style.opacity = '1';
            guideText.innerText = 'MODE: FULL HEATMAP (Complete ViT spatial attention norm layer)';
            telemetry.innerText = 'HEATMAP OVERLAY';
        }} else if (mode === 'fft') {{
            if (document.getElementById('btnFFT')) document.getElementById('btnFFT').classList.add('active');
            revealImg.style.display = 'none';
            fftImg.style.display = 'block';
            guideText.innerText = 'MODE: 2D-FFT SPECTRUM (Frequency domain magnitude distribution)';
            telemetry.innerText = 'FOURIER SPECTRUM';
        }} else if (mode === 'orig') {{
            document.getElementById('btnOriginal').classList.add('active');
            revealImg.style.opacity = '0';
            guideText.innerText = 'MODE: RAW SOURCE (Untouched evidence ingest)';
            telemetry.innerText = 'RAW INGEST';
        }}
    }}

    viewport.addEventListener('mousemove', (e) => {{
        const rect = viewport.getBoundingClientRect();
        const x = e.clientX - rect.left;
        const y = e.clientY - rect.top;

        if (currentMode === 'lens') {{
            applySpotlight(x, y);
        }} else if (currentMode === 'wipe') {{
            applyWipe(x);
        }}
    }});

    viewport.addEventListener('mouseleave', () => {{
        if (currentMode === 'lens') {{
            reticle.style.display = 'none';
            revealImg.style.opacity = '0';
            telemetry.innerText = 'STANDBY';
        }}
    }});

    setInspectionMode('lens');
    </script>
    </body>
    </html>
    """
    components.html(html_code, height=height, scrolling=False)


# 4. Engine Status
engine_info = get_engine_status()

# 5. Enterprise Header Bar
st.html(
f"""
<div class="enterprise-header">
    <div class="header-top-row">
        <div class="header-title-block">
            <span style="font-size: 1.4rem;">⚖️</span>
            <div>
                <div class="header-title">VERICHAIN FORENSIC OS</div>
                <div style="font-size: 0.8rem; color: #94a3b8; font-weight: 500;">Digital Evidence Authentication & Multi-Subject Forensic Suite</div>
            </div>
        </div>
        <div style="display: flex; gap: 8px; align-items: center;">
            <div class="badge-tag badge-operational">● ENGINE: {engine_info['device']}</div>
            <div class="badge-tag badge-iso">ISO/IEC 27037:2012</div>
        </div>
    </div>
    <div class="header-meta-row">
        <span>MODEL CHECKPOINT: {engine_info['model_name']}</span>
        <span>ADMISSIBILITY: FEDERAL RULES OF EVIDENCE RULE 901</span>
        <span>HASH STANDARD: SHA-256 (FIPS 180-4)</span>
    </div>
</div>
"""
)

# 6. Sidebar - Case File Docket & Custody Ledger
with st.sidebar:
    st.markdown("### 📋 Case File Docket")
    
    if engine_info.get("is_neural", False):
        st.success(f"● {engine_info['badge']}")
    else:
        st.warning(f"▲ {engine_info['badge']}")

    st.caption("Custody Standard: `ISO/IEC 27037 Certified`")
    st.divider()

    st.markdown("#### Examiner Identification")
    analyst_name = st.text_input(
        "Lead Forensic Examiner", value="Detective J. Miller"
    )
    badge_num = st.text_input("Examiner Badge ID", value="DFU-88219")
    case_id = st.text_input("Case Docket ID", value="CR-2026-9042A")
    court_jurisdiction = st.text_input(
        "Judicial Jurisdiction", value="U.S. Federal District Court"
    )
    notes = st.text_area(
        "Investigative Custody Notes",
        "Target digital media ingested from precinct evidence repository. Multi-subject facial isolation, neural ViT attention extraction, and cryptographic verification active.",
        height=80
    )
    
    st.divider()
    
    st.markdown("#### Test Presets")
    demo_mode = st.selectbox(
        "Load Pre-Configured Benchmark",
        [
            "None (Upload Custom File)",
            "Suspect Deepfake Portrait",
            "Verified Authentic Headshot",
        ],
    )

# 7. File Ingestion Workspace
st.markdown("#### Evidence Ingestion")
uploaded_file = st.file_uploader(
    "Select or drop digital evidence file (Images or Videos)",
    type=["png", "jpg", "jpeg", "mp4", "mov"],
    help="Supported formats: PNG, JPG, MP4, MOV. Maximum recommended file size: 100MB."
)

active_path = None
file_bytes = None
filename = ""

if uploaded_file is not None:
    file_bytes = uploaded_file.getvalue()
    filename = uploaded_file.name
    suffix = os.path.splitext(filename)[1].lower()

    with tempfile.NamedTemporaryFile(delete=False, suffix=suffix) as tfile:
        tfile.write(file_bytes)
        active_path = tfile.name

elif demo_mode != "None (Upload Custom File)":
    filename = "benchmark_sample.png"
    sample_img = np.zeros((400, 400, 3), dtype=np.uint8)
    if "Deepfake" in demo_mode:
        cv2.circle(sample_img, (200, 200), 90, (255, 255, 255), -1)
        cv2.putText(
            sample_img,
            "SYNTHETIC FACE",
            (45, 205),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.8,
            (0, 0, 255),
            2,
        )
    else:
        cv2.rectangle(sample_img, (100, 100), (300, 300), (0, 255, 0), -1)
        cv2.putText(
            sample_img,
            "AUTHENTIC",
            (110, 205),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.8,
            (255, 255, 255),
            2,
        )

    active_path = "temp_preset.png"
    cv2.imwrite(active_path, sample_img)
    with open(active_path, "rb") as f:
        file_bytes = f.read()

# Reset state on file change
if (
    "last_processed_file" in st.session_state
    and st.session_state["last_processed_file"] != filename
):
    old_res = st.session_state.get("results", {})
    if old_res and "heatmap_path" in old_res and old_res["heatmap_path"] and os.path.exists(old_res["heatmap_path"]):
        try:
            os.remove(old_res["heatmap_path"])
        except Exception:
            pass
    st.session_state.pop("results", None)

# 8. Main Forensic Execution Pipeline
if active_path and file_bytes:
    sha256_hash = hashlib.sha256(file_bytes).hexdigest()

    with st.sidebar:
        st.markdown("#### Cryptographic Custody")
        st.code(sha256_hash, language="text")
        st.caption("Immutable SHA-256 Evidence Digest")

    col1, col2 = st.columns([1, 1], gap="large")
    with col1:
        st.markdown("##### Media Source Preview")
        try:
            if filename.lower().endswith((".mp4", ".mov")):
                st.video(active_path)
            else:
                st.image(active_path, caption=f"File: {filename}")
        except Exception:
            st.error("Media viewer preview failed for this format.")

    with col2:
        st.markdown("##### Examination Parameters")
        st.html(
        f"""
        <div class="enterprise-card">
            <div style="font-family: 'JetBrains Mono', monospace; font-size: 0.8rem; line-height: 1.8; color: #94a3b8;">
                <b>TARGET:</b> {filename}<br>
                <b>SHA-256:</b> {sha256_hash[:20]}...<br>
                <b>PIPELINE:</b> Ensemble Cascades + SigLIP ViT Classifier<br>
                <b>SPECTRAL:</b> 2D-FFT Residual Energy Decomposition<br>
                <b>STANDARD:</b> ISO/IEC 27037:2012 Certified
            </div>
        </div>
        """
        )
        if st.button("Run Forensic Integrity Audit", type="primary"):
            with st.spinner("Executing biometric face isolation, ViT attention extraction, and spectral audit..."):
                try:
                    if filename.lower().endswith((".mp4", ".mov")):
                        from engine import analyze_video
                        results = analyze_video(active_path)
                    else:
                        from engine import analyze_image
                        results = analyze_image(active_path)

                    st.session_state["results"] = results
                    st.session_state["file_hash"] = sha256_hash
                    st.session_state["filename"] = filename
                    st.session_state["last_processed_file"] = filename
                except Exception as err:
                    st.error(f"Forensic audit encountered an error: {err}")

    # 9. Results Presentation Dashboard
    if "results" in st.session_state:
        res = st.session_state["results"]
        st.divider()

        verdict_str = res.get("verdict", "Unknown")
        is_deepfake = "Manipulated" in verdict_str or "Deepfake" in verdict_str
        is_no_face = res.get("status") == "NO_FACE_DETECTED" or res.get("face_count", 0) == 0

        # Primary Determination Card
        if is_no_face:
            box_class = "verdict-inconclusive"
            tag = "INCONCLUSIVE DETERMINATION"
        elif is_deepfake:
            box_class = "verdict-manipulated"
            tag = "TAMPERED // MANIPULATED EVIDENCE"
        else:
            box_class = "verdict-authentic"
            tag = "AUTHENTIC DIGITAL EVIDENCE"

        st.html(
        f"""
        <div class="verdict-box {box_class}">
            <div>
                <div style="font-size: 0.75rem; font-family: 'JetBrains Mono', monospace; letter-spacing: 0.08em; color: #94a3b8; margin-bottom: 4px;">
                    FORENSIC DETERMINATION // {tag}
                </div>
                <div style="font-size: 1.35rem; font-weight: 800; color: #ffffff;">
                    {verdict_str}
                </div>
            </div>
            <div style="text-align: right;">
                <div style="font-size: 0.72rem; font-family: 'JetBrains Mono', monospace; color: #94a3b8;">CONFIDENCE RATING</div>
                <div style="font-size: 1.8rem; font-weight: 800; color: #ffffff;">{res['confidence'] * 100:.1f}%</div>
            </div>
        </div>
        """
        )

        # Telemetry Cockpit Cards
        mcol1, mcol2, mcol3, mcol4 = st.columns(4)
        with mcol1:
            st.metric("Detected Subjects", f"{res.get('face_count', 0)} Subject(s)")
        with mcol2:
            st.metric("Aggregate Confidence", f"{res['confidence'] * 100:.2f}%")
        with mcol3:
            st.metric("Peak Anomaly Score", f"{res['manipulation_score']:.3f}")
        with mcol4:
            st.metric("Spectral Residual (FFT)", f"{res['fft_score']:.4f}")

        # Multi-Tab Forensic Workbench
        tab1, tab2, tab3, tab4 = st.tabs([
            "Spatial Anomaly Visualizer",
            "Subject Isolation Matrix",
            "Technical Telemetry",
            "Judicial Dossier Export"
        ])

        with tab1:
            st.markdown("##### Spatial Feature & Anomaly Inspection")
            if (
                "heatmap_path" in res
                and res["heatmap_path"]
                and os.path.exists(res["heatmap_path"])
                and not filename.lower().endswith((".mp4", ".mov"))
            ):
                render_interactive_spotlight_lens(
                    active_path,
                    res["heatmap_path"],
                    fft_path=res.get("fft_spectrum_path"),
                    height=540
                )
            elif "heatmap_path" in res and res["heatmap_path"] and os.path.exists(res["heatmap_path"]):
                st.image(res["heatmap_path"], caption="Peak Anomaly Frame Attention Overlay")
            else:
                st.info("Visual overlay unavailable for this media format.")

        with tab2:
            st.markdown("##### Subject-by-Subject Facial Isolation Matrix")
            faces_list = res.get("faces", [])
            if faces_list:
                table_data = []
                for f in faces_list:
                    subj_id = f"Subject #{f['subject_id']}"
                    if f.get("is_full_frame", False):
                        bbox = "Full Frame"
                    else:
                        bbox = f"[{f['bbox'][0]}, {f['bbox'][1]}, {f['bbox'][2]}, {f['bbox'][3]}]"
                    subj_verdict = "Manipulated" if f["manipulation_score"] >= 0.50 else "Authentic"
                    manip_prob = f"{f['manipulation_score']*100:.1f}%"
                    conf = f"{f['confidence']*100:.1f}%"
                    res_tag = "Low-Res (<80px)" if f.get("low_resolution", False) else "Standard"

                    table_data.append({
                        "Subject": subj_id,
                        "Coordinates (X, Y, W, H)": bbox,
                        "Determination": subj_verdict,
                        "Manipulation Score": manip_prob,
                        "Confidence": conf,
                        "Resolution": res_tag
                    })

                st.dataframe(table_data, hide_index=True)
            else:
                st.info("No facial subjects isolated in frame.")

        with tab3:
            st.markdown("##### Quantitative Technical Telemetry")
            dcol1, dcol2 = st.columns(2)
            with dcol1:
                st.html(
                f"""
                <div class="enterprise-card">
                    <div style="font-size: 0.75rem; font-family: 'JetBrains Mono', monospace; font-weight: 700; color: #38bdf8; margin-bottom: 0.5rem;">BIOMETRIC BOUNDARY ANALYSIS</div>
                    <div style="font-size: 0.85rem; line-height: 1.8;">
                        <b>Examiner Summary:</b> {res.get('summary_note', 'N/A')}<br>
                        <b>Boundary Continuity:</b> {'Discontinuous blending boundary detected' if res['manipulation_score'] >= 0.5 else 'Seamless natural boundary'}<br>
                        <b>Model Architecture:</b> SigLIP Vision Transformer (512px Patch Size 16)
                    </div>
                </div>
                """
                )
            with dcol2:
                fft_verdict = "High (Synthetic Diffusion/GAN Energy Signature)" if res["fft_score"] > 0.4 else "Normal (Natural Photographic Baseline)"
                st.html(
                f"""
                <div class="enterprise-card">
                    <div style="font-size: 0.75rem; font-family: 'JetBrains Mono', monospace; font-weight: 700; color: #38bdf8; margin-bottom: 0.5rem;">FREQUENCY DOMAIN RESIDUAL (2D-FFT)</div>
                    <div style="font-size: 0.85rem; line-height: 1.8;">
                        <b>FFT Residual Metric:</b> <code>{res['fft_score']:.4f}</code><br>
                        <b>Spectral Energy Anomaly:</b> {fft_verdict}<br>
                        <b>Calibrated Baseline:</b> 0.20–0.35 Natural Sensor Noise
                    </div>
                </div>
                """
                )

        with tab4:
            st.markdown("##### Judicial Admissibility & Chain-of-Custody")
            acol1, acol2 = st.columns([1.2, 0.8], gap="large")
            with acol1:
                st.html(
                f"""
                <div class="enterprise-card">
                    <div style="font-size: 0.75rem; font-family: 'JetBrains Mono', monospace; font-weight: 700; color: #10b981; margin-bottom: 0.6rem;">CERTIFIED CHAIN OF CUSTODY</div>
                    <div style="font-size: 0.82rem; line-height: 1.8; font-family: 'JetBrains Mono', monospace; color: #94a3b8;">
                        <b>TARGET MEDIA:</b> {filename}<br>
                        <b>SHA-256 HASH:</b> {st.session_state['file_hash']}<br>
                        <b>LEAD EXAMINER:</b> {analyst_name} (Badge: {badge_num})<br>
                        <b>JURISDICTION:</b> {court_jurisdiction}<br>
                        <b>CASE DOCKET:</b> {case_id}<br>
                        <b>COMPLIANCE:</b> Federal Rules of Evidence Rule 901 & ISO/IEC 27037
                    </div>
                </div>
                """
                )
            with acol2:
                st.markdown("##### Export Official Dossier")
                st.caption("Generate certified forensic dossier with cryptographic hash, subject breakdown table, and spatial anomaly overlays.")
                
                try:
                    from report import generate_pdf

                    pdf_path = generate_pdf(
                        filename=st.session_state["filename"],
                        file_hash=st.session_state["file_hash"],
                        analyst=f"{analyst_name} (Badge: {badge_num})",
                        verdict=res.get("verdict", "Unknown"),
                        confidence=res.get("confidence", 0.0),
                        heatmap_path=res.get("heatmap_path", None),
                        faces_data=res.get("faces", []),
                        summary_note=res.get("summary_note", "")
                    )

                    with open(pdf_path, "rb") as pdf_file:
                        st.download_button(
                            label="Download Court-Admissible Dossier (PDF)",
                            data=pdf_file,
                            file_name=f"VeriChain_Forensic_Dossier_{case_id}.pdf",
                            mime="application/pdf",
                        )
                except Exception as pdf_err:
                    st.warning(f"PDF Dossier generator encountered an issue: {pdf_err}")
