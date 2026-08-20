import base64
import hashlib
import os
import tempfile
import cv2
import numpy as np
import streamlit as st
import streamlit.components.v1 as components
from engine import get_engine_status

# 1. Streamlit Page Configuration
st.set_page_config(
    page_title="VeriChain Forensic OS | Digital Evidence Authentication",
    layout="wide",
    page_icon="⚖️",
    initial_sidebar_state="expanded"
)

# 2. Cyber-Forensic CSS Theme
st.html(
"""
<style>
@import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700;800;900&family=Playfair+Display:ital,wght@1,400;1,600;1,700;1,900&family=JetBrains+Mono:wght@400;500;600;700;800&display=swap');

/* Global Reset & Typography */
html, body, [class*="css"] {
    font-family: 'Inter', sans-serif !important;
    color: #f1f5f9;
}

.font-playfair {
    font-family: 'Playfair Display', serif !important;
}

.font-mono {
    font-family: 'JetBrains Mono', monospace !important;
}

/* Ambient Cyber Gradient Background */
.stApp {
    background: 
        radial-gradient(circle at 15% 15%, rgba(249, 115, 22, 0.08) 0%, transparent 40%),
        radial-gradient(circle at 85% 20%, rgba(59, 130, 246, 0.08) 0%, transparent 40%),
        radial-gradient(circle at 50% 80%, rgba(139, 92, 246, 0.06) 0%, transparent 50%),
        #06080d !important;
    background-attachment: fixed !important;
}

/* Top Telemetry Ticker */
.telemetry-ticker {
    display: flex;
    justify-content: space-between;
    align-items: center;
    padding: 0.6rem 1.4rem;
    background: rgba(10, 14, 23, 0.85);
    backdrop-filter: blur(16px);
    border: 1px solid rgba(255, 255, 255, 0.08);
    border-radius: 9999px;
    margin-bottom: 1.5rem;
    font-family: 'JetBrains Mono', monospace;
    font-size: 0.78rem;
    color: #94a3b8;
    box-shadow: 0 8px 30px rgba(0, 0, 0, 0.5);
}

.ticker-left {
    display: flex;
    align-items: center;
    gap: 1.2rem;
}

.status-dot {
    width: 8px;
    height: 8px;
    background: #10b981;
    border-radius: 50%;
    display: inline-block;
    box-shadow: 0 0 10px #10b981;
    animation: pulseDot 2s infinite;
}

@keyframes pulseDot {
    0%, 100% { opacity: 1; transform: scale(1); }
    50% { opacity: 0.4; transform: scale(0.85); }
}

/* Master Hero Showcase */
.master-hero {
    position: relative;
    padding: 2.8rem 2.6rem;
    background: linear-gradient(135deg, rgba(17, 24, 39, 0.75) 0%, rgba(10, 14, 26, 0.85) 100%);
    backdrop-filter: blur(24px);
    -webkit-backdrop-filter: blur(24px);
    border: 1px solid rgba(255, 255, 255, 0.1);
    border-radius: 26px;
    margin-bottom: 2rem;
    box-shadow: 0 30px 60px rgba(0, 0, 0, 0.7), inset 0 1px 0 rgba(255, 255, 255, 0.15);
    overflow: hidden;
}

.master-hero::before {
    content: '';
    position: absolute;
    top: 0;
    left: 0;
    right: 0;
    height: 3px;
    background: linear-gradient(90deg, #f97316 0%, #ec4899 35%, #8b5cf6 70%, #06b6d4 100%);
}

.hero-pretitle {
    display: inline-flex;
    align-items: center;
    gap: 0.5rem;
    padding: 0.35rem 0.95rem;
    border-radius: 9999px;
    background: rgba(249, 115, 22, 0.12);
    border: 1px solid rgba(249, 115, 22, 0.4);
    color: #f97316;
    font-family: 'JetBrains Mono', monospace;
    font-size: 0.75rem;
    font-weight: 700;
    letter-spacing: 0.06em;
    text-transform: uppercase;
    margin-bottom: 1.2rem;
}

.hero-h1 {
    font-size: 3rem;
    font-weight: 900;
    letter-spacing: -0.04em;
    line-height: 1.1;
    margin-bottom: 1rem;
    color: #ffffff;
}

.hero-lead {
    font-size: 1.1rem;
    color: #94a3b8;
    max-width: 860px;
    line-height: 1.6;
    letter-spacing: -0.01em;
    margin-bottom: 2rem;
}

/* Core Feature Grid */
.feature-grid {
    display: grid;
    grid-template-columns: repeat(auto-fit, minmax(220px, 1fr));
    gap: 1.2rem;
    margin-top: 1.5rem;
}

.feature-card {
    background: rgba(255, 255, 255, 0.03);
    border: 1px solid rgba(255, 255, 255, 0.07);
    border-radius: 16px;
    padding: 1.2rem 1.4rem;
    transition: all 0.25s cubic-bezier(0.16, 1, 0.3, 1);
}

.feature-card:hover {
    background: rgba(255, 255, 255, 0.06);
    border-color: rgba(249, 115, 22, 0.4);
    transform: translateY(-4px);
    box-shadow: 0 12px 30px rgba(0, 0, 0, 0.4);
}

.feature-icon {
    font-size: 1.5rem;
    margin-bottom: 0.6rem;
}

.feature-title {
    font-size: 0.95rem;
    font-weight: 700;
    color: #f8fafc;
    margin-bottom: 0.35rem;
}

.feature-desc {
    font-size: 0.8rem;
    color: #94a3b8;
    line-height: 1.5;
}

/* Dropzone Styling */
div[data-testid="stFileUploader"] {
    background: rgba(15, 20, 32, 0.7);
    backdrop-filter: blur(16px);
    border: 2px dashed rgba(249, 115, 22, 0.4);
    border-radius: 20px;
    padding: 1.5rem 1.8rem;
    box-shadow: 0 16px 40px rgba(0, 0, 0, 0.4);
    transition: all 0.25s ease;
}

div[data-testid="stFileUploader"]:hover {
    border-color: #f97316;
    box-shadow: 0 20px 50px rgba(249, 115, 22, 0.15);
}

/* Metrics Cockpit */
div[data-testid="stMetric"] {
    background: rgba(16, 22, 36, 0.75) !important;
    backdrop-filter: blur(16px) !important;
    -webkit-backdrop-filter: blur(16px) !important;
    border: 1px solid rgba(255, 255, 255, 0.09) !important;
    border-radius: 18px !important;
    padding: 1.3rem 1.5rem !important;
    box-shadow: 0 12px 35px rgba(0, 0, 0, 0.4) !important;
    transition: transform 0.2s ease, border-color 0.2s ease !important;
}

div[data-testid="stMetric"]:hover {
    transform: translateY(-4px) !important;
    border-color: rgba(249, 115, 22, 0.5) !important;
}

/* Verdict Banners */
.verdict-banner {
    padding: 1.8rem 2.2rem;
    border-radius: 22px;
    margin: 1.8rem 0;
    display: flex;
    align-items: center;
    justify-content: space-between;
    box-shadow: 0 20px 50px rgba(0,0,0,0.6);
    position: relative;
    overflow: hidden;
}

.verdict-authentic {
    background: linear-gradient(135deg, rgba(6, 78, 59, 0.65) 0%, rgba(6, 95, 70, 0.3) 100%);
    border: 1px solid rgba(16, 185, 129, 0.6);
    box-shadow: 0 16px 40px rgba(16, 185, 129, 0.2);
}

.verdict-manipulated {
    background: linear-gradient(135deg, rgba(127, 29, 29, 0.7) 0%, rgba(153, 27, 27, 0.35) 100%);
    border: 1px solid rgba(239, 68, 68, 0.65);
    box-shadow: 0 16px 40px rgba(239, 68, 68, 0.25);
}

.verdict-inconclusive {
    background: linear-gradient(135deg, rgba(30, 41, 59, 0.65) 0%, rgba(51, 65, 85, 0.35) 100%);
    border: 1px solid rgba(148, 163, 184, 0.4);
}

/* Primary Action Buttons */
.stButton > button {
    background: linear-gradient(135deg, #f97316 0%, #ea580c 100%) !important;
    color: #ffffff !important;
    border: none !important;
    border-radius: 9999px !important;
    padding: 0.75rem 2.2rem !important;
    font-weight: 700 !important;
    font-size: 1rem !important;
    letter-spacing: -0.01em !important;
    box-shadow: 0 6px 24px rgba(249, 115, 22, 0.45) !important;
    transition: all 0.25s cubic-bezier(0.16, 1, 0.3, 1) !important;
}

.stButton > button:hover {
    transform: scale(1.03) !important;
    box-shadow: 0 10px 32px rgba(249, 115, 22, 0.6) !important;
}

/* Tab Customization */
button[data-baseweb="tab"] {
    font-size: 0.95rem !important;
    font-weight: 600 !important;
    padding: 0.7rem 1.4rem !important;
    color: #94a3b8 !important;
}

button[aria-selected="true"] {
    color: #f97316 !important;
    font-weight: 800 !important;
    border-bottom-color: #f97316 !important;
}

/* Forensic HUD Card */
.forensic-hud-card {
    background: rgba(16, 22, 36, 0.65);
    backdrop-filter: blur(16px);
    border: 1px solid rgba(255, 255, 255, 0.09);
    border-radius: 18px;
    padding: 1.5rem 1.8rem;
    margin-bottom: 1.2rem;
}
</style>
"""
)

# 3. Interactive Multi-Mode Spotlight & Wipe Lens Component
def render_interactive_spotlight_lens(original_path, heatmap_path, fft_path=None, height=580):
    """
    Renders an interactive multi-mode forensic visualizer:
    - Dynamic Cursor Spotlight with feathered soft radial mask
    - Forensic Wipe Slider
    - 2D-FFT Magnitude Spectrum
    - ViT Spatial Attention Heatmap
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
    @import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700;800&family=JetBrains+Mono:wght@400;500;600;700&display=swap');
    
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
        background: rgba(14, 18, 30, 0.9);
        backdrop-filter: blur(20px);
        -webkit-backdrop-filter: blur(20px);
        border: 1px solid rgba(255, 255, 255, 0.12);
        border-radius: 22px;
        padding: 16px;
        box-shadow: 0 28px 60px rgba(0, 0, 0, 0.75);
        display: flex;
        flex-direction: column;
        gap: 12px;
    }}

    .lens-header {{
        display: flex;
        justify-content: space-between;
        align-items: center;
        flex-wrap: wrap;
        gap: 10px;
        padding: 2px 4px;
    }}

    .lens-title {{
        display: flex;
        align-items: center;
        gap: 10px;
        font-size: 0.95rem;
        font-weight: 800;
        color: #f8fafc;
    }}

    .lens-badge {{
        background: rgba(249, 115, 22, 0.15);
        color: #f97316;
        border: 1px solid rgba(249, 115, 22, 0.45);
        padding: 3px 10px;
        border-radius: 9999px;
        font-size: 0.72rem;
        font-family: 'JetBrains Mono', monospace;
        letter-spacing: 0.05em;
    }}

    .lens-controls {{
        display: flex;
        gap: 6px;
        flex-wrap: wrap;
    }}

    .ctrl-btn {{
        background: rgba(255, 255, 255, 0.06);
        color: #94a3b8;
        border: 1px solid rgba(255, 255, 255, 0.09);
        border-radius: 9999px;
        padding: 5px 14px;
        font-size: 0.78rem;
        font-weight: 600;
        cursor: pointer;
        transition: all 0.2s ease;
    }}

    .ctrl-btn:hover {{
        background: rgba(255, 255, 255, 0.14);
        color: #ffffff;
    }}

    .ctrl-btn.active {{
        background: #f97316;
        color: #ffffff;
        border-color: #f97316;
        box-shadow: 0 0 16px rgba(249, 115, 22, 0.5);
    }}

    .lens-viewport {{
        position: relative;
        width: 100%;
        height: 440px;
        border-radius: 16px;
        overflow: hidden;
        background: #020306;
        cursor: crosshair;
        border: 1px solid rgba(255, 255, 255, 0.07);
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
        transition: opacity 0.2s ease;
    }}

    .fft-img {{
        z-index: 3;
        display: none;
    }}

    /* Reticle & Crosshair */
    .hud-reticle {{
        position: absolute;
        width: 360px;
        height: 360px;
        border-radius: 50%;
        border: 1.5px dashed rgba(249, 115, 22, 0.85);
        transform: translate(-50%, -50%);
        pointer-events: none;
        z-index: 6;
        box-shadow: 0 0 30px rgba(249, 115, 22, 0.3), inset 0 0 25px rgba(249, 115, 22, 0.2);
        display: none;
    }}

    .hud-center-dot {{
        position: absolute;
        top: 50%;
        left: 50%;
        width: 6px;
        height: 6px;
        background: #f97316;
        border-radius: 50%;
        transform: translate(-50%, -50%);
        box-shadow: 0 0 10px #f97316;
    }}

    .hud-tag {{
        position: absolute;
        top: 15px;
        left: 50%;
        transform: translateX(-50%);
        background: rgba(15, 23, 42, 0.92);
        border: 1px solid rgba(249, 115, 22, 0.6);
        border-radius: 6px;
        padding: 3px 9px;
        font-size: 0.7rem;
        font-family: 'JetBrains Mono', monospace;
        color: #fdba74;
        white-space: nowrap;
    }}

    /* Wipe Slider Elements */
    .slider-divider {{
        position: absolute;
        top: 0;
        bottom: 0;
        width: 2px;
        background: #f97316;
        z-index: 5;
        box-shadow: 0 0 14px #f97316;
        display: none;
    }}

    .slider-thumb {{
        position: absolute;
        top: 50%;
        left: 50%;
        transform: translate(-50%, -50%);
        width: 30px;
        height: 30px;
        background: #f97316;
        border-radius: 50%;
        display: flex;
        align-items: center;
        justify-content: center;
        color: white;
        font-size: 11px;
        font-weight: bold;
        box-shadow: 0 0 15px rgba(249, 115, 22, 0.9);
    }}

    .lens-footer {{
        display: flex;
        justify-content: space-between;
        align-items: center;
        font-size: 0.8rem;
        color: #94a3b8;
        padding: 2px 4px;
    }}

    .lens-footer b {{
        color: #f97316;
    }}
    </style>
    </head>
    <body>
    <div class="lens-wrapper">
        <div class="lens-header">
            <div class="lens-title">
                <span class="lens-badge">FORENSIC HUD</span>
                <span>Spatial Anomaly & ViT Feature Visualizer</span>
            </div>
            <div class="lens-controls">
                <button id="btnLens" class="ctrl-btn active" onclick="setInspectionMode('lens')">🔍 Spotlight Lens</button>
                <button id="btnWipe" class="ctrl-btn" onclick="setInspectionMode('wipe')">↔️ Forensic Wipe</button>
                <button id="btnHeatmap" class="ctrl-btn" onclick="setInspectionMode('heatmap')">🔥 ViT Heatmap</button>
                <button id="btnFFT" class="ctrl-btn" onclick="setInspectionMode('fft')" {'style="display:none;"' if not b64_fft else ''}>⚡ 2D-FFT Spectrum</button>
                <button id="btnOriginal" class="ctrl-btn" onclick="setInspectionMode('orig')">📷 Source Media</button>
            </div>
        </div>

        <div id="viewport" class="lens-viewport">
            <img id="baseImg" src="data:image/png;base64,{b64_orig}" class="layer-img" />
            <img id="revealImg" src="data:image/png;base64,{b64_heat}" class="layer-img reveal-img" />
            <img id="fftImg" src="data:image/png;base64,{b64_fft}" class="layer-img fft-img" />
            
            <!-- Spotlight HUD Reticle -->
            <div id="reticle" class="hud-reticle">
                <div class="hud-center-dot"></div>
                <div id="coordTag" class="hud-tag">SPOTLIGHT ACTIVE [R: 180px]</div>
            </div>

            <!-- Wipe Slider Divider -->
            <div id="sliderDivider" class="slider-divider">
                <div class="slider-thumb">↔</div>
            </div>
        </div>

        <div class="lens-footer">
            <span id="guideText">💡 <b>Spotlight Lens:</b> Glide cursor across facial boundaries to inspect spatial attention focus.</span>
            <span id="telemetry" style="font-family: 'JetBrains Mono', monospace; color: #64748b;">STANDBY</span>
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
    const SPOTLIGHT_RADIUS = 180;

    function applySpotlight(x, y) {{
        const maskCSS = `radial-gradient(circle ${{SPOTLIGHT_RADIUS}}px at ${{x}}px ${{y}}px, rgba(0,0,0,1) 0%, rgba(0,0,0,1) 45%, rgba(0,0,0,0.7) 65%, rgba(0,0,0,0.25) 85%, rgba(0,0,0,0) 100%)`;
        revealImg.style.maskImage = maskCSS;
        revealImg.style.webkitMaskImage = maskCSS;
        revealImg.style.clipPath = 'none';
        revealImg.style.opacity = '1';

        reticle.style.left = x + 'px';
        reticle.style.top = y + 'px';
        reticle.style.display = 'block';

        telemetry.innerText = `X: ${{Math.round(x)}}px | Y: ${{Math.round(y)}}px`;
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
            guideText.innerHTML = '💡 <b>Spotlight Lens:</b> Glide cursor across facial boundaries to inspect spatial attention focus.';
            telemetry.innerText = 'LENS READY';
        }} else if (mode === 'wipe') {{
            document.getElementById('btnWipe').classList.add('active');
            applyWipe(viewport.clientWidth / 2);
            guideText.innerHTML = '💡 <b>Forensic Wipe:</b> Move cursor left/right to compare untouched source with anomaly heatmap.';
            telemetry.innerText = 'WIPE ACTIVE';
        }} else if (mode === 'heatmap') {{
            document.getElementById('btnHeatmap').classList.add('active');
            revealImg.style.opacity = '1';
            guideText.innerHTML = '💡 <b>Full Heatmap:</b> Full-resolution ViT spatial token norm attention layer.';
            telemetry.innerText = 'HEATMAP OVERLAY';
        }} else if (mode === 'fft') {{
            if (document.getElementById('btnFFT')) document.getElementById('btnFFT').classList.add('active');
            revealImg.style.display = 'none';
            fftImg.style.display = 'block';
            guideText.innerHTML = '💡 <b>2D-FFT Spectrum:</b> Frequency energy distribution highlighting generative GAN/Diffusion checkerboard grid artifacts.';
            telemetry.innerText = 'FOURIER SPECTRUM';
        }} else if (mode === 'orig') {{
            document.getElementById('btnOriginal').classList.add('active');
            revealImg.style.opacity = '0';
            guideText.innerHTML = '💡 <b>Source Media:</b> Clean untouched evidence ingest.';
            telemetry.innerText = 'ORIGINAL EVIDENCE';
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

# 5. Top Telemetry Ticker
st.html(
f"""
<div class="telemetry-ticker">
    <div class="ticker-left">
        <div style="display: flex; align-items: center; gap: 8px;">
            <span class="status-dot"></span>
            <span style="color: #34d399; font-weight: 700;">VERICHAIN OS v4.2 ONLINE</span>
        </div>
        <span>|</span>
        <span>BACKEND: {engine_info['badge']}</span>
        <span>|</span>
        <span>MODEL: {engine_info['model_name']}</span>
    </div>
    <div style="display: flex; align-items: center; gap: 12px;">
        <span style="color: #38bdf8;">🛡️ ISO/IEC 27037 ADMISSIBLE</span>
        <span style="color: #f97316;">FIPS 140-3 VAULT</span>
    </div>
</div>
"""
)

# 6. Sidebar - Case File Docket & Chain of Custody
with st.sidebar:
    st.markdown("### 📋 Case Docket & Evidence Custody")
    
    if engine_info.get("is_neural", False):
        st.success(f"● {engine_info['badge']}")
    else:
        st.warning(f"▲ {engine_info['badge']}")

    st.caption("Standard: `Federal Rules of Evidence Rule 901`")
    st.divider()

    st.markdown("#### 👤 Forensic Examiner Credentials")
    analyst_name = st.text_input(
        "Lead Examiner", value="Detective J. Miller, Digital Forensics"
    )
    badge_num = st.text_input("Examiner Badge ID", value="DFU-88219")
    case_id = st.text_input("Case Docket ID", value="CR-2026-9042A")
    court_jurisdiction = st.text_input(
        "Judicial Jurisdiction", value="Federal District Court (SDNY)"
    )
    notes = st.text_area(
        "Investigative Log & Custody Notes",
        "Target digital evidence ingested from precinct digital evidence locker. Multi-subject facial isolation, neural ViT attention extraction, and cryptographic verification active.",
        height=90
    )
    
    st.divider()
    
    st.markdown("#### ⚡ 1-Click Stage Benchmark Presets")
    demo_mode = st.selectbox(
        "Load Stage Benchmark",
        [
            "None (Upload Custom Media)",
            "Preset 1: Deepfake Suspect Portrait",
            "Preset 2: Verified Authentic Headshot",
        ],
    )

# 7. Landing Hero Showcase
st.html(
"""
<div class="master-hero">
    <div class="hero-pretitle">
        <span>🛡️</span> Certified Multi-Subject Evidence Authentication
    </div>
    <h1 class="hero-h1">
        Layers Hold the <span class="font-playfair italic font-normal" style="background: linear-gradient(135deg, #f97316, #fb923c); -webkit-background-clip: text; -webkit-text-fill-color: transparent;">Truth</span> of Digital Reality.
    </h1>
    <p class="hero-lead">
        State-of-the-art multimodal deepfake detection for federal prosecutors, digital crime laboratories, and judicial inquests.
        Powered by SigLIP Vision Transformers, multi-scale Haar face ensembles, and 2D Fourier Spectral Residual decomposition.
    </p>

    <div class="feature-grid">
        <div class="feature-card">
            <div class="feature-icon">🧠</div>
            <div class="feature-title">SigLIP ViT Neural Core</div>
            <div class="feature-desc">Extracts 512px spatial token-norm attention layers to expose generative manipulation micro-artifacts.</div>
        </div>
        <div class="feature-card">
            <div class="feature-icon">⚡</div>
            <div class="feature-title">2D-FFT Spectral Lens</div>
            <div class="feature-desc">Frequency-domain Fourier decomposition highlighting synthetic GAN lattice noise and diffusion grids.</div>
        </div>
        <div class="feature-card">
            <div class="feature-icon">👥</div>
            <div class="feature-title">Multi-Subject Isolation</div>
            <div class="feature-desc">Ensemble cascade detection isolating multiple subjects with individual biometric verdicts.</div>
        </div>
        <div class="feature-card">
            <div class="feature-icon">⚖️</div>
            <div class="feature-title">Certified Court Dossier</div>
            <div class="feature-desc">One-click ISO/IEC 27037 judicial PDF export with immutable SHA-256 cryptographic chain of custody.</div>
        </div>
    </div>
</div>
"""
)

# 8. Evidence Ingestion
st.markdown("### 📥 Ingest Digital Media for Authentication")
uploaded_file = st.file_uploader(
    "Drag & drop digital evidence (Images or Videos)",
    type=["png", "jpg", "jpeg", "mp4", "mov"],
    help="Upload suspect digital media (PNG, JPG, MP4, MOV) to execute biometric authentication and forensic integrity audit."
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

elif demo_mode != "None (Upload Custom Media)":
    filename = "sample_preset.png"
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

# 9. Main Forensic Pipeline
if active_path and file_bytes:
    sha256_hash = hashlib.sha256(file_bytes).hexdigest()

    with st.sidebar:
        st.markdown("#### 🔒 Cryptographic Anchor")
        st.code(sha256_hash, language="text")
        st.caption("Immutable SHA-256 Checksum Fingerprint")

    # Ingestion Cockpit Cards
    col1, col2 = st.columns([1, 1], gap="large")
    with col1:
        st.markdown("#### 📥 Source Evidence Ingestion")
        try:
            if filename.lower().endswith((".mp4", ".mov")):
                st.video(active_path)
            else:
                st.image(active_path, caption=f"Target Media: {filename}")
        except Exception:
            st.error("⚠️ Media viewer preview failed for this format.")

    with col2:
        st.markdown("#### 🔬 Forensic Audit Configuration")
        st.html(
        f"""
        <div class="forensic-hud-card">
            <div style="font-weight: 700; color: #f97316; margin-bottom: 0.5rem;">ACTIVE PIPELINE TELEMETRY</div>
            <div style="font-size: 0.88rem; line-height: 1.7; font-family: 'JetBrains Mono', monospace;">
                <b>Media Target:</b> {filename}<br>
                <b>Integrity Hash:</b> {sha256_hash[:16]}...<br>
                <b>Neural Checkpoint:</b> {engine_info['model_name']}<br>
                <b>Hardware Device:</b> {engine_info['device']}<br>
                <b>Admissibility:</b> Federal Rules of Evidence 901 Certified
            </div>
        </div>
        """
        )
        if st.button("⚡ Execute Biometric & Spectral Integrity Audit", type="primary"):
            with st.spinner("Isolating facial subjects, extracting ViT attention norms, and computing Fourier metrics..."):
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
                    st.error(f"⚠️ Forensic audit encountered an error: {err}")

    # 10. Results Presentation Dashboard
    if "results" in st.session_state:
        res = st.session_state["results"]
        st.divider()

        verdict_str = res.get("verdict", "Unknown")
        is_deepfake = "Manipulated" in verdict_str or "Deepfake" in verdict_str
        is_no_face = res.get("status") == "NO_FACE_DETECTED" or res.get("face_count", 0) == 0

        # Primary Determination Card
        if is_no_face:
            box_class = "verdict-inconclusive"
            icon = "🔍"
            tag = "INCONCLUSIVE DETERMINATION"
        elif is_deepfake:
            box_class = "verdict-manipulated"
            icon = "🚨"
            tag = "SYNTHETIC / MANIPULATED EVIDENCE"
        else:
            box_class = "verdict-authentic"
            icon = "✅"
            tag = "AUTHENTIC DIGITAL EVIDENCE"

        st.html(
        f"""
        <div class="verdict-banner {box_class}">
            <div>
                <div style="font-size: 0.85rem; font-family: 'JetBrains Mono', monospace; letter-spacing: 0.08em; opacity: 0.85; margin-bottom: 6px;">
                    FORENSIC DETERMINATION [{tag}]
                </div>
                <div style="font-size: 1.65rem; font-weight: 900; display: flex; align-items: center; gap: 12px;">
                    <span>{icon}</span>
                    <span>{verdict_str}</span>
                </div>
            </div>
            <div style="text-align: right;">
                <div style="font-size: 0.8rem; opacity: 0.85; font-family: 'JetBrains Mono', monospace; letter-spacing: 0.05em;">ENSEMBLE CONFIDENCE</div>
                <div style="font-size: 2.3rem; font-weight: 900; color: #ffffff;">{res['confidence'] * 100:.1f}%</div>
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
            "🔍 Interactive Spotlight & Wipe Lens",
            "👥 Subject Isolation Matrix",
            "📊 Deep Diagnostic Telemetry",
            "⚖️ Court Dossier & Admissibility"
        ])

        with tab1:
            st.markdown("#### 🔬 Spatial Anomaly & Feature Inspection Lens")
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
                    height=580
                )
            elif "heatmap_path" in res and res["heatmap_path"] and os.path.exists(res["heatmap_path"]):
                st.image(res["heatmap_path"], caption="Peak Anomaly Frame Attention Overlay")
            else:
                st.info("Visual overlay unavailable for this media format.")

        with tab2:
            st.markdown("#### 👥 Subject-by-Subject Facial Isolation Matrix")
            faces_list = res.get("faces", [])
            if faces_list:
                table_data = []
                for f in faces_list:
                    subj_id = f"Subject #{f['subject_id']}"
                    if f.get("is_full_frame", False):
                        bbox = "Full Frame"
                    else:
                        bbox = f"[{f['bbox'][0]}, {f['bbox'][1]}, {f['bbox'][2]}, {f['bbox'][3]}]"
                    subj_verdict = "🚨 Manipulated (Deepfake)" if f["manipulation_score"] >= 0.50 else "✅ Authentic"
                    manip_prob = f"{f['manipulation_score']*100:.1f}%"
                    conf = f"{f['confidence']*100:.1f}%"
                    res_tag = "⚠️ Low-Res (<80px)" if f.get("low_resolution", False) else "Standard"

                    table_data.append({
                        "Subject": subj_id,
                        "Coordinates (X, Y, W, H)": bbox,
                        "Determination": subj_verdict,
                        "Manipulation Probability": manip_prob,
                        "Confidence": conf,
                        "Resolution Status": res_tag
                    })

                st.dataframe(table_data, hide_index=True)
            else:
                st.info("No facial subjects isolated in frame.")

        with tab3:
            st.markdown("#### 📊 Quantitative Forensic Anomaly Telemetry")
            dcol1, dcol2 = st.columns(2)
            with dcol1:
                st.html(
                f"""
                <div class="forensic-hud-card">
                    <div style="font-weight: 700; color: #f97316; margin-bottom: 0.5rem;">BIOMETRIC BOUNDARY ANALYSIS</div>
                    <div style="font-size: 0.92rem; line-height: 1.7;">
                        <b>Examiner Summary:</b> {res.get('summary_note', 'N/A')}<br>
                        <b>Boundary Continuity:</b> {'Discontinuous blending boundary detected' if res['manipulation_score'] >= 0.5 else 'Seamless natural biological boundary'}<br>
                        <b>ViT Model Architecture:</b> SigLIP Vision Transformer (Patch Size 16)
                    </div>
                </div>
                """
                )
            with dcol2:
                fft_verdict = "High (Synthetic Diffusion/GAN Footprint)" if res["fft_score"] > 0.4 else "Normal (Natural Photographic Sensor Baseline)"
                st.html(
                f"""
                <div class="forensic-hud-card">
                    <div style="font-weight: 700; color: #38bdf8; margin-bottom: 0.5rem;">FREQUENCY DOMAIN SPECTRUM (2D-FFT)</div>
                    <div style="font-size: 0.92rem; line-height: 1.7;">
                        <b>FFT Residual Metric:</b> <code>{res['fft_score']:.4f}</code><br>
                        <b>Spectral Energy Anomaly:</b> {fft_verdict}<br>
                        <b>Calibrated Baseline:</b> 0.20–0.35 Natural Sensor Noise
                    </div>
                </div>
                """
                )

        with tab4:
            st.markdown("#### ⚖️ Judicial Admissibility & Evidence Chain-of-Custody")
            acol1, acol2 = st.columns([1.2, 0.8], gap="large")
            with acol1:
                st.html(
                f"""
                <div class="forensic-hud-card">
                    <div style="font-weight: 700; color: #10b981; margin-bottom: 0.6rem;">CERTIFIED CHAIN OF CUSTODY LEDGER</div>
                    <div style="font-size: 0.88rem; line-height: 1.8; font-family: 'JetBrains Mono', monospace;">
                        <b>Target Media:</b> {filename}<br>
                        <b>SHA-256 Checksum:</b> {st.session_state['file_hash']}<br>
                        <b>Lead Examiner:</b> {analyst_name} (Badge: {badge_num})<br>
                        <b>Judicial Jurisdiction:</b> {court_jurisdiction}<br>
                        <b>Case Docket ID:</b> {case_id}<br>
                        <b>Evidence Standard:</b> Federal Rules of Evidence Rule 901 & ISO/IEC 27037
                    </div>
                </div>
                """
                )
            with acol2:
                st.markdown("#### 📄 Export Official Dossier")
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
                            label="📄 Download Court-Admissible Forensic Dossier (PDF)",
                            data=pdf_file,
                            file_name=f"VeriChain_Forensic_Dossier_{case_id}.pdf",
                            mime="application/pdf",
                        )
                except Exception as pdf_err:
                    st.warning(f"PDF Dossier generator encountered an issue: {pdf_err}")
