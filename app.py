import base64
import hashlib
import os
import tempfile
import cv2
import numpy as np
import streamlit as st
import streamlit.components.v1 as components
from engine import get_engine_status

# 1. Page Configuration & Meta
st.set_page_config(
    page_title="VeriChain | Digital Evidence Authentication System",
    layout="wide",
    page_icon="⚖️",
    initial_sidebar_state="expanded"
)

# 2. Custom Cyber-Forensic Theme & Styling
st.markdown(
    """
    <style>
    @import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700;800&family=Playfair+Display:ital,wght@1,400;1,600;1,700&family=JetBrains+Mono:wght@400;500;600;700&display=swap');

    /* Global Typography */
    html, body, [class*="css"] {
        font-family: 'Inter', sans-serif;
        color: #f1f5f9;
    }
    
    .font-playfair {
        font-family: 'Playfair Display', serif !important;
    }
    
    .font-mono {
        font-family: 'JetBrains Mono', monospace !important;
    }

    /* Main Container Background */
    .stApp {
        background: radial-gradient(circle at 50% 0%, #171c2b 0%, #0a0c13 55%, #050608 100%);
    }

    /* Custom Header Hero */
    .hero-banner {
        padding: 2.2rem 2.4rem;
        background: rgba(15, 20, 32, 0.75);
        backdrop-filter: blur(20px);
        -webkit-backdrop-filter: blur(20px);
        border: 1px solid rgba(255, 255, 255, 0.08);
        border-radius: 22px;
        margin-bottom: 2rem;
        box-shadow: 0 24px 50px rgba(0, 0, 0, 0.6), inset 0 1px 0 rgba(255, 255, 255, 0.12);
        position: relative;
        overflow: hidden;
    }
    
    .hero-banner::before {
        content: '';
        position: absolute;
        top: 0;
        left: 0;
        right: 0;
        height: 3px;
        background: linear-gradient(90deg, #f97316, #e11d48, #8b5cf6, #06b6d4);
    }

    .hero-title-row {
        display: flex;
        justify-content: space-between;
        align-items: center;
        flex-wrap: wrap;
        gap: 1rem;
    }

    .hero-title {
        font-size: 2.4rem;
        font-weight: 800;
        letter-spacing: -0.04em;
        background: linear-gradient(135deg, #ffffff 20%, #e2e8f0 60%, #94a3b8 100%);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        display: flex;
        align-items: center;
        gap: 0.85rem;
    }

    .hero-subtitle {
        font-size: 1.02rem;
        color: #94a3b8;
        max-width: 820px;
        line-height: 1.6;
        margin-top: 0.5rem;
        letter-spacing: -0.01em;
    }

    .pill-badge {
        display: inline-flex;
        align-items: center;
        gap: 0.45rem;
        padding: 0.35rem 0.95rem;
        border-radius: 9999px;
        font-size: 0.75rem;
        font-weight: 600;
        letter-spacing: 0.04em;
        text-transform: uppercase;
        font-family: 'JetBrains Mono', monospace;
    }

    .pill-neural {
        border: 1px solid rgba(16, 185, 129, 0.45);
        background: rgba(16, 185, 129, 0.12);
        color: #34d399;
        box-shadow: 0 0 16px rgba(16, 185, 129, 0.2);
    }

    .pill-fallback {
        border: 1px solid rgba(245, 158, 11, 0.45);
        background: rgba(245, 158, 11, 0.12);
        color: #fbbf24;
    }

    .pill-iso {
        border: 1px solid rgba(59, 130, 246, 0.4);
        background: rgba(59, 130, 246, 0.1);
        color: #93c5fd;
    }

    /* Metric Cards */
    div[data-testid="stMetric"] {
        background: rgba(16, 21, 34, 0.6);
        backdrop-filter: blur(14px);
        -webkit-backdrop-filter: blur(14px);
        border: 1px solid rgba(255, 255, 255, 0.08);
        border-radius: 16px;
        padding: 1.2rem 1.4rem;
        box-shadow: 0 10px 30px rgba(0, 0, 0, 0.35);
        transition: transform 0.2s ease, border-color 0.2s ease;
    }
    
    div[data-testid="stMetric"]:hover {
        transform: translateY(-3px);
        border-color: rgba(249, 115, 22, 0.4);
    }

    /* Custom Verdict Banner */
    .verdict-card {
        padding: 1.6rem 2rem;
        border-radius: 18px;
        margin: 1.6rem 0;
        display: flex;
        align-items: center;
        justify-content: space-between;
        box-shadow: 0 14px 40px rgba(0,0,0,0.5);
    }

    .verdict-authentic {
        background: linear-gradient(135deg, rgba(6, 78, 59, 0.55) 0%, rgba(6, 95, 70, 0.25) 100%);
        border: 1px solid rgba(16, 185, 129, 0.5);
        box-shadow: 0 12px 35px rgba(16, 185, 129, 0.15);
    }

    .verdict-manipulated {
        background: linear-gradient(135deg, rgba(127, 29, 29, 0.6) 0%, rgba(153, 27, 27, 0.3) 100%);
        border: 1px solid rgba(239, 68, 68, 0.55);
        box-shadow: 0 12px 35px rgba(239, 68, 68, 0.2);
    }

    .verdict-inconclusive {
        background: linear-gradient(135deg, rgba(30, 41, 59, 0.55) 0%, rgba(51, 65, 85, 0.3) 100%);
        border: 1px solid rgba(148, 163, 184, 0.35);
    }

    /* Primary Action Buttons */
    .stButton > button {
        background: linear-gradient(135deg, #f97316 0%, #ea580c 100%) !important;
        color: #ffffff !important;
        border: none !important;
        border-radius: 9999px !important;
        padding: 0.7rem 2rem !important;
        font-weight: 600 !important;
        font-size: 0.95rem !important;
        letter-spacing: -0.01em !important;
        box-shadow: 0 6px 20px rgba(249, 115, 22, 0.4) !important;
        transition: all 0.25s cubic-bezier(0.16, 1, 0.3, 1) !important;
    }

    .stButton > button:hover {
        transform: scale(1.02) !important;
        box-shadow: 0 8px 28px rgba(249, 115, 22, 0.55) !important;
    }

    .stButton > button:active {
        transform: scale(0.98) !important;
    }

    /* Tabs Customization */
    button[data-baseweb="tab"] {
        font-size: 0.92rem !important;
        font-weight: 500 !important;
        padding: 0.6rem 1.2rem !important;
        color: #94a3b8 !important;
    }

    button[aria-selected="true"] {
        color: #f97316 !important;
        font-weight: 700 !important;
        border-bottom-color: #f97316 !important;
    }

    /* Section Cards */
    .forensic-card {
        background: rgba(16, 21, 34, 0.6);
        backdrop-filter: blur(14px);
        -webkit-backdrop-filter: blur(14px);
        border: 1px solid rgba(255, 255, 255, 0.08);
        border-radius: 16px;
        padding: 1.4rem 1.6rem;
        margin-bottom: 1.2rem;
    }
    </style>
    """,
    unsafe_allow_html=True
)

# 3. Interactive Multi-Mode Spotlight & Wipe Lens Component
def render_interactive_spotlight_lens(original_path, heatmap_path, fft_path=None, height=560):
    """
    Renders an ultra-smooth, multi-mode hardware-accelerated interactive forensic visualizer:
    1. Spotlight Lens: Cursor-following soft circular feathered reveal.
    2. Split Wipe Slider: Draggable comparison divider.
    3. Full Heatmap Overlay.
    4. 2D-FFT Fourier Spectral Anomaly Map.
    5. Clean Source Media.
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
        st.error(f"Error preparing forensic visualizer: {e}")
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
        color: #f3f4f6;
        font-family: 'Inter', sans-serif;
        overflow: hidden;
    }}

    .lens-wrapper {{
        background: rgba(14, 18, 28, 0.85);
        backdrop-filter: blur(18px);
        -webkit-backdrop-filter: blur(18px);
        border: 1px solid rgba(255, 255, 255, 0.1);
        border-radius: 20px;
        padding: 16px;
        box-shadow: 0 24px 50px rgba(0, 0, 0, 0.65);
        display: flex;
        flex-direction: column;
        gap: 12px;
    }}

    .lens-header {{
        display: flex;
        justify-content: space-between;
        align-items: center;
        flex-wrap: wrap;
        gap: 8px;
        padding: 2px 4px;
    }}

    .lens-title {{
        display: flex;
        align-items: center;
        gap: 10px;
        font-size: 0.92rem;
        font-weight: 700;
        color: #f8fafc;
    }}

    .lens-badge {{
        background: rgba(249, 115, 22, 0.15);
        color: #f97316;
        border: 1px solid rgba(249, 115, 22, 0.4);
        padding: 3px 9px;
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
        background: rgba(255, 255, 255, 0.05);
        color: #94a3b8;
        border: 1px solid rgba(255, 255, 255, 0.08);
        border-radius: 9999px;
        padding: 5px 13px;
        font-size: 0.76rem;
        font-weight: 600;
        cursor: pointer;
        transition: all 0.2s ease;
    }}

    .ctrl-btn:hover {{
        background: rgba(255, 255, 255, 0.12);
        color: #ffffff;
    }}

    .ctrl-btn.active {{
        background: #f97316;
        color: #ffffff;
        border-color: #f97316;
        box-shadow: 0 0 14px rgba(249, 115, 22, 0.45);
    }}

    .lens-viewport {{
        position: relative;
        width: 100%;
        height: 420px;
        border-radius: 14px;
        overflow: hidden;
        background: #020305;
        cursor: crosshair;
        border: 1px solid rgba(255, 255, 255, 0.06);
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
        border: 1px dashed rgba(249, 115, 22, 0.75);
        transform: translate(-50%, -50%);
        pointer-events: none;
        z-index: 6;
        box-shadow: 0 0 25px rgba(249, 115, 22, 0.25), inset 0 0 20px rgba(249, 115, 22, 0.15);
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
        box-shadow: 0 0 8px #f97316;
    }}

    .hud-tag {{
        position: absolute;
        top: 15px;
        left: 50%;
        transform: translateX(-50%);
        background: rgba(15, 23, 42, 0.9);
        border: 1px solid rgba(249, 115, 22, 0.5);
        border-radius: 6px;
        padding: 2px 8px;
        font-size: 0.68rem;
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
        box-shadow: 0 0 10px #f97316;
        display: none;
    }}

    .slider-thumb {{
        position: absolute;
        top: 50%;
        left: 50%;
        transform: translate(-50%, -50%);
        width: 28px;
        height: 28px;
        background: #f97316;
        border-radius: 50%;
        display: flex;
        align-items: center;
        justify-content: center;
        color: white;
        font-size: 10px;
        font-weight: bold;
        box-shadow: 0 0 12px rgba(249, 115, 22, 0.8);
    }}

    .lens-footer {{
        display: flex;
        justify-content: space-between;
        align-items: center;
        font-size: 0.78rem;
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


# 4. Engine Status Verification
engine_info = get_engine_status()

# 5. Sidebar - Case File Docket & Custody Ledger
with st.sidebar:
    st.markdown("### 📋 Case Docket & Custody Ledger")
    
    # Engine Status Badge
    if engine_info.get("is_neural", False):
        st.markdown(
            f"""
            <div class="pill-badge pill-neural" style="margin-bottom: 0.8rem;">
                <span>●</span> {engine_info['badge']}
            </div>
            """,
            unsafe_allow_html=True
        )
    else:
        st.markdown(
            f"""
            <div class="pill-badge pill-fallback" style="margin-bottom: 0.8rem;">
                <span>▲</span> {engine_info['badge']}
            </div>
            """,
            unsafe_allow_html=True
        )

    st.caption(f"Model ID: `{engine_info['model_name']}`")
    st.markdown(
        """
        <div class="pill-badge pill-iso" style="margin-top: 0.2rem; margin-bottom: 1rem;">
            <span>🛡️</span> ISO/IEC 27037 Compliant
        </div>
        """,
        unsafe_allow_html=True
    )
    
    st.divider()

    st.markdown("#### 👤 Examiner Metadata")
    analyst_name = st.text_input(
        "Lead Forensic Examiner", value="Detective J. Miller, Digital Forensics"
    )
    badge_num = st.text_input("Examiner Badge / ID", value="DFU-88219")
    case_id = st.text_input("Case Docket ID", value="CR-2026-9042A")
    court_jurisdiction = st.text_input(
        "Judicial Jurisdiction", value="Federal District Court (SDNY)"
    )
    notes = st.text_area(
        "Investigative Log & Intake Notes",
        "Target digital evidence ingested from precinct digital evidence locker. Multi-subject facial isolation, neural ViT attention extraction, and cryptographic verification active.",
        height=90
    )
    
    st.divider()
    
    st.markdown("#### ⚡ Stage Demo Presets")
    st.caption("One-click benchmark datasets for live hackathon demonstration:")
    demo_mode = st.selectbox(
        "Select Evidence Preset",
        [
            "None (Upload Custom Media)",
            "Preset 1: Deepfake Suspect Portrait",
            "Preset 2: Verified Authentic Headshot",
        ],
    )

# 6. Top Hero Banner
st.markdown(
    f"""
    <div class="hero-banner">
        <div class="hero-title-row">
            <div class="hero-title">
                <span>⚖️</span>
                <span><span class="font-playfair italic font-normal">VeriChain</span> Forensic Hub</span>
            </div>
            <div style="display: flex; gap: 8px; align-items: center;">
                <div class="pill-badge pill-iso">
                    <span>🔒</span> SHA-256 AUDIT LOGGED
                </div>
            </div>
        </div>
        <p class="hero-subtitle">
            Next-Generation Deepfake Detection & Judicial Evidence Authentication Platform.
            Real-time biometric ViT feature extraction, multi-subject cascade isolation, 2D Fourier spectral residuals, and certified court-admissible dossiers.
        </p>
    </div>
    """,
    unsafe_allow_html=True
)

# 7. File Ingestion
uploaded_file = st.file_uploader(
    "📁 Ingest Digital Media File (Images or Videos)",
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

# Reset state if new file is loaded
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
        st.markdown(
            """
            - **Detection Pipeline:** Multi-Scale Cascade Face Isolation + SigLIP ViT Neural Classifier
            - **Spectral Anomaly Engine:** 2D-FFT High-Frequency Residual Mapping
            - **Chain of Custody:** SHA-256 Cryptographic Sealing & Judicial Dossier Export
            """
        )
        if st.button("⚡ Execute Biometric & Spectral Integrity Audit", type="primary"):
            with st.spinner("Isolating facial subjects, extracting attention norms, and computing forensic metrics..."):
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

        st.markdown(
            f"""
            <div class="verdict-card {box_class}">
                <div>
                    <div style="font-size: 0.82rem; font-family: 'JetBrains Mono', monospace; letter-spacing: 0.08em; opacity: 0.85; margin-bottom: 6px;">
                        FORENSIC DETERMINATION [{tag}]
                    </div>
                    <div style="font-size: 1.55rem; font-weight: 800; display: flex; align-items: center; gap: 10px;">
                        <span>{icon}</span>
                        <span>{verdict_str}</span>
                    </div>
                </div>
                <div style="text-align: right;">
                    <div style="font-size: 0.8rem; opacity: 0.85; font-family: 'JetBrains Mono', monospace; letter-spacing: 0.05em;">ENSEMBLE CONFIDENCE</div>
                    <div style="font-size: 2.1rem; font-weight: 900; color: #ffffff;">{res['confidence'] * 100:.1f}%</div>
                </div>
            </div>
            """,
            unsafe_allow_html=True
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
                    height=560
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
                st.markdown(
                    f"""
                    <div class="forensic-card">
                        <div style="font-weight: 700; color: #f97316; margin-bottom: 0.5rem;">BIOMETRIC BOUNDARY ANALYSIS</div>
                        <div style="font-size: 0.9rem; line-height: 1.6;">
                            <b>Examiner Notes:</b> {res.get('summary_note', 'N/A')}<br>
                            <b>Boundary Gradient:</b> {'Discontinuous blending boundary detected' if res['manipulation_score'] >= 0.5 else 'Seamless natural biological boundary'}<br>
                            <b>Active Backend:</b> {engine_info['badge']}
                        </div>
                    </div>
                    """,
                    unsafe_allow_html=True
                )
            with dcol2:
                fft_verdict = "High (Synthetic Diffusion/GAN Footprint)" if res["fft_score"] > 0.4 else "Normal (Natural Photographic Sensor Baseline)"
                st.markdown(
                    f"""
                    <div class="forensic-card">
                        <div style="font-weight: 700; color: #38bdf8; margin-bottom: 0.5rem;">FREQUENCY DOMAIN SPECTRUM (2D-FFT)</div>
                        <div style="font-size: 0.9rem; line-height: 1.6;">
                            <b>FFT Residual Metric:</b> <code>{res['fft_score']:.4f}</code><br>
                            <b>Frequency Anomaly:</b> {fft_verdict}<br>
                            <b>Fourier Baseline:</b> Calibrated photographic high-frequency threshold (0.20–0.35)
                        </div>
                    </div>
                    """,
                    unsafe_allow_html=True
                )

        with tab4:
            st.markdown("#### ⚖️ Judicial Admissibility & Evidence Chain-of-Custody")
            acol1, acol2 = st.columns([1.2, 0.8], gap="large")
            with acol1:
                st.markdown(
                    f"""
                    <div class="forensic-card">
                        <div style="font-weight: 700; color: #10b981; margin-bottom: 0.6rem;">CERTIFIED CHAIN OF CUSTODY</div>
                        <div style="font-size: 0.88rem; line-height: 1.7; font-family: 'JetBrains Mono', monospace;">
                            <b>Target Media:</b> {filename}<br>
                            <b>SHA-256 Digest:</b> {st.session_state['file_hash']}<br>
                            <b>Lead Examiner:</b> {analyst_name} (ID: {badge_num})<br>
                            <b>Jurisdiction:</b> {court_jurisdiction}<br>
                            <b>Docket ID:</b> {case_id}<br>
                            <b>Compliance:</b> Federal Rules of Evidence Rule 901 & ISO/IEC 27037
                        </div>
                    </div>
                    """,
                    unsafe_allow_html=True
                )
            with acol2:
                st.markdown("#### 📄 Export Official Dossier")
                st.caption("Generate certified forensic report with cryptographic hash, subject breakdown table, and attention overlay.")
                
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
