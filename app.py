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
    page_title="VeriChain Forensic Evidence Hub",
    layout="wide",
    page_icon="⚖️",
    initial_sidebar_state="expanded"
)

# 2. Custom Cyber-Forensic Styling & Glassmorphism Theme
st.markdown(
    """
    <style>
    @import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700&family=Playfair+Display:ital,wght@1,400;1,500;1,600;1,700&family=JetBrains+Mono:wght@400;500;700&display=swap');

    /* Global Typography */
    html, body, [class*="css"] {
        font-family: 'Inter', sans-serif;
        color: #f3f4f6;
    }
    
    .font-playfair {
        font-family: 'Playfair Display', serif !important;
    }
    
    .font-mono {
        font-family: 'JetBrains Mono', monospace !important;
    }

    /* Main Container Background */
    .stApp {
        background: radial-gradient(circle at 50% 0%, #151926 0%, #090a0f 65%, #050608 100%);
    }

    /* Custom Header Hero */
    .hero-container {
        padding: 2.2rem 1.8rem;
        background: rgba(18, 22, 34, 0.65);
        backdrop-filter: blur(18px);
        -webkit-backdrop-filter: blur(18px);
        border: 1px solid rgba(255, 255, 255, 0.08);
        border-radius: 20px;
        margin-bottom: 2rem;
        box-shadow: 0 20px 40px rgba(0, 0, 0, 0.5), inset 0 1px 0 rgba(255, 255, 255, 0.1);
        position: relative;
        overflow: hidden;
    }
    
    .hero-container::before {
        content: '';
        position: absolute;
        top: 0;
        left: 0;
        right: 0;
        height: 2px;
        background: linear-gradient(90deg, transparent, #f97316, #e11d48, #3b82f6, transparent);
    }

    .hero-title {
        font-size: 2.5rem;
        font-weight: 700;
        letter-spacing: -0.03em;
        margin-bottom: 0.35rem;
        background: linear-gradient(135deg, #ffffff 30%, #cbd5e1 70%, #94a3b8 100%);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        display: flex;
        align-items: center;
        gap: 0.75rem;
    }

    .hero-subtitle {
        font-size: 1.05rem;
        color: #94a3b8;
        max-width: 820px;
        line-height: 1.5;
        letter-spacing: -0.01em;
    }

    .badge-pill {
        display: inline-flex;
        align-items: center;
        gap: 0.4rem;
        padding: 0.3rem 0.85rem;
        border-radius: 9999px;
        font-size: 0.8rem;
        font-weight: 600;
        letter-spacing: 0.02em;
        text-transform: uppercase;
        border: 1px solid rgba(255, 255, 255, 0.12);
        background: rgba(255, 255, 255, 0.04);
    }

    .badge-neural {
        border-color: rgba(16, 185, 129, 0.4);
        background: rgba(16, 185, 129, 0.1);
        color: #34d399;
    }

    .badge-fallback {
        border-color: rgba(245, 158, 11, 0.4);
        background: rgba(245, 158, 11, 0.1);
        color: #fbbf24;
    }

    /* Metric Cards */
    div[data-testid="stMetric"] {
        background: rgba(18, 22, 34, 0.55);
        backdrop-filter: blur(12px);
        -webkit-backdrop-filter: blur(12px);
        border: 1px solid rgba(255, 255, 255, 0.07);
        border-radius: 14px;
        padding: 1.1rem 1.3rem;
        box-shadow: 0 8px 24px rgba(0, 0, 0, 0.3);
        transition: transform 0.2s ease, border-color 0.2s ease;
    }
    
    div[data-testid="stMetric"]:hover {
        transform: translateY(-2px);
        border-color: rgba(249, 115, 22, 0.35);
    }

    /* Custom Verdict Banner */
    .verdict-box {
        padding: 1.4rem 1.8rem;
        border-radius: 16px;
        margin: 1.5rem 0;
        display: flex;
        align-items: center;
        justify-content: space-between;
        box-shadow: 0 12px 30px rgba(0,0,0,0.4);
    }

    .verdict-authentic {
        background: linear-gradient(135deg, rgba(6, 78, 59, 0.45) 0%, rgba(6, 95, 70, 0.2) 100%);
        border: 1px solid rgba(16, 185, 129, 0.4);
    }

    .verdict-manipulated {
        background: linear-gradient(135deg, rgba(127, 29, 29, 0.5) 0%, rgba(153, 27, 27, 0.25) 100%);
        border: 1px solid rgba(239, 68, 68, 0.45);
    }

    .verdict-inconclusive {
        background: linear-gradient(135deg, rgba(30, 41, 59, 0.5) 0%, rgba(51, 65, 85, 0.25) 100%);
        border: 1px solid rgba(148, 163, 184, 0.3);
    }

    /* Primary Action Buttons */
    .stButton > button {
        background: linear-gradient(135deg, #f97316 0%, #ea580c 100%) !important;
        color: #ffffff !important;
        border: none !important;
        border-radius: 9999px !important;
        padding: 0.65rem 1.8rem !important;
        font-weight: 600 !important;
        letter-spacing: -0.01em !important;
        box-shadow: 0 4px 18px rgba(249, 115, 22, 0.35) !important;
        transition: all 0.25s cubic-bezier(0.16, 1, 0.3, 1) !important;
    }

    .stButton > button:hover {
        transform: scale(1.02) !important;
        box-shadow: 0 6px 24px rgba(249, 115, 22, 0.5) !important;
    }

    .stButton > button:active {
        transform: scale(0.97) !important;
    }

    /* Dataframe Table Styling */
    div[data-testid="stDataFrame"] {
        border: 1px solid rgba(255, 255, 255, 0.08);
        border-radius: 12px;
        overflow: hidden;
    }
    </style>
    """,
    unsafe_allow_html=True
)

# 3. Interactive Forensic Spotlight Reveal Component Helper
def render_interactive_spotlight_lens(original_path, heatmap_path, height=540):
    """
    Renders an ultra-smooth, hardware-accelerated interactive spotlight reveal canvas in Streamlit.
    Hovering the cursor over the evidence dynamically reveals the neural feature heatmap through a soft circular lens.
    """
    try:
        with open(original_path, "rb") as f:
            b64_orig = base64.b64encode(f.read()).decode("utf-8")
        with open(heatmap_path, "rb") as f:
            b64_heat = base64.b64encode(f.read()).decode("utf-8")
    except Exception as e:
        st.error(f"Error preparing spotlight visualizer: {e}")
        return

    html_code = f"""
    <!DOCTYPE html>
    <html>
    <head>
    <style>
    @import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700&family=JetBrains+Mono:wght@400;500&display=swap');
    
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
        backdrop-filter: blur(16px);
        -webkit-backdrop-filter: blur(16px);
        border: 1px solid rgba(255, 255, 255, 0.1);
        border-radius: 18px;
        padding: 14px;
        box-shadow: 0 20px 40px rgba(0, 0, 0, 0.6);
        display: flex;
        flex-direction: column;
        gap: 10px;
    }}

    .lens-header {{
        display: flex;
        justify-content: space-between;
        align-items: center;
        padding: 2px 6px;
    }}

    .lens-title {{
        display: flex;
        align-items: center;
        gap: 8px;
        font-size: 0.88rem;
        font-weight: 600;
        color: #e2e8f0;
    }}

    .lens-badge {{
        background: rgba(249, 115, 22, 0.15);
        color: #f97316;
        border: 1px solid rgba(249, 115, 22, 0.4);
        padding: 2px 8px;
        border-radius: 9999px;
        font-size: 0.7rem;
        font-family: 'JetBrains Mono', monospace;
        letter-spacing: 0.05em;
    }}

    .lens-controls {{
        display: flex;
        gap: 6px;
    }}

    .ctrl-btn {{
        background: rgba(255, 255, 255, 0.06);
        color: #94a3b8;
        border: 1px solid rgba(255, 255, 255, 0.08);
        border-radius: 9999px;
        padding: 4px 12px;
        font-size: 0.75rem;
        font-weight: 500;
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
        box-shadow: 0 0 12px rgba(249, 115, 22, 0.4);
    }}

    .lens-viewport {{
        position: relative;
        width: 100%;
        height: 420px;
        border-radius: 12px;
        overflow: hidden;
        background: #020305;
        cursor: crosshair;
        border: 1px solid rgba(255, 255, 255, 0.05);
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
        transition: opacity 0.25s ease;
    }}

    .hud-reticle {{
        position: absolute;
        width: 360px;
        height: 360px;
        border-radius: 50%;
        border: 1px dashed rgba(249, 115, 22, 0.7);
        transform: translate(-50%, -50%);
        pointer-events: none;
        z-index: 5;
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
        background: rgba(15, 23, 42, 0.85);
        border: 1px solid rgba(249, 115, 22, 0.5);
        border-radius: 6px;
        padding: 2px 8px;
        font-size: 0.65rem;
        font-family: 'JetBrains Mono', monospace;
        color: #fdba74;
        white-space: nowrap;
    }}

    .lens-footer {{
        display: flex;
        justify-content: space-between;
        align-items: center;
        font-size: 0.75rem;
        color: #94a3b8;
        padding: 2px 6px;
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
                <span class="lens-badge">FORENSIC SPOTLIGHT</span>
                <span>Neural Feature & Anomaly Lens</span>
            </div>
            <div class="lens-controls">
                <button id="btnLens" class="ctrl-btn active" onclick="setInspectionMode('lens')">🔍 Spotlight Lens</button>
                <button id="btnHeatmap" class="ctrl-btn" onclick="setInspectionMode('heatmap')">🔥 Full Heatmap</button>
                <button id="btnOriginal" class="ctrl-btn" onclick="setInspectionMode('orig')">📷 Source Media</button>
            </div>
        </div>

        <div id="viewport" class="lens-viewport">
            <img id="baseImg" src="data:image/png;base64,{b64_orig}" class="layer-img" />
            <img id="revealImg" src="data:image/png;base64,{b64_heat}" class="layer-img reveal-img" />
            
            <div id="reticle" class="hud-reticle">
                <div class="hud-center-dot"></div>
                <div id="coordTag" class="hud-tag">LENS ACTIVE [R: 180px]</div>
            </div>
        </div>

        <div class="lens-footer">
            <span>💡 <b>Interactive Inspection:</b> Glide cursor across facial regions to peel back the layer and inspect ViT spatial focus.</span>
            <span id="telemetry" style="font-family: 'JetBrains Mono', monospace; color: #64748b;">READY</span>
        </div>
    </div>

    <script>
    const viewport = document.getElementById('viewport');
    const revealImg = document.getElementById('revealImg');
    const reticle = document.getElementById('reticle');
    const coordTag = document.getElementById('coordTag');
    const telemetry = document.getElementById('telemetry');

    let currentMode = 'lens';
    const SPOTLIGHT_RADIUS = 180;

    function applySpotlight(x, y) {{
        const maskCSS = `radial-gradient(circle ${{SPOTLIGHT_RADIUS}}px at ${{x}}px ${{y}}px, rgba(0,0,0,1) 0%, rgba(0,0,0,1) 45%, rgba(0,0,0,0.7) 65%, rgba(0,0,0,0.25) 85%, rgba(0,0,0,0) 100%)`;
        revealImg.style.maskImage = maskCSS;
        revealImg.style.webkitMaskImage = maskCSS;
        revealImg.style.opacity = '1';

        reticle.style.left = x + 'px';
        reticle.style.top = y + 'px';
        reticle.style.display = 'block';

        telemetry.innerText = `X: ${{Math.round(x)}}px | Y: ${{Math.round(y)}}px`;
    }}

    function setInspectionMode(mode) {{
        currentMode = mode;
        document.querySelectorAll('.ctrl-btn').forEach(btn => btn.classList.remove('active'));

        if (mode === 'lens') {{
            document.getElementById('btnLens').classList.add('active');
            reticle.style.display = 'none';
            revealImg.style.opacity = '0';
            revealImg.style.maskImage = 'none';
            revealImg.style.webkitMaskImage = 'none';
        }} else if (mode === 'heatmap') {{
            document.getElementById('btnHeatmap').classList.add('active');
            reticle.style.display = 'none';
            revealImg.style.maskImage = 'none';
            revealImg.style.webkitMaskImage = 'none';
            revealImg.style.opacity = '1';
            telemetry.innerText = 'HEATMAP OVERLAY';
        }} else if (mode === 'orig') {{
            document.getElementById('btnOriginal').classList.add('active');
            reticle.style.display = 'none';
            revealImg.style.maskImage = 'none';
            revealImg.style.webkitMaskImage = 'none';
            revealImg.style.opacity = '0';
            telemetry.innerText = 'ORIGINAL EVIDENCE';
        }}
    }}

    viewport.addEventListener('mousemove', (e) => {{
        if (currentMode !== 'lens') return;
        const rect = viewport.getBoundingClientRect();
        const x = e.clientX - rect.left;
        const y = e.clientY - rect.top;
        applySpotlight(x, y);
    }});

    viewport.addEventListener('mouseleave', () => {{
        if (currentMode === 'lens') {{
            reticle.style.display = 'none';
            revealImg.style.opacity = '0';
            telemetry.innerText = 'STANDBY';
        }}
    }});

    // Set initial mode
    setInspectionMode('lens');
    </script>
    </body>
    </html>
    """
    components.html(html_code, height=height, scrolling=False)


# 4. Engine Status
engine_info = get_engine_status()

# 5. Sidebar - Case Docket & Ingestion Metadata
with st.sidebar:
    st.markdown("### 📋 Case File Docket")
    
    # Live Engine Status Badge
    if engine_info.get("is_neural", False):
        st.markdown(
            f"""
            <div class="badge-pill badge-neural" style="margin-bottom: 0.75rem;">
                <span style="font-size: 0.9rem;">●</span> {engine_info['badge']}
            </div>
            """,
            unsafe_allow_html=True
        )
    else:
        st.markdown(
            f"""
            <div class="badge-pill badge-fallback" style="margin-bottom: 0.75rem;">
                <span style="font-size: 0.9rem;">▲</span> {engine_info['badge']}
            </div>
            """,
            unsafe_allow_html=True
        )

    st.caption(f"Model ID: `{engine_info['model_name']}`")
    st.divider()

    analyst_name = st.text_input(
        "Lead Examiner / Analyst", value="Detective J. Miller, Digital Forensics"
    )
    case_id = st.text_input("Case Docket ID", value="CR-2026-9042A")
    court_jurisdiction = st.text_input(
        "Jurisdiction", value="Federal District Court"
    )
    notes = st.text_area(
        "Investigative Notes",
        "Target evidence ingested from precinct evidence locker. Multi-subject facial isolation and cryptographic verification active.",
        height=100
    )
    st.divider()
    demo_mode = st.selectbox(
        "Stage Presets (Offline Insurance)",
        [
            "None (Upload Mode)",
            "Load Sample: Deepfake Suspect",
            "Load Sample: Verified Authentic",
        ],
    )

# 6. Main Hero Header
st.markdown(
    """
    <div class="hero-container">
        <div class="hero-title">
            <span>⚖️</span>
            <span><span class="font-playfair italic font-normal">VeriChain</span> Forensic Hub</span>
        </div>
        <p class="hero-subtitle">
            Certified Multi-Subject Digital Media Authentication & Biometric Chain-of-Custody Suite.
            Interactive spotlight reveal, spectral frequency analysis, and judicial dossier generation.
        </p>
    </div>
    """,
    unsafe_allow_html=True
)

# 7. File Ingestion
uploaded_file = st.file_uploader(
    "Ingest Digital Evidence (Images or Videos)",
    type=["png", "jpg", "jpeg", "mp4", "mov"],
    help="Upload PNG/JPG/MP4 evidence for biometric face detection and neural artifact analysis."
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

elif demo_mode != "None (Upload Mode)":
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

# Reset state if new file is loaded to prevent stale display
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

# 8. Forensic Processing Pipeline
if active_path and file_bytes:
    sha256_hash = hashlib.sha256(file_bytes).hexdigest()

    with st.sidebar:
        st.subheader("🔒 Cryptographic Chain of Custody")
        st.code(sha256_hash, language="text")
        st.caption("SHA-256 Checksum Immutable Fingerprint")

    col1, col2 = st.columns([1, 1], gap="large")
    with col1:
        st.subheader("📁 Source Evidence Ingestion")
        try:
            if filename.lower().endswith((".mp4", ".mov")):
                st.video(active_path)
            else:
                st.image(active_path, caption=f"Evidence: {filename}")
        except Exception:
            st.error("⚠️ Media viewer preview failed for this format.")

    with col2:
        st.subheader("🔬 Forensic Analysis Execution")
        st.markdown(
            "Execute deep neural ViT classification, multi-subject cascade isolation, and 2D-FFT frequency domain residual check."
        )
        if st.button("🚀 Run Biometric & Spectral Integrity Audit", type="primary"):
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

    # 9. Results Presentation
    if "results" in st.session_state:
        res = st.session_state["results"]
        st.divider()

        verdict_str = res.get("verdict", "Unknown")
        is_deepfake = "Manipulated" in verdict_str or "Deepfake" in verdict_str
        is_no_face = res.get("status") == "NO_FACE_DETECTED" or res.get("face_count", 0) == 0

        # Primary Determination Banner
        if is_no_face:
            box_class = "verdict-inconclusive"
            icon = "🔍"
            tag = "INCONCLUSIVE"
        elif is_deepfake:
            box_class = "verdict-manipulated"
            icon = "🚨"
            tag = "SYNTHETIC / MANIPULATED"
        else:
            box_class = "verdict-authentic"
            icon = "✅"
            tag = "AUTHENTIC MEDIA"

        st.markdown(
            f"""
            <div class="verdict-box {box_class}">
                <div>
                    <div style="font-size: 0.8rem; font-family: 'JetBrains Mono', monospace; letter-spacing: 0.08em; opacity: 0.8; margin-bottom: 4px;">
                        FORENSIC DETERMINATION [{tag}]
                    </div>
                    <div style="font-size: 1.4rem; font-weight: 700; display: flex; align-items: center; gap: 8px;">
                        <span>{icon}</span>
                        <span>{verdict_str}</span>
                    </div>
                </div>
                <div style="text-align: right;">
                    <div style="font-size: 0.8rem; opacity: 0.8; font-family: 'JetBrains Mono', monospace;">CONFIDENCE</div>
                    <div style="font-size: 1.8rem; font-weight: 800; color: #ffffff;">{res['confidence'] * 100:.1f}%</div>
                </div>
            </div>
            """,
            unsafe_allow_html=True
        )

        # Telemetry Metrics Grid
        mcol1, mcol2, mcol3, mcol4 = st.columns(4)
        with mcol1:
            st.metric("Detected Subjects", f"{res.get('face_count', 0)} Subject(s)")
        with mcol2:
            st.metric("Aggregate Confidence", f"{res['confidence'] * 100:.2f}%")
        with mcol3:
            st.metric("Peak Anomaly Score", f"{res['manipulation_score']:.3f}")
        with mcol4:
            st.metric("Spectral Residual (FFT)", f"{res['fft_score']:.4f}")

        # Subject-by-Subject Breakdown
        faces_list = res.get("faces", [])
        if faces_list:
            st.markdown(f"### 👥 Subject-by-Subject Facial Isolation ({len(faces_list)} Detected)")
            
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

        # 10. Interactive Spotlight Reveal Section
        st.markdown("### 🔍 Spatial Anomaly Spotlight Lens")
        rcol1, rcol2 = st.columns([1.1, 0.9], gap="large")
        
        with rcol1:
            if (
                "heatmap_path" in res
                and res["heatmap_path"]
                and os.path.exists(res["heatmap_path"])
                and not filename.lower().endswith((".mp4", ".mov"))
            ):
                render_interactive_spotlight_lens(active_path, res["heatmap_path"], height=520)
            elif "heatmap_path" in res and res["heatmap_path"] and os.path.exists(res["heatmap_path"]):
                st.image(res["heatmap_path"], caption="Peak Anomaly Frame Attention Overlay")
            else:
                st.info("Interactive visual overlay unavailable for this media format.")

        with rcol2:
            st.markdown("#### 🔬 Forensic Diagnostic Telemetry")
            st.markdown(f"**Examiner Summary:**\n`{res.get('summary_note', 'N/A')}`")
            
            fft_verdict = "High (Synthetic Diffusion/GAN Energy Signature)" if res["fft_score"] > 0.4 else "Normal (Natural Photographic Sensor Baseline)"
            boundary_verdict = "Discontinuous Blending Boundary Detected" if res["manipulation_score"] >= 0.50 else "Seamless Natural Gradient"
            
            st.markdown(f"- **Frequency Domain Residual:** `{fft_verdict}`")
            st.markdown(f"- **Biometric Boundary Continuity:** `{boundary_verdict}`")
            st.markdown(f"- **Active ViT Engine:** `{engine_info['badge']}`")

            st.divider()
            
            # Court Admissible PDF Dossier Generator
            try:
                from report import generate_pdf

                pdf_path = generate_pdf(
                    filename=st.session_state["filename"],
                    file_hash=st.session_state["file_hash"],
                    analyst=analyst_name,
                    verdict=res.get("verdict", "Unknown"),
                    confidence=res.get("confidence", 0.0),
                    heatmap_path=res.get("heatmap_path", None),
                    faces_data=res.get("faces", []),
                    summary_note=res.get("summary_note", "")
                )

                with open(pdf_path, "rb") as pdf_file:
                    st.download_button(
                        label="📄 Export Court-Admissible Forensic Dossier (PDF)",
                        data=pdf_file,
                        file_name=f"VeriChain_Dossier_{case_id}.pdf",
                        mime="application/pdf",
                    )
            except Exception as pdf_err:
                st.warning(f"PDF Dossier generator encountered an issue: {pdf_err}")