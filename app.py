import hashlib
import os
import tempfile
import cv2
import numpy as np
import streamlit as st

# Set UI Configuration
st.set_page_config(
    page_title="VeriChain Evidence Hub",
    layout="wide",
    page_icon="⚖️"
)

st.title("⚖️ VeriChain: Digital Evidence Authentication System")
st.caption(
    "Forensic Media Integrity & Multi-Subject Chain-of-Custody Suite for Law Enforcement and Judicial Inquests"
)

# Sidebar - Evidence Intake & Case Metadata
with st.sidebar:
    st.header("📋 Case File Docket")
    analyst_name = st.text_input(
        "Investigator / Analyst", value="Detective J. Miller, Digital Forensics"
    )
    case_id = st.text_input("Case Docket ID", value="CR-2026-9042A")
    court_jurisdiction = st.text_input(
        "Jurisdiction", value="Federal District Court"
    )
    notes = st.text_area(
        "Investigative Notes",
        "Target evidence ingested from precinct evidence locker. Multi-subject facial isolation and cryptographic verification active.",
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

# File Uploader
uploaded_file = st.file_uploader(
    "Ingest Digital Media File", type=["png", "jpg", "jpeg", "mp4", "mov"]
)

active_path = None
file_bytes = None
filename = ""

# Handle File Ingestion or Preset Loading
if uploaded_file is not None:
    file_bytes = uploaded_file.getvalue()
    filename = uploaded_file.name
    suffix = os.path.splitext(filename)[1].lower()

    # Write to temporary file safely
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
    st.session_state.pop("results", None)

if active_path and file_bytes:
    sha256_hash = hashlib.sha256(file_bytes).hexdigest()

    with st.sidebar:
        st.subheader("🔒 Cryptographic Custody")
        st.code(sha256_hash, language="text")
        st.success("SHA-256 Fingerprint Anchored")

    col1, col2 = st.columns([1, 1])
    with col1:
        st.subheader("Source Evidence")
        try:
            if filename.lower().endswith((".mp4", ".mov")):
                st.video(active_path)
            else:
                st.image(active_path)
        except Exception:
            st.error("⚠️ Media viewer preview failed for this format.")

    with col2:
        st.subheader("Forensic Examination")
        if st.button(
            "🚀 Run Biometric & Spectral Integrity Audit",
            type="primary",
        ):
            with st.spinner(
                "Detecting facial subjects, isolating crops, and analyzing spectral anomalies..."
            ):
                try:
                    # Dynamically route between Video and Image Engine
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
                    st.error(f"⚠️ Audit failed to process media safely: {err}")

    # Display Results if Available
    if "results" in st.session_state:
        res = st.session_state["results"]
        st.divider()

        # Primary Verdict Header
        verdict_str = res.get("verdict", "Unknown")
        is_deepfake = "Manipulated" in verdict_str or "Deepfake" in verdict_str
        is_no_face = res.get("status") == "NO_FACE_DETECTED" or res.get("face_count", 0) == 0

        if is_no_face:
            st.info(f"🔍 **Forensic Verdict:** {verdict_str}")
        elif is_deepfake:
            st.error(f"🚨 **Forensic Verdict:** {verdict_str}")
        else:
            st.success(f"✅ **Forensic Verdict:** {verdict_str}")

        mcol1, mcol2, mcol3, mcol4 = st.columns(4)
        with mcol1:
            st.metric("Detected Subjects", f"{res.get('face_count', 0)} Face(s)")
        with mcol2:
            st.metric("Aggregate Confidence", f"{res['confidence'] * 100:.2f}%")
        with mcol3:
            st.metric("Peak Manipulation Score", f"{res['manipulation_score']:.3f}")
        with mcol4:
            st.metric("Spectral Residual Metric", f"{res['fft_score']:.4f}")

        if is_no_face:
            st.warning(
                "ℹ️ **Zero Facial Subjects Detected:** The system engaged non-facial spectral frequency & sensor noise mode. "
                "For facial biometric and deepfake analysis, provide media containing human subjects."
            )

        # Subject-by-Subject Forensic Breakdown (if faces exist)
        faces_list = res.get("faces", [])
        if faces_list:
            st.subheader(f"👥 Subject-by-Subject Facial Isolation ({len(faces_list)} Subjects)")
            
            # Display Table of Subjects
            table_data = []
            for f in faces_list:
                subj_id = f"Subject #{f['subject_id']}"
                bbox = f"[{f['bbox'][0]}, {f['bbox'][1]}, {f['bbox'][2]}, {f['bbox'][3]}]"
                subj_verdict = "🚨 Manipulated (Deepfake)" if f["manipulation_score"] >= 0.50 else "✅ Authentic"
                manip_prob = f"{f['manipulation_score']*100:.1f}%"
                conf = f"{f['confidence']*100:.1f}%"
                table_data.append({
                    "Subject": subj_id,
                    "Coordinates (X, Y, W, H)": bbox,
                    "Determination": subj_verdict,
                    "Manipulation Probability": manip_prob,
                    "Confidence": conf
                })
            
            st.dataframe(table_data, hide_index=True)

        st.subheader("🔍 Visual Evidence & Boundary Artifact Localization")
        rcol1, rcol2 = st.columns(2)
        with rcol1:
            if "heatmap_path" in res and os.path.exists(res["heatmap_path"]):
                st.image(
                    res["heatmap_path"],
                    caption="Neural Attention, Subject Bounding Boxes & Spectral Edge Overlay"
                )
            else:
                st.info("Heatmap visualization unavailable for this input.")

        with rcol2:
            st.markdown("#### Forensic Anomaly Breakdown")
            st.write(f"- **Examiner Summary:** {res.get('summary_note', 'N/A')}")
            st.write(
                "- **FFT High-Frequency Residual:** "
                f"`{'High (Synthetic GAN/Diffusion footprint)' if res['fft_score'] > 0.4 else 'Normal (Natural Sensor Noise)'}`"
            )
            st.write(
                "- **Facial Boundary & Edge Continuity:** "
                f"`{'Discontinuous blending boundary detected' if res['manipulation_score'] > 0.5 else 'Seamless natural boundary'}`"
            )

            # PDF Generation with Report Module
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
                        label="📄 Download Court-Admissible Forensic Dossier (PDF)",
                        data=pdf_file,
                        file_name=f"VeriChain_Dossier_{case_id}.pdf",
                        mime="application/pdf",
                    )
            except Exception as pdf_err:
                st.warning(f"PDF Dossier generator encountered an issue: {pdf_err}")