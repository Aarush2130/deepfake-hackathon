import os
import cv2
import torch
import numpy as np
from PIL import Image
from transformers import pipeline

# 1. Attempt to load Hugging Face model; fail gracefully to FFT
try:
    print("Loading classification model...")
    classifier = pipeline(
        "image-classification", 
        model="prithivMLmods/Deep-Fake-Detector-v2-Model", 
        device=0 if torch.cuda.is_available() else -1
    )
    print("AI Model loaded successfully.")
except Exception as e:
    print(f"Warning: Model unavailable ({e}). Defaulting to FFT frequency analysis.")
    classifier = None

def get_fft_anomaly(gray_img):
    """Calculates frequency domain anomaly score via 2D FFT."""
    try:
        # Resize to standardized dimensions for uniform FFT energy comparison
        standard_gray = cv2.resize(gray_img, (256, 256))
        f = np.fft.fft2(standard_gray)
        fshift = np.fft.fftshift(f)
        magnitude = np.log(np.abs(fshift) + 1e-6)
        
        h, w = magnitude.shape
        cy, cx = h // 2, w // 2
        r = 25
        
        center = magnitude[cy - r:cy + r, cx - r:cx + r]
        outer_mean = (magnitude.sum() - center.sum()) / (h * w - (2 * r) ** 2 + 1e-6)
        ratio = float(center.mean() / (outer_mean + 1e-6))
        return float(np.clip((ratio - 0.8) * 1.5, 0.05, 0.98))
    except Exception:
        return 0.50

def generate_heatmap(image_bgr):
    """Generates an edge-discontinuity activation overlay."""
    try:
        # Downscale large images for rapid heatmap generation
        h, w = image_bgr.shape[:2]
        if max(h, w) > 720:
            scale = 720.0 / max(h, w)
            image_bgr = cv2.resize(image_bgr, (int(w * scale), int(h * scale)))

        gray = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2GRAY)
        blurred = cv2.GaussianBlur(gray, (9, 9), 0)
        edges = cv2.Laplacian(blurred, cv2.CV_64F)
        edges = np.uint8(np.absolute(edges))
        edges = cv2.GaussianBlur(edges, (15, 15), 0)
        
        norm_cam = cv2.normalize(edges, None, 0, 255, cv2.NORM_MINMAX, dtype=cv2.CV_8U)
        heatmap = cv2.applyColorMap(norm_cam, cv2.COLORMAP_JET)
        return cv2.addWeighted(image_bgr, 0.6, heatmap, 0.4, 0)
    except Exception:
        return image_bgr

def analyze_image(image_path):
    """Analyzes a single image file safely."""
    img_bgr = cv2.imread(image_path)
    if img_bgr is None:
        raise ValueError(f"Corrupted or unsupported image file: {image_path}")
        
    gray = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2GRAY)
    fft_score = get_fft_anomaly(gray)
    fake_prob = fft_score

    # Run neural classifier if available
    if classifier:
        try:
            img_rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)
            pil_img = Image.fromarray(img_rgb)
            preds = classifier(pil_img)
            for p in preds:
                lbl = p["label"].lower()
                if "fake" in lbl or "synthetic" in lbl or "manipulated" in lbl:
                    fake_prob = float(p["score"])
                    break
                elif "real" in lbl or "authentic" in lbl:
                    fake_prob = 1.0 - float(p["score"])
                    break
        except Exception:
            fake_prob = fft_score

    is_fake = fake_prob >= 0.50
    verdict = "Manipulated (Deepfake)" if is_fake else "Authentic Media"
    heatmap = generate_heatmap(img_bgr)
    
    heatmap_path = "temp_heatmap.png"
    cv2.imwrite(heatmap_path, heatmap)
    
    return {
        "verdict": verdict,
        "confidence": fake_prob if is_fake else (1.0 - fake_prob),
        "manipulation_score": fake_prob,
        "fft_score": fft_score,
        "heatmap_path": heatmap_path
    }

def analyze_video(video_path, max_frames=12):
    """Samples video frames quickly and isolates the worst anomaly."""
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        raise ValueError(f"Could not open video source: {video_path}")
        
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    step = max(1, total_frames // max_frames)
    
    frame_idx = 0
    scores = []
    worst_score = -1.0
    worst_frame = None
    
    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break
        if frame_idx % step == 0:
            temp_path = f"temp_frame_{frame_idx}.png"
            cv2.imwrite(temp_path, frame)
            try:
                res = analyze_image(temp_path)
                scores.append(res["manipulation_score"])
                if res["manipulation_score"] > worst_score:
                    worst_score = res["manipulation_score"]
                    worst_frame = frame
            finally:
                if os.path.exists(temp_path):
                    os.remove(temp_path)
        frame_idx += 1
        
    cap.release()
    
    avg_score = float(np.mean(scores)) if scores else 0.50
    is_fake = avg_score >= 0.50
    verdict = "Manipulated (Deepfake)" if is_fake else "Authentic Media"
    
    worst_heatmap_path = "temp_heatmap.png"
    if worst_frame is not None:
        cv2.imwrite(worst_heatmap_path, generate_heatmap(worst_frame))
        
    return {
        "verdict": verdict,
        "confidence": avg_score if is_fake else (1.0 - avg_score),
        "manipulation_score": avg_score,
        "fft_score": float(np.max(scores)) if scores else avg_score,
        "heatmap_path": worst_heatmap_path
    }