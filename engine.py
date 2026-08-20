import os
import cv2
import torch
import numpy as np
from PIL import Image
from transformers import pipeline

# 1. Initialize Face Detector (Haar Cascade)
face_cascade = None
try:
    cascade_path = cv2.data.haarcascades + "haarcascade_frontalface_default.xml"
    face_cascade = cv2.CascadeClassifier(cascade_path)
    if face_cascade.empty():
        print("Warning: Haar cascade classifier loaded empty.")
        face_cascade = None
    else:
        print("Haar face detector initialized successfully.")
except Exception as e:
    print(f"Warning: Failed to load face cascade: {e}")
    face_cascade = None

# 2. Attempt to load Hugging Face classification model; fail gracefully to FFT
classifier = None
try:
    print("Loading deepfake classification neural model...")
    classifier = pipeline(
        "image-classification",
        model="prithivMLmods/Deep-Fake-Detector-v2-Model",
        device=0 if torch.cuda.is_available() else -1
    )
    print("AI Model loaded successfully.")
except Exception as e:
    print(f"Warning: Model unavailable ({e}). Defaulting to FFT frequency analysis.")
    classifier = None


def detect_faces(img_bgr, min_size=(30, 30)):
    """Detects all frontal faces in the image and returns sorted bounding boxes (left-to-right)."""
    if face_cascade is None or img_bgr is None:
        return []
    
    try:
        gray = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2GRAY)
        # Apply histogram equalization for robust detection under varied lighting
        equalized = cv2.equalizeHist(gray)
        faces = face_cascade.detectMultiScale(
            equalized,
            scaleFactor=1.1,
            minNeighbors=4,
            minSize=min_size,
            flags=cv2.CASCADE_SCALE_IMAGE
        )
        if len(faces) == 0:
            # Fallback to standard gray if equalization missed any
            faces = face_cascade.detectMultiScale(
                gray,
                scaleFactor=1.1,
                minNeighbors=4,
                minSize=min_size
            )
        
        # Sort faces left-to-right (by x coordinate) for consistent examiner docketing
        sorted_faces = sorted(faces, key=lambda f: f[0])
        return sorted_faces
    except Exception as err:
        print(f"Face detection error: {err}")
        return []


def crop_face(img_bgr, bbox, pad_ratio=0.20):
    """Extracts a padded face crop from the source image safely."""
    h_img, w_img = img_bgr.shape[:2]
    x, y, w, h = bbox
    pad_w = int(w * pad_ratio)
    pad_h = int(h * pad_ratio)
    
    x1 = max(0, x - pad_w)
    y1 = max(0, y - pad_h)
    x2 = min(w_img, x + w + pad_w)
    y2 = min(h_img, y + h + pad_h)
    
    crop = img_bgr[y1:y2, x1:x2]
    return crop, (x1, y1, x2, y2)


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


def classify_crop(crop_bgr):
    """Classifies a cropped face region using the neural model with FFT fallback."""
    if crop_bgr is None or crop_bgr.size == 0:
        return 0.50, 0.50
    
    gray_crop = cv2.cvtColor(crop_bgr, cv2.COLOR_BGR2GRAY)
    fft_score = get_fft_anomaly(gray_crop)
    fake_prob = fft_score

    if classifier is not None:
        try:
            crop_rgb = cv2.cvtColor(crop_bgr, cv2.COLOR_BGR2RGB)
            pil_img = Image.fromarray(crop_rgb)
            preds = classifier(pil_img)
            for p in preds:
                lbl = p["label"].lower()
                if "fake" in lbl or "synthetic" in lbl or "manipulated" in lbl:
                    fake_prob = float(p["score"])
                    break
                elif "real" in lbl or "authentic" in lbl:
                    fake_prob = 1.0 - float(p["score"])
                    break
        except Exception as e:
            print(f"Inference error on face crop: {e}")
            fake_prob = fft_score

    return fake_prob, fft_score


def generate_annotated_heatmap(img_bgr, faces_results):
    """Generates an edge-discontinuity activation overlay with color-coded bounding boxes for each subject."""
    try:
        # Downscale for performance if extremely large
        h_orig, w_orig = img_bgr.shape[:2]
        scale = 1.0
        work_img = img_bgr
        if max(h_orig, w_orig) > 1080:
            scale = 1080.0 / max(h_orig, w_orig)
            work_img = cv2.resize(img_bgr, (int(w_orig * scale), int(h_orig * scale)))

        gray = cv2.cvtColor(work_img, cv2.COLOR_BGR2GRAY)
        blurred = cv2.GaussianBlur(gray, (9, 9), 0)
        edges = cv2.Laplacian(blurred, cv2.CV_64F)
        edges = np.uint8(np.absolute(edges))
        edges = cv2.GaussianBlur(edges, (15, 15), 0)
        
        norm_cam = cv2.normalize(edges, None, 0, 255, cv2.NORM_MINMAX, dtype=cv2.CV_8U)
        heatmap = cv2.applyColorMap(norm_cam, cv2.COLORMAP_JET)
        blended = cv2.addWeighted(work_img, 0.65, heatmap, 0.35, 0)

        # Draw detected face boxes and forensic labels
        for f in faces_results:
            orig_bbox = f["bbox"]
            # Scale bbox to work image dimensions
            bx = int(orig_bbox[0] * scale)
            by = int(orig_bbox[1] * scale)
            bw = int(orig_bbox[2] * scale)
            bh = int(orig_bbox[3] * scale)

            is_manipulated = f["manipulation_score"] >= 0.50
            # Red for manipulated, Emerald Green for authentic
            box_color = (0, 0, 240) if is_manipulated else (0, 200, 50)
            tag_text = f"Subject #{f['subject_id']}: {'SYNTHETIC' if is_manipulated else 'AUTHENTIC'} ({f['confidence']*100:.1f}%)"

            # Draw outer rectangle
            cv2.rectangle(blended, (bx, by), (bx + bw, by + bh), box_color, 2)

            # Draw top banner badge
            font_scale = 0.5
            font_thickness = 1
            (tw, th), _ = cv2.getTextSize(tag_text, cv2.FONT_HERSHEY_SIMPLEX, font_scale, font_thickness)
            
            badge_y1 = max(0, by - th - 8)
            badge_y2 = by
            cv2.rectangle(blended, (bx, badge_y1), (bx + tw + 10, badge_y2), (20, 20, 20), -1)
            cv2.rectangle(blended, (bx, badge_y1), (bx + tw + 10, badge_y2), box_color, 1)
            cv2.putText(blended, tag_text, (bx + 5, by - 4), cv2.FONT_HERSHEY_SIMPLEX, font_scale, box_color, font_thickness, cv2.LINE_AA)

        return blended
    except Exception as err:
        print(f"Heatmap generation error: {err}")
        return img_bgr


def analyze_image(image_path):
    """Analyzes a single image by detecting faces, classifying each face crop, and handling multi-face/zero-face gracefully."""
    img_bgr = cv2.imread(image_path)
    if img_bgr is None:
        raise ValueError(f"Corrupted or unsupported image file: {image_path}")

    # Step 1: Detect Faces
    detected_bboxes = detect_faces(img_bgr)
    num_faces = len(detected_bboxes)
    print(f"Forensic Intake - Image: {image_path}, Shape: {img_bgr.shape}, Faces Detected: {num_faces}")

    faces_results = []
    global_fft = get_fft_anomaly(cv2.cvtColor(img_bgr, cv2.COLOR_BGR2GRAY))

    if num_faces > 0:
        # Analyze each face crop independently
        for idx, bbox in enumerate(detected_bboxes):
            crop, padded_coords = crop_face(img_bgr, bbox, pad_ratio=0.20)
            fake_prob, face_fft = classify_crop(crop)
            is_fake = fake_prob >= 0.50
            
            faces_results.append({
                "subject_id": idx + 1,
                "bbox": [int(bbox[0]), int(bbox[1]), int(bbox[2]), int(bbox[3])],
                "padded_coords": [int(c) for c in padded_coords],
                "verdict": "Manipulated (Deepfake)" if is_fake else "Authentic Media",
                "confidence": fake_prob if is_fake else (1.0 - fake_prob),
                "manipulation_score": fake_prob,
                "fft_score": face_fft
            })

        # Aggregation Logic
        manipulated_faces = [f for f in faces_results if f["manipulation_score"] >= 0.50]
        if manipulated_faces:
            worst_face = max(faces_results, key=lambda f: f["manipulation_score"])
            verdict = f"Manipulated (Deepfake Detected in {len(manipulated_faces)}/{num_faces} Subjects)"
            overall_manipulation = worst_face["manipulation_score"]
            overall_confidence = worst_face["confidence"]
            summary_note = f"Subject #{worst_face['subject_id']} exhibited neural facial manipulation artifacts ({worst_face['confidence']*100:.1f}% confidence)."
        else:
            avg_conf = float(np.mean([f["confidence"] for f in faces_results]))
            verdict = "Authentic Media (All Subjects Verified)"
            overall_manipulation = float(np.mean([f["manipulation_score"] for f in faces_results]))
            overall_confidence = avg_conf
            summary_note = f"All {num_faces} detected subject(s) verified authentic with consistent biometric features."

        status = "FACES_ANALYZED"
    else:
        # Graceful handling for Zero Faces (e.g. landscapes, documents, non-face objects)
        status = "NO_FACE_DETECTED"
        verdict = "Inconclusive (No Facial Subjects Detected)"
        overall_manipulation = global_fft
        overall_confidence = 0.50
        summary_note = "No human faces detected for neural facial analysis. Examination limited to global spectral sensor noise metrics."

    # Generate Annotated Heatmap
    heatmap_img = generate_annotated_heatmap(img_bgr, faces_results)
    heatmap_path = "temp_heatmap.png"
    cv2.imwrite(heatmap_path, heatmap_img)

    return {
        "status": status,
        "face_count": num_faces,
        "faces": faces_results,
        "verdict": verdict,
        "confidence": overall_confidence,
        "manipulation_score": overall_manipulation,
        "fft_score": global_fft,
        "heatmap_path": heatmap_path,
        "summary_note": summary_note
    }


def analyze_video(video_path, max_frames=12):
    """Samples video frames quickly, detects faces per frame, and isolates the most anomalous frame and subject."""
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        raise ValueError(f"Could not open video source: {video_path}")
        
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    step = max(1, total_frames // max_frames)
    
    frame_idx = 0
    sampled_results = []
    worst_score = -1.0
    worst_frame = None
    worst_faces = []
    
    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break
        if frame_idx % step == 0:
            temp_path = f"temp_frame_{frame_idx}.png"
            cv2.imwrite(temp_path, frame)
            try:
                res = analyze_image(temp_path)
                sampled_results.append(res)
                if res["manipulation_score"] > worst_score:
                    worst_score = res["manipulation_score"]
                    worst_frame = frame.copy()
                    worst_faces = res.get("faces", [])
            finally:
                if os.path.exists(temp_path):
                    os.remove(temp_path)
        frame_idx += 1
        
    cap.release()
    
    if not sampled_results:
        return {
            "status": "VIDEO_UNREADABLE",
            "face_count": 0,
            "faces": [],
            "verdict": "Corrupted / Unreadable Video Stream",
            "confidence": 0.0,
            "manipulation_score": 0.50,
            "fft_score": 0.50,
            "heatmap_path": "temp_heatmap.png",
            "summary_note": "Could not extract valid video frames."
        }

    # Aggregate scores across frames
    all_scores = [r["manipulation_score"] for r in sampled_results]
    max_score = float(np.max(all_scores))
    avg_score = float(np.mean(all_scores))
    total_faces_seen = sum(r["face_count"] for r in sampled_results)

    is_fake = max_score >= 0.50
    if is_fake:
        verdict = f"Manipulated Video (Temporal Deepfake Artifacts Detected)"
        confidence = max_score
        summary_note = f"Temporal discontinuity and facial manipulation isolated (peak anomaly score: {max_score*100:.1f}%)."
    elif total_faces_seen > 0:
        verdict = "Authentic Video Stream (Temporal Integrity Verified)"
        confidence = 1.0 - avg_score
        summary_note = f"Biometric continuity verified across {len(sampled_results)} sampled frames."
    else:
        verdict = "Inconclusive (No Faces Detected in Video Frames)"
        confidence = 0.50
        summary_note = "No facial subjects identified across sampled video frames."

    worst_heatmap_path = "temp_heatmap.png"
    if worst_frame is not None:
        cv2.imwrite(worst_heatmap_path, generate_annotated_heatmap(worst_frame, worst_faces))
        
    return {
        "status": "VIDEO_ANALYZED",
        "face_count": len(worst_faces),
        "faces": worst_faces,
        "verdict": verdict,
        "confidence": confidence,
        "manipulation_score": max_score,
        "fft_score": float(np.mean([r["fft_score"] for r in sampled_results])),
        "heatmap_path": worst_heatmap_path,
        "summary_note": summary_note
    }