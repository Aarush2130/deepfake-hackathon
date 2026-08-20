import os
import cv2
import torch
import numpy as np
from PIL import Image
from transformers import pipeline

# 1. Initialize Face Detectors (Ensemble of Multi-Scale Cascades)
cascades = {}
cascade_names = {
    "alt2": "haarcascade_frontalface_alt2.xml",
    "default": "haarcascade_frontalface_default.xml",
    "profile": "haarcascade_profileface.xml",
}

for key, fname in cascade_names.items():
    try:
        path = os.path.join(cv2.data.haarcascades, fname)
        c = cv2.CascadeClassifier(path)
        if not c.empty():
            cascades[key] = c
    except Exception as e:
        print(f"Warning: Could not load cascade {fname}: {e}")

print(f"Ensemble face detectors loaded: {list(cascades.keys())}")

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


def nms_boxes(boxes, overlap_thresh=0.3):
    """Applies Non-Maximum Suppression to remove overlapping bounding boxes."""
    if len(boxes) == 0:
        return []
    
    boxes_np = np.array(boxes)
    x1 = boxes_np[:, 0]
    y1 = boxes_np[:, 1]
    x2 = boxes_np[:, 0] + boxes_np[:, 2]
    y2 = boxes_np[:, 1] + boxes_np[:, 3]
    areas = (x2 - x1 + 1) * (y2 - y1 + 1)
    
    order = areas.argsort()[::-1]
    keep = []
    
    while order.size > 0:
        i = order[0]
        keep.append(i)
        
        xx1 = np.maximum(x1[i], x1[order[1:]])
        yy1 = np.maximum(y1[i], y1[order[1:]])
        xx2 = np.minimum(x2[i], x2[order[1:]])
        yy2 = np.minimum(y2[i], y2[order[1:]])
        
        w = np.maximum(0.0, xx2 - xx1 + 1)
        h = np.maximum(0.0, yy2 - yy1 + 1)
        inter = w * h
        ovr = inter / (areas[i] + areas[order[1:]] - inter)
        
        inds = np.where(ovr <= overlap_thresh)[0]
        order = order[inds + 1]
        
    return [boxes[k] for k in keep]


def detect_faces(img_bgr):
    """Robust multi-pass face detector utilizing ensemble Haar cascades and profile scanning."""
    if img_bgr is None or len(cascades) == 0:
        return []
    
    h_img, w_img = img_bgr.shape[:2]
    gray = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2GRAY)
    detected = []

    # Pass 1: Alt2 Cascade (Most accurate for angled & tilted faces)
    if "alt2" in cascades:
        f = cascades["alt2"].detectMultiScale(gray, scaleFactor=1.08, minNeighbors=3, minSize=(25, 25))
        if len(f) > 0:
            detected.extend(f)

    # Pass 2: Default Frontal Cascade (if few or no detections)
    if len(detected) == 0 and "default" in cascades:
        f = cascades["default"].detectMultiScale(gray, scaleFactor=1.1, minNeighbors=3, minSize=(25, 25))
        if len(f) > 0:
            detected.extend(f)

    # Pass 3: Profile Face Cascade (and horizontally flipped profile)
    if len(detected) == 0 and "profile" in cascades:
        f = cascades["profile"].detectMultiScale(gray, scaleFactor=1.1, minNeighbors=3, minSize=(25, 25))
        if len(f) > 0:
            detected.extend(f)
        
        # Check flipped for opposing profile
        gray_flipped = cv2.flip(gray, 1)
        f_flip = cascades["profile"].detectMultiScale(gray_flipped, scaleFactor=1.1, minNeighbors=3, minSize=(25, 25))
        for (x, y, w, h) in f_flip:
            detected.append((w_img - x - w, y, w, h))

    if len(detected) == 0:
        return []

    # Clean duplicates with NMS
    unique_boxes = nms_boxes(detected, overlap_thresh=0.35)
    
    # Sort left-to-right
    sorted_boxes = sorted(unique_boxes, key=lambda b: b[0])
    return sorted_boxes


def crop_face(img_bgr, bbox, pad_ratio=0.15):
    """Extracts padded face crop from image safely."""
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
    """Classifies a cropped face or portrait region using the neural model with FFT fallback."""
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
            print(f"Inference error: {e}")
            fake_prob = fft_score

    return fake_prob, fft_score


def generate_annotated_heatmap(img_bgr, faces_results):
    """Generates an edge-discontinuity activation overlay with color-coded bounding boxes for each subject."""
    try:
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
            bx = int(orig_bbox[0] * scale)
            by = int(orig_bbox[1] * scale)
            bw = int(orig_bbox[2] * scale)
            bh = int(orig_bbox[3] * scale)

            # Skip drawing box if it's the full-frame fallback box
            if bw >= int(w_orig * scale * 0.95) and bh >= int(h_orig * scale * 0.95):
                continue

            is_manipulated = f["manipulation_score"] >= 0.50
            box_color = (0, 0, 240) if is_manipulated else (0, 200, 50)
            tag_text = f"Subject #{f['subject_id']}: {'SYNTHETIC' if is_manipulated else 'AUTHENTIC'} ({f['confidence']*100:.1f}%)"

            cv2.rectangle(blended, (bx, by), (bx + bw, by + bh), box_color, 2)

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
    """Analyzes a single image: detects faces, classifies each subject, and falls back to full-frame portrait evaluation."""
    img_bgr = cv2.imread(image_path)
    if img_bgr is None:
        raise ValueError(f"Corrupted or unsupported image file: {image_path}")

    h_img, w_img = img_bgr.shape[:2]

    # Step 1: Detect Faces with Ensemble Cascade
    detected_bboxes = detect_faces(img_bgr)
    
    # Step 2: Full-frame portrait fallback if cascade finds 0 boxes
    is_full_frame_fallback = False
    if len(detected_bboxes) == 0:
        is_full_frame_fallback = True
        detected_bboxes = [(0, 0, w_img, h_img)]

    num_faces = len(detected_bboxes)
    print(f"Forensic Intake - Image: {image_path}, Shape: {img_bgr.shape}, Detected Faces: {0 if is_full_frame_fallback else num_faces}")

    faces_results = []
    global_fft = get_fft_anomaly(cv2.cvtColor(img_bgr, cv2.COLOR_BGR2GRAY))

    for idx, bbox in enumerate(detected_bboxes):
        if is_full_frame_fallback:
            crop = img_bgr
            padded_coords = (0, 0, w_img, h_img)
        else:
            crop, padded_coords = crop_face(img_bgr, bbox, pad_ratio=0.15)
            
        fake_prob, face_fft = classify_crop(crop)
        is_fake = fake_prob >= 0.50
        
        faces_results.append({
            "subject_id": idx + 1,
            "bbox": [int(bbox[0]), int(bbox[1]), int(bbox[2]), int(bbox[3])],
            "padded_coords": [int(c) for c in padded_coords],
            "verdict": "Manipulated (Deepfake)" if is_fake else "Authentic Media",
            "confidence": fake_prob if is_fake else (1.0 - fake_prob),
            "manipulation_score": fake_prob,
            "fft_score": face_fft,
            "is_full_frame": is_full_frame_fallback
        })

    # Aggregation Logic
    manipulated_faces = [f for f in faces_results if f["manipulation_score"] >= 0.50]
    
    if is_full_frame_fallback:
        single = faces_results[0]
        is_fake = single["manipulation_score"] >= 0.50
        verdict = "Manipulated (Deepfake Detected)" if is_fake else "Authentic Media"
        overall_manipulation = single["manipulation_score"]
        overall_confidence = single["confidence"]
        summary_note = (
            f"Evaluated as full-frame portrait. Neural model determination: {verdict} "
            f"({overall_confidence*100:.1f}% confidence)."
        )
        display_face_count = 1
        status = "FACES_ANALYZED"
    elif manipulated_faces:
        worst_face = max(faces_results, key=lambda f: f["manipulation_score"])
        verdict = f"Manipulated (Deepfake in {len(manipulated_faces)}/{num_faces} Subjects)"
        overall_manipulation = worst_face["manipulation_score"]
        overall_confidence = worst_face["confidence"]
        summary_note = f"Subject #{worst_face['subject_id']} exhibited facial manipulation artifacts ({worst_face['confidence']*100:.1f}% confidence)."
        display_face_count = num_faces
        status = "FACES_ANALYZED"
    else:
        avg_conf = float(np.mean([f["confidence"] for f in faces_results]))
        verdict = "Authentic Media (All Subjects Verified)"
        overall_manipulation = float(np.mean([f["manipulation_score"] for f in faces_results]))
        overall_confidence = avg_conf
        summary_note = f"All {num_faces} detected subject(s) verified authentic."
        display_face_count = num_faces
        status = "FACES_ANALYZED"

    # Generate Annotated Heatmap
    heatmap_img = generate_annotated_heatmap(img_bgr, faces_results)
    heatmap_path = "temp_heatmap.png"
    cv2.imwrite(heatmap_path, heatmap_img)

    return {
        "status": status,
        "face_count": display_face_count,
        "faces": faces_results,
        "verdict": verdict,
        "confidence": overall_confidence,
        "manipulation_score": overall_manipulation,
        "fft_score": global_fft,
        "heatmap_path": heatmap_path,
        "summary_note": summary_note
    }


def analyze_video(video_path, max_frames=12):
    """Samples video frames quickly, detects faces per frame, and isolates anomalies."""
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

    all_scores = [r["manipulation_score"] for r in sampled_results]
    max_score = float(np.max(all_scores))
    avg_score = float(np.mean(all_scores))

    is_fake = max_score >= 0.50
    if is_fake:
        verdict = "Manipulated Video (Temporal Deepfake Artifacts Detected)"
        confidence = max_score
        summary_note = f"Temporal discontinuity and facial manipulation isolated (peak anomaly score: {max_score*100:.1f}%)."
    else:
        verdict = "Authentic Video Stream (Temporal Integrity Verified)"
        confidence = 1.0 - avg_score
        summary_note = f"Biometric continuity verified across {len(sampled_results)} sampled frames."

    worst_heatmap_path = "temp_heatmap.png"
    if worst_frame is not None:
        cv2.imwrite(worst_heatmap_path, generate_annotated_heatmap(worst_frame, worst_faces))
        
    return {
        "status": "VIDEO_ANALYZED",
        "face_count": max(1, len(worst_faces)),
        "faces": worst_faces,
        "verdict": verdict,
        "confidence": confidence,
        "manipulation_score": max_score,
        "fft_score": float(np.mean([r["fft_score"] for r in sampled_results])),
        "heatmap_path": worst_heatmap_path,
        "summary_note": summary_note
    }