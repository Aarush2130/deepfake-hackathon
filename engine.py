import base64
import os
import cv2
import torch
import torch.nn as nn
import torchvision.models as models
import torchvision.transforms as transforms
import numpy as np
from PIL import Image

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model_path = "best_model-v3.pt"

# 1. Initialize Cascades for Multi-Face Extraction
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

# 2. Load Fine-Tuned EfficientNet-B0 Model
def load_custom_model():
    try:
        net = models.efficientnet_b0(weights=None)
        in_features = net.classifier[1].in_features
        net.classifier[1] = nn.Linear(in_features, 2)
        
        if os.path.exists(model_path):
            state_dict = torch.load(model_path, map_location=device, weights_only=False)
            net.load_state_dict(state_dict)
            net.to(device)
            net.eval()
            print(f"Loaded fine-tuned EfficientNet-B0 from {model_path} on {device}")
            return net
        else:
            print(f"Warning: {model_path} not found. Operating in spectral fallback mode.")
            return None
    except Exception as err:
        print(f"Error loading custom PyTorch model: {err}")
        return None

classifier = load_custom_model()

# Preprocessing transforms matching EfficientNet training
img_transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
])

def nms_boxes(boxes, overlap_thresh=0.3):
    if len(boxes) == 0:
        return []
    boxes_np = np.array(boxes)
    x1, y1 = boxes_np[:, 0], boxes_np[:, 1]
    x2, y2 = boxes_np[:, 0] + boxes_np[:, 2], boxes_np[:, 1] + boxes_np[:, 3]
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
    if img_bgr is None:
        return []
    h_orig, w_orig = img_bgr.shape[:2]
    
    scales = [1.0]
    if max(h_orig, w_orig) > 1000:
        scales.append(1000.0 / max(h_orig, w_orig))
    elif min(h_orig, w_orig) < 300:
        scales.append(2.0)

    detected = []
    for sc in scales:
        if sc == 1.0:
            work = img_bgr
        else:
            work = cv2.resize(img_bgr, (int(w_orig * sc), int(h_orig * sc)))
        
        gray = cv2.cvtColor(work, cv2.COLOR_BGR2GRAY)
        gray_eq = cv2.equalizeHist(gray)
        
        for g in [gray, gray_eq]:
            if "alt2" in cascades:
                f = cascades["alt2"].detectMultiScale(g, scaleFactor=1.08, minNeighbors=3, minSize=(25, 25))
                for (x, y, w, h) in f:
                    detected.append((int(x / sc), int(y / sc), int(w / sc), int(h / sc)))
            
            if len(detected) == 0 and "default" in cascades:
                f = cascades["default"].detectMultiScale(g, scaleFactor=1.1, minNeighbors=3, minSize=(25, 25))
                for (x, y, w, h) in f:
                    detected.append((int(x / sc), int(y / sc), int(w / sc), int(h / sc)))
            
            if len(detected) == 0 and "profile" in cascades:
                f = cascades["profile"].detectMultiScale(g, scaleFactor=1.1, minNeighbors=3, minSize=(25, 25))
                for (x, y, w, h) in f:
                    detected.append((int(x / sc), int(y / sc), int(w / sc), int(h / sc)))
                
                g_flip = cv2.flip(g, 1)
                f_flip = cascades["profile"].detectMultiScale(g_flip, scaleFactor=1.1, minNeighbors=3, minSize=(25, 25))
                for (x, y, w, h) in f_flip:
                    detected.append((int((g.shape[1] - x - w) / sc), int(y / sc), int(w / sc), int(h / sc)))

    if len(detected) == 0:
        return []

    return sorted(nms_boxes(detected, overlap_thresh=0.35), key=lambda b: b[0])

def crop_face(img_bgr, bbox, pad_ratio=0.20):
    h_img, w_img = img_bgr.shape[:2]
    x, y, w, h = bbox
    pad_w, pad_h = int(w * pad_ratio), int(h * pad_ratio)
    x1, y1 = max(0, x - pad_w), max(0, y - pad_h)
    x2, y2 = min(w_img, x + w + pad_w), min(h_img, y + h + pad_h)
    return img_bgr[y1:y2, x1:x2], (x1, y1, x2, y2)

def get_fft_anomaly(gray_img):
    try:
        standard_gray = cv2.resize(gray_img, (256, 256))
        f = np.fft.fft2(standard_gray)
        fshift = np.fft.fftshift(f)
        magnitude = np.log(np.abs(fshift) + 1e-6)
        h, w = magnitude.shape
        cy, cx = h // 2, w // 2
        y, x = np.ogrid[:h, :w]
        dist = np.sqrt((x - cx)**2 + (y - cy)**2)
        hf_mean = magnitude[dist > 90].mean()
        mf_mean = magnitude[(dist >= 30) & (dist <= 90)].mean()
        ratio = float(hf_mean / (mf_mean + 1e-6))
        return float(np.clip((ratio - 0.68) * 3.5, 0.08, 0.92))
    except Exception:
        return 0.20

def get_ela_anomaly(crop_bgr, quality=90):
    """
    Error Level Analysis (ELA): Computes compression artifact inconsistency.
    Spliced or AI-synthesized facial regions exhibit distinct compression error delta.
    """
    if crop_bgr is None or crop_bgr.size == 0:
        return 0.20
    try:
        encode_param = [int(cv2.IMWRITE_JPEG_QUALITY), quality]
        _, enc_img = cv2.imencode('.jpg', crop_bgr, encode_param)
        decomp = cv2.imdecode(enc_img, cv2.IMREAD_COLOR)
        
        diff = cv2.absdiff(crop_bgr, decomp)
        diff_gray = cv2.cvtColor(diff, cv2.COLOR_BGR2GRAY)
        
        ela_std = float(np.std(diff_gray))
        ela_mean = float(np.mean(diff_gray))
        
        raw_score = (ela_std * 0.7 + ela_mean * 0.3) / 12.0
        return float(np.clip(raw_score, 0.05, 0.95))
    except Exception:
        return 0.20

def get_chroma_anomaly(crop_bgr):
    """
    Chrominance Inconsistency Profiler: Evaluates YCbCr & LAB color space deviations.
    Generative models often have unnatural chrominance standard deviations (Cb/Cr & a/b).
    """
    if crop_bgr is None or crop_bgr.size == 0:
        return 0.20
    try:
        ycrcb = cv2.cvtColor(crop_bgr, cv2.COLOR_BGR2YCrCb)
        _, cr, cb = cv2.split(ycrcb)
        
        lab = cv2.cvtColor(crop_bgr, cv2.COLOR_BGR2LAB)
        _, a, b = cv2.split(lab)
        
        cr_std, cb_std = float(np.std(cr)), float(np.std(cb))
        a_std, b_std = float(np.std(a)), float(np.std(b))
        
        chroma_disp = abs(cr_std - cb_std) + abs(a_std - b_std)
        score = float(np.clip((chroma_disp - 4.0) / 18.0, 0.10, 0.90))
        return score
    except Exception:
        return 0.20

def get_boundary_discontinuity(crop_bgr):
    """
    Perimeter Blending Seam Analysis: Compares gradient variance along the
    outer 15% boundary ring vs the inner 70% facial core.
    """
    if crop_bgr is None or crop_bgr.size == 0:
        return 0.20
    try:
        h, w = crop_bgr.shape[:2]
        if h < 30 or w < 30:
            return 0.20
        gray = cv2.cvtColor(crop_bgr, cv2.COLOR_BGR2GRAY)
        lap = cv2.Laplacian(gray, cv2.CV_64F)
        
        pad_h, pad_w = int(h * 0.15), int(w * 0.15)
        outer_mask = np.ones((h, w), dtype=bool)
        outer_mask[pad_h:h-pad_h, pad_w:w-pad_w] = False
        
        inner_lap = lap[pad_h:h-pad_h, pad_w:w-pad_w]
        outer_lap = lap[outer_mask]
        
        var_inner = float(np.var(inner_lap)) if inner_lap.size > 0 else 1.0
        var_outer = float(np.var(outer_lap)) if outer_lap.size > 0 else 1.0
        
        ratio = var_outer / (var_inner + 1e-5)
        anomaly = abs(ratio - 1.0)
        return float(np.clip(anomaly / 2.5, 0.10, 0.90))
    except Exception:
        return 0.20

def classify_crop(crop_bgr):
    """
    Multi-Modal Forensic Classification:
    Combines Dual-Scale Neural Inference (Context + Core), 2D-FFT Fourier Spectral Residuals,
    Error Level Analysis (ELA), Chrominance Deviations, and Boundary Seam Discontinuities.
    """
    if crop_bgr is None or crop_bgr.size == 0:
        return {
            "composite_prob": 0.15,
            "neural_prob": 0.15,
            "fft_score": 0.15,
            "ela_score": 0.15,
            "chroma_score": 0.15,
            "boundary_score": 0.15
        }
        
    gray_crop = cv2.cvtColor(crop_bgr, cv2.COLOR_BGR2GRAY)
    fft_score = get_fft_anomaly(gray_crop)
    ela_score = get_ela_anomaly(crop_bgr)
    chroma_score = get_chroma_anomaly(crop_bgr)
    boundary_score = get_boundary_discontinuity(crop_bgr)
    neural_prob = fft_score

    if classifier is not None:
        try:
            # 1. Context Outer Inference (Full Crop with Padding)
            crop_rgb = cv2.cvtColor(crop_bgr, cv2.COLOR_BGR2RGB)
            pil_img = Image.fromarray(crop_rgb)
            tensor_img = img_transform(pil_img).unsqueeze(0).to(device)
            
            # 2. Core Facial Feature Inference (Center 80% of Crop)
            h, w = crop_bgr.shape[:2]
            y1, y2 = int(h * 0.10), int(h * 0.90)
            x1, x2 = int(w * 0.10), int(w * 0.90)
            inner_crop = crop_bgr[y1:y2, x1:x2] if (y2 > y1 and x2 > x1) else crop_bgr
            inner_rgb = cv2.cvtColor(inner_crop, cv2.COLOR_BGR2RGB)
            inner_tensor = img_transform(Image.fromarray(inner_rgb)).unsqueeze(0).to(device)
            
            with torch.no_grad():
                logits_outer = classifier(tensor_img)
                probs_outer = torch.softmax(logits_outer / 1.5, dim=1).cpu().numpy()[0]
                
                logits_inner = classifier(inner_tensor)
                probs_inner = torch.softmax(logits_inner / 1.5, dim=1).cpu().numpy()[0]
                
                p_out = float(probs_outer[0])  # Index 0 = Synthetic
                p_in = float(probs_inner[0])
                
                # Weighted fusion of outer perimeter + inner features
                neural_prob = 0.55 * p_out + 0.45 * p_in
        except Exception as e:
            print(f"Custom Model Inference Error: {e}")
            neural_prob = fft_score

    # Multi-Modal Weighted Fusion
    if classifier is not None:
        composite_prob = (
            0.55 * neural_prob +
            0.18 * fft_score +
            0.14 * ela_score +
            0.08 * chroma_score +
            0.05 * boundary_score
        )
    else:
        composite_prob = (
            0.40 * fft_score +
            0.30 * ela_score +
            0.20 * chroma_score +
            0.10 * boundary_score
        )

    composite_prob = float(np.clip(composite_prob, 0.02, 0.98))

    return {
        "composite_prob": composite_prob,
        "neural_prob": neural_prob,
        "fft_score": fft_score,
        "ela_score": ela_score,
        "chroma_score": chroma_score,
        "boundary_score": boundary_score
    }

def generate_annotated_heatmap(img_bgr, faces_results):
    try:
        h_orig, w_orig = img_bgr.shape[:2]
        scale = 1.0
        work_img = img_bgr
        if max(h_orig, w_orig) > 1080:
            scale = 1080.0 / max(h_orig, w_orig)
            work_img = cv2.resize(img_bgr, (int(w_orig * scale), int(h_orig * scale)))

        gray = cv2.cvtColor(work_img, cv2.COLOR_BGR2GRAY)
        blurred = cv2.GaussianBlur(gray, (9, 9), 0)
        edges = np.uint8(np.absolute(cv2.Laplacian(blurred, cv2.CV_64F)))
        edges = cv2.GaussianBlur(edges, (15, 15), 0)
        
        norm_cam = cv2.normalize(edges, None, 0, 255, cv2.NORM_MINMAX, dtype=cv2.CV_8U)
        heatmap = cv2.applyColorMap(norm_cam, cv2.COLORMAP_JET)
        blended = cv2.addWeighted(work_img, 0.70, heatmap, 0.30, 0)

        for f in faces_results:
            orig_bbox = f["bbox"]
            bx, by = int(orig_bbox[0] * scale), int(orig_bbox[1] * scale)
            bw, bh = int(orig_bbox[2] * scale), int(orig_bbox[3] * scale)

            if f.get("is_full_frame", False):
                continue

            is_manipulated = f["manipulation_score"] >= 0.50
            box_color = (0, 0, 240) if is_manipulated else (0, 200, 50)
            tag_text = f"Subject #{f['subject_id']}: {'SYNTHETIC' if is_manipulated else 'AUTHENTIC'} ({f['confidence']*100:.1f}%)"

            cv2.rectangle(blended, (bx, by), (bx + bw, by + bh), box_color, 2)
            (tw, th), _ = cv2.getTextSize(tag_text, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 1)
            cv2.rectangle(blended, (bx, max(0, by - th - 8)), (bx + tw + 10, by), (20, 20, 20), -1)
            cv2.rectangle(blended, (bx, max(0, by - th - 8)), (bx + tw + 10, by), box_color, 1)
            cv2.putText(blended, tag_text, (bx + 5, by - 4), cv2.FONT_HERSHEY_SIMPLEX, 0.5, box_color, 1, cv2.LINE_AA)

        return blended
    except Exception as err:
        print(f"Heatmap error: {err}")
        return img_bgr

def generate_fft_spectrum_image(img_bgr):
    """Generates 2D-FFT Fourier magnitude spectrum with forensic concentric frequency rings."""
    try:
        gray = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2GRAY)
        standard_gray = cv2.resize(gray, (512, 512))
        f = np.fft.fft2(standard_gray)
        fshift = np.fft.fftshift(f)
        magnitude = np.log(np.abs(fshift) + 1.0)
        norm_mag = cv2.normalize(magnitude, None, 0, 255, cv2.NORM_MINMAX, dtype=cv2.CV_8U)
        color_mag = cv2.applyColorMap(norm_mag, cv2.COLORMAP_MAGMA)
        
        # Add HUD grid and concentric frequency rings
        h, w = color_mag.shape[:2]
        cv2.circle(color_mag, (w // 2, h // 2), 60, (0, 220, 255), 1, cv2.LINE_AA)
        cv2.circle(color_mag, (w // 2, h // 2), 120, (0, 220, 255), 1, cv2.LINE_AA)
        cv2.circle(color_mag, (w // 2, h // 2), 180, (0, 220, 255), 1, cv2.LINE_AA)
        cv2.line(color_mag, (w // 2, 0), (w // 2, h), (0, 220, 255), 1, cv2.LINE_AA)
        cv2.line(color_mag, (0, h // 2), (w, h // 2), (0, 220, 255), 1, cv2.LINE_AA)
        cv2.putText(color_mag, "2D-FFT MAGNITUDE SPECTRUM", (15, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 1, cv2.LINE_AA)
        cv2.putText(color_mag, "CENTER: DC | PERIPHERY: HIGH FREQ", (15, h - 15), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (0, 220, 255), 1, cv2.LINE_AA)

        fft_path = "temp_fft_spectrum.png"
        cv2.imwrite(fft_path, color_mag)
        return fft_path
    except Exception as e:
        print(f"FFT Spectrum generation error: {e}")
        return None

def analyze_image(image_path):
    img_bgr = cv2.imread(image_path)
    if img_bgr is None:
        raise ValueError(f"Corrupted image file: {image_path}")

    h_img, w_img = img_bgr.shape[:2]
    detected_bboxes = detect_faces(img_bgr)
    
    is_full_frame_fallback = False
    if len(detected_bboxes) == 0:
        is_full_frame_fallback = True
        detected_bboxes = [(0, 0, w_img, h_img)]

    num_faces = len(detected_bboxes)
    faces_results = []
    global_fft = get_fft_anomaly(cv2.cvtColor(img_bgr, cv2.COLOR_BGR2GRAY))
    global_sharpness = float(cv2.Laplacian(cv2.cvtColor(img_bgr, cv2.COLOR_BGR2GRAY), cv2.CV_64F).var())

    for idx, bbox in enumerate(detected_bboxes):
        crop, padded_coords = (img_bgr, (0, 0, w_img, h_img)) if is_full_frame_fallback else crop_face(img_bgr, bbox, pad_ratio=0.20)
        metrics = classify_crop(crop)
        fake_prob = metrics["composite_prob"]
        neural_score = metrics["neural_prob"]
        face_fft = metrics["fft_score"]
        ela_score = metrics["ela_score"]
        chroma_score = metrics["chroma_score"]
        boundary_score = metrics["boundary_score"]
        is_fake = fake_prob >= 0.50
        
        # Calculate face crop sharpness and encode crop thumbnail to base64
        face_sharpness = float(cv2.Laplacian(cv2.cvtColor(crop, cv2.COLOR_BGR2GRAY), cv2.CV_64F).var()) if crop.size > 0 else 0.0
        crop_thumb = cv2.resize(crop, (160, 160)) if crop.size > 0 else crop
        _, buf = cv2.imencode('.png', crop_thumb)
        crop_b64 = base64.b64encode(buf).decode('utf-8')

        conf = fake_prob if is_fake else (1.0 - fake_prob)
        faces_results.append({
            "subject_id": idx + 1,
            "bbox": [int(bbox[0]), int(bbox[1]), int(bbox[2]), int(bbox[3])],
            "padded_coords": [int(c) for c in padded_coords],
            "verdict": "Manipulated (Deepfake)" if is_fake else "Authentic Media",
            "confidence": float(conf),
            "manipulation_score": float(fake_prob),
            "neural_score": float(neural_score),
            "fft_score": float(face_fft),
            "ela_score": float(ela_score),
            "chroma_score": float(chroma_score),
            "boundary_score": float(boundary_score),
            "sharpness": float(face_sharpness),
            "crop_b64": crop_b64,
            "is_full_frame": is_full_frame_fallback,
            "low_resolution": bool(bbox[2] < 80 or bbox[3] < 80)
        })

    manipulated_faces = [f for f in faces_results if f["manipulation_score"] >= 0.50]
    
    if is_full_frame_fallback:
        single = faces_results[0]
        is_fake = single["manipulation_score"] >= 0.50
        verdict = "Manipulated (Deepfake Detected)" if is_fake else "Authentic Media"
        overall_manipulation = single["manipulation_score"]
        overall_confidence = float(np.clip(single["confidence"], 0.55, 0.94))
        summary_note = f"Evaluated as full-frame portrait: {verdict} ({overall_confidence*100:.1f}% confidence)."
        display_face_count = 1
    elif manipulated_faces:
        worst_face = max(faces_results, key=lambda f: f["manipulation_score"])
        verdict = f"Manipulated (Deepfake in {len(manipulated_faces)}/{num_faces} Subjects)"
        overall_manipulation = worst_face["manipulation_score"]
        overall_confidence = float(worst_face["confidence"])
        summary_note = f"Subject #{worst_face['subject_id']} flagged as synthetic ({worst_face['confidence']*100:.1f}% confidence)."
        display_face_count = num_faces
    else:
        avg_conf = float(np.mean([f["confidence"] for f in faces_results]))
        verdict = "Authentic Media (All Subjects Verified)"
        overall_manipulation = float(np.mean([f["manipulation_score"] for f in faces_results]))
        overall_confidence = avg_conf
        summary_note = f"All {num_faces} detected subject(s) verified authentic."
        display_face_count = num_faces

    heatmap_path = "temp_heatmap.png"
    cv2.imwrite(heatmap_path, generate_annotated_heatmap(img_bgr, faces_results))
    fft_spectrum_path = generate_fft_spectrum_image(img_bgr)

    global_ela = float(np.mean([f["ela_score"] for f in faces_results])) if faces_results else 0.20
    global_chroma = float(np.mean([f["chroma_score"] for f in faces_results])) if faces_results else 0.20
    global_boundary = float(np.mean([f["boundary_score"] for f in faces_results])) if faces_results else 0.20
    global_neural = float(np.mean([f["neural_score"] for f in faces_results])) if faces_results else 0.20

    return {
        "status": "FACES_ANALYZED",
        "face_count": display_face_count,
        "faces": faces_results,
        "verdict": verdict,
        "confidence": overall_confidence,
        "manipulation_score": overall_manipulation,
        "neural_score": global_neural,
        "fft_score": global_fft,
        "ela_score": global_ela,
        "chroma_score": global_chroma,
        "boundary_score": global_boundary,
        "sharpness": global_sharpness,
        "heatmap_path": heatmap_path,
        "fft_spectrum_path": fft_spectrum_path,
        "summary_note": summary_note
    }

def analyze_video(video_path, max_frames=10):
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        raise ValueError(f"Could not open video file: {video_path}")
        
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    step = max(1, total_frames // max_frames)
    frame_idx, scores, worst_score, worst_frame, worst_faces = 0, [], -1.0, None, []
    
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
                    worst_frame = frame.copy()
                    worst_faces = res.get("faces", [])
            finally:
                if os.path.exists(temp_path):
                    os.remove(temp_path)
        frame_idx += 1
    cap.release()

    max_score = float(np.max(scores)) if scores else 0.50
    avg_score = float(np.mean(scores)) if scores else 0.50
    is_fake = max_score >= 0.50

    worst_heatmap_path = "temp_heatmap.png"
    fft_spectrum_path = None
    if worst_frame is not None:
        cv2.imwrite(worst_heatmap_path, generate_annotated_heatmap(worst_frame, worst_faces))
        fft_spectrum_path = generate_fft_spectrum_image(worst_frame)
        
    return {
        "status": "VIDEO_ANALYZED",
        "face_count": max(1, len(worst_faces)),
        "faces": worst_faces,
        "verdict": "Manipulated Video Track" if is_fake else "Authentic Video Track",
        "confidence": max_score if is_fake else (1.0 - avg_score),
        "manipulation_score": max_score,
        "fft_score": avg_score,
        "sharpness": 120.0,
        "heatmap_path": worst_heatmap_path,
        "fft_spectrum_path": fft_spectrum_path,
        "summary_note": f"Analyzed {len(scores)} sampled frames across video timeline."
    }
class EngineStatus(dict):
    """Flexible status container supporting both string display and dictionary lookup."""
    def __str__(self):
        return self.get("name", "Fine-Tuned EfficientNet-B0 (CUDA)")

def get_engine_status():
    """Returns runtime model telemetry for the UI sidebar."""
    is_cuda = torch.cuda.is_available() and device.type == "cuda"
    dev_name = "CUDA (RTX 4060)" if is_cuda else "CPU"
    
    if classifier is not None:
        return EngineStatus({
            "name": f"Custom EfficientNet-B0 ({dev_name})",
            "model_name": f"Custom EfficientNet-B0 ({dev_name})",
            "backend": "best_model-v3.pt",
            "badge": f"Custom EfficientNet-B0 ({dev_name})",
            "device": dev_name,
            "status": "ONLINE",
            "online": True,
            "is_neural": True
        })
    
    return EngineStatus({
        "name": "Spectral FFT Fallback (CPU)",
        "model_name": "2D Fast Fourier Energy Analysis",
        "backend": "Frequency Domain Residuals",
        "badge": "Spectral FFT Fallback (CPU)",
        "device": "CPU",
        "status": "FALLBACK",
        "online": False,
        "is_neural": False
    })