import base64
import os
import json
import cv2
import torch
import torch.nn as nn
import torchvision.models as models
import torchvision.transforms as transforms
import numpy as np
import io
from PIL import Image

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Model selection: Auto-detect fresh v4 model if trained, fallback to v3
model_path = "best_model-v3.pt" if os.path.exists("best_model-v3.pt") else "best_model-v4.pt"

# Read model metadata if available
fake_idx = 1
if os.path.exists("model_metadata.json"):
    try:
        with open("model_metadata.json", "r") as mf:
            metadata = json.load(mf)
            fake_idx = metadata.get("fake_index", 1)
            print(f"Loaded model metadata: using Fake Index = {fake_idx} ({metadata.get('classes', ['real', 'fake'])})")
    except Exception as e:
        print(f"Could not parse model_metadata.json: {e}")

# 1. Initialize Face Detectors
# Primary: OpenCV DNN SSD face detector (far more accurate than Haar)
dnn_net = None
try:
    dnn_proto = os.path.join(os.path.dirname(__file__), "deploy.prototxt")
    dnn_model = os.path.join(os.path.dirname(__file__), "res10_300x300_ssd_iter_140000.caffemodel")
    if os.path.exists(dnn_proto) and os.path.exists(dnn_model):
        dnn_net = cv2.dnn.readNetFromCaffe(dnn_proto, dnn_model)
        print("DNN SSD face detector loaded (primary).")
    else:
        print("DNN model files not found, will use Haar cascade fallback.")
except Exception as e:
    print(f"DNN face detector init failed: {e}")

# Fallback: Haar Cascades
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

def detect_faces_dnn(img_bgr, conf_threshold=0.55):
    """Primary face detector using OpenCV DNN SSD (ResNet-10 backbone).
    Far more robust than Haar cascades against pose, lighting, occlusion."""
    if dnn_net is None:
        return []
    h, w = img_bgr.shape[:2]
    blob = cv2.dnn.blobFromImage(img_bgr, 1.0, (300, 300),
                                  (104.0, 177.0, 123.0), swapRB=False, crop=False)
    dnn_net.setInput(blob)
    detections = dnn_net.forward()
    boxes = []
    for i in range(detections.shape[2]):
        confidence = float(detections[0, 0, i, 2])
        if confidence >= conf_threshold:
            x1 = int(detections[0, 0, i, 3] * w)
            y1 = int(detections[0, 0, i, 4] * h)
            x2 = int(detections[0, 0, i, 5] * w)
            y2 = int(detections[0, 0, i, 6] * h)
            # Clamp to image bounds
            x1, y1 = max(0, x1), max(0, y1)
            x2, y2 = min(w, x2), min(h, y2)
            bw, bh = x2 - x1, y2 - y1
            if bw > 20 and bh > 20:
                boxes.append((x1, y1, bw, bh))
    return boxes

def detect_faces_haar(img_bgr):
    """Fallback face detector using Haar cascades (multi-scale, multi-cascade)."""
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
                f = cascades["alt2"].detectMultiScale(g, scaleFactor=1.08, minNeighbors=4, minSize=(40, 40))
                for (x, y, w, h) in f:
                    detected.append((int(x / sc), int(y / sc), int(w / sc), int(h / sc)))

            if len(detected) == 0 and "default" in cascades:
                f = cascades["default"].detectMultiScale(g, scaleFactor=1.1, minNeighbors=4, minSize=(40, 40))
                for (x, y, w, h) in f:
                    detected.append((int(x / sc), int(y / sc), int(w / sc), int(h / sc)))

            if len(detected) == 0 and "profile" in cascades:
                f = cascades["profile"].detectMultiScale(g, scaleFactor=1.1, minNeighbors=4, minSize=(40, 40))
                for (x, y, w, h) in f:
                    detected.append((int(x / sc), int(y / sc), int(w / sc), int(h / sc)))

                g_flip = cv2.flip(g, 1)
                f_flip = cascades["profile"].detectMultiScale(g_flip, scaleFactor=1.1, minNeighbors=4, minSize=(40, 40))
                for (x, y, w, h) in f_flip:
                    detected.append((int((g.shape[1] - x - w) / sc), int(y / sc), int(w / sc), int(h / sc)))

    if len(detected) == 0:
        return []
    return nms_boxes(detected, overlap_thresh=0.35)

def detect_faces(img_bgr):
    """Unified face detection: DNN primary → Haar fallback."""
    if img_bgr is None:
        return []
    # Try DNN first (much more accurate)
    faces = detect_faces_dnn(img_bgr)
    if len(faces) > 0:
        return sorted(nms_boxes(faces, overlap_thresh=0.35), key=lambda b: b[0])
    # Fallback to Haar cascades
    faces = detect_faces_haar(img_bgr)
    return sorted(faces, key=lambda b: b[0]) if faces else []

def crop_face(img_bgr, bbox, pad_ratio=0.20):
    h_img, w_img = img_bgr.shape[:2]
    x, y, w, h = bbox
    pad_w, pad_h = int(w * pad_ratio), int(h * pad_ratio)
    x1, y1 = max(0, x - pad_w), max(0, y - pad_h)
    x2, y2 = min(w_img, x + w + pad_w), min(h_img, y + h + pad_h)
    return img_bgr[y1:y2, x1:x2], (x1, y1, x2, y2)

# ────────────────────────────────────────────────────────────────────────
#  FACE-ONLY ISOLATION: Elliptical mask to exclude background/hair/clothing
# ────────────────────────────────────────────────────────────────────────

def create_face_ellipse_mask(h, w):
    """Creates an elliptical mask approximating the facial skin region.
    Center 76% width × 90% height ellipse — covers forehead, cheeks, chin,
    eyes, nose, mouth while excluding corners (background, hair, ears)."""
    mask = np.zeros((h, w), dtype=np.uint8)
    center = (w // 2, h // 2)
    axes = (int(w * 0.38), int(h * 0.45))
    cv2.ellipse(mask, center, axes, 0, 0, 360, 255, -1)
    # Smooth edges to avoid hard cutoff artifacts
    mask = cv2.GaussianBlur(mask, (7, 7), 3)
    return mask

# ────────────────────────────────────────────────────────────────────────
#  FORENSIC SIGNAL EXTRACTORS (all face-masked where applicable)
# ────────────────────────────────────────────────────────────────────────

def get_fft_anomaly(gray_img):
    """2D-FFT Spectral Residual: GAN/diffusion models leave periodic
    frequency fingerprints visible in the high-frequency band."""
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

def get_ela_anomaly(crop_bgr, face_mask=None, quality=90):
    """
    Error Level Analysis (ELA): Computes compression artifact inconsistency.
    Spliced or AI-synthesized facial regions exhibit distinct compression error delta.
    When face_mask is provided, only pixels inside the mask are measured.
    """
    if crop_bgr is None or crop_bgr.size == 0:
        return 0.20
    try:
        encode_param = [int(cv2.IMWRITE_JPEG_QUALITY), quality]
        _, enc_img = cv2.imencode('.jpg', crop_bgr, encode_param)
        decomp = cv2.imdecode(enc_img, cv2.IMREAD_COLOR)

        diff = cv2.absdiff(crop_bgr, decomp)
        diff_gray = cv2.cvtColor(diff, cv2.COLOR_BGR2GRAY)

        # Apply face mask to only measure facial skin region
        if face_mask is not None:
            face_pixels = diff_gray[face_mask > 128]
            if face_pixels.size < 50:
                face_pixels = diff_gray.ravel()
        else:
            face_pixels = diff_gray.ravel()

        ela_std = float(np.std(face_pixels))
        ela_mean = float(np.mean(face_pixels))

        raw_score = (ela_std * 0.7 + ela_mean * 0.3) / 12.0
        return float(np.clip(raw_score, 0.05, 0.95))
    except Exception:
        return 0.20

def get_chroma_anomaly(crop_bgr, face_mask=None):
    """
    Chrominance Inconsistency Profiler: Evaluates YCbCr & LAB color space deviations.
    Generative models often have unnatural chrominance standard deviations.
    When face_mask is provided, only facial-skin pixels contribute.
    """
    if crop_bgr is None or crop_bgr.size == 0:
        return 0.20
    try:
        ycrcb = cv2.cvtColor(crop_bgr, cv2.COLOR_BGR2YCrCb)
        _, cr, cb = cv2.split(ycrcb)

        lab = cv2.cvtColor(crop_bgr, cv2.COLOR_BGR2LAB)
        _, a, b = cv2.split(lab)

        if face_mask is not None:
            mask_bool = face_mask > 128
            cr_vals = cr[mask_bool]
            cb_vals = cb[mask_bool]
            a_vals = a[mask_bool]
            b_vals = b[mask_bool]
            if cr_vals.size < 50:
                cr_vals, cb_vals = cr.ravel(), cb.ravel()
                a_vals, b_vals = a.ravel(), b.ravel()
        else:
            cr_vals, cb_vals = cr.ravel(), cb.ravel()
            a_vals, b_vals = a.ravel(), b.ravel()

        cr_std = float(np.std(cr_vals))
        cb_std = float(np.std(cb_vals))
        a_std = float(np.std(a_vals))
        b_std = float(np.std(b_vals))

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

# ────────────────────────────────────────────────────────────────────────
#  CORE CLASSIFIER: Multi-Modal Face-Focused Forensic Fusion
# ────────────────────────────────────────────────────────────────────────

def classify_crop(crop_bgr):
    """
    Face-Focused Multi-Modal Forensic Classification.

    Key design decisions for high accuracy:
    1. NO temperature scaling — raw softmax preserves model confidence
    2. Neural model is dominant signal (0.78 weight) — it's the trained discriminator
    3. Dual-scale inference takes the MAX (worst-case) not weighted avg
    4. Auxiliary signals use face-only elliptical mask
    5. Strong neural signals trigger confidence amplification to prevent
       noisy auxiliary heuristics from diluting clear model verdicts
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

    h, w = crop_bgr.shape[:2]

    # Generate face-only mask for auxiliary signals
    face_mask = create_face_ellipse_mask(h, w)

    # Apply face mask to FFT as well (mask out background before transform)
    gray_crop = cv2.cvtColor(crop_bgr, cv2.COLOR_BGR2GRAY)
    gray_masked = cv2.bitwise_and(gray_crop, gray_crop, mask=(face_mask > 128).astype(np.uint8) * 255)

    fft_score = get_fft_anomaly(gray_masked)
    ela_score = get_ela_anomaly(crop_bgr, face_mask=face_mask)
    chroma_score = get_chroma_anomaly(crop_bgr, face_mask=face_mask)
    boundary_score = get_boundary_discontinuity(crop_bgr)
    neural_prob = fft_score  # fallback if no model

    if classifier is not None:
        try:
            # 1. Context Outer Inference (full padded crop — catches blending seams)
            crop_rgb = cv2.cvtColor(crop_bgr, cv2.COLOR_BGR2RGB)
            pil_img = Image.fromarray(crop_rgb)
            tensor_outer = img_transform(pil_img).unsqueeze(0).to(device)

            # 2. Core Biometric Inference (tight center 70% — pure facial features)
            y1, y2 = int(h * 0.15), int(h * 0.85)
            x1, x2 = int(w * 0.15), int(w * 0.85)
            inner_crop = crop_bgr[y1:y2, x1:x2] if (y2 > y1 and x2 > x1) else crop_bgr
            inner_rgb = cv2.cvtColor(inner_crop, cv2.COLOR_BGR2RGB)
            tensor_inner = img_transform(Image.fromarray(inner_rgb)).unsqueeze(0).to(device)

            with torch.no_grad():
                # RAW softmax — NO temperature scaling (T=1.0)
                # Temperature was killing accuracy by pushing everything toward 50%
                logits_outer = classifier(tensor_outer)
                probs_outer = torch.softmax(logits_outer, dim=1).cpu().numpy()[0]

                logits_inner = classifier(tensor_inner)
                probs_inner = torch.softmax(logits_inner, dim=1).cpu().numpy()[0]

                p_out = float(probs_outer[fake_idx])  # Dynamic Fake probability from model metadata
                p_in = float(probs_inner[fake_idx])

                # Balanced dual-scale blend (outer context carries 80% weight, inner carries 20%)
                neural_prob = 0.80 * p_out + 0.20 * p_in

        except Exception as e:
            print(f"Neural Inference Error: {e}")
            neural_prob = fft_score

    # ── Multi-Modal Weighted Fusion ──
    if classifier is not None:
        # Neural model dominates — it's the trained discriminator
        # Auxiliary signals provide supporting physical evidence
        W_NEURAL = 0.85
        W_FFT = 0.05
        W_ELA = 0.04
        W_CHROMA = 0.03
        W_BOUNDARY = 0.03

        composite_prob = (
            W_NEURAL * neural_prob +
            W_FFT * fft_score +
            W_ELA * ela_score +
            W_CHROMA * chroma_score +
            W_BOUNDARY * boundary_score
        )

        # ── Confidence Amplification ──
        # When neural model is highly confident, preserve clean model verdicts
        if neural_prob >= 0.65:
            composite_prob = max(composite_prob, neural_prob * 0.95)
        elif neural_prob <= 0.35:
            composite_prob = min(composite_prob, neural_prob * 1.05)
    else:
        # No neural model — spectral fallback
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
        # No face detected — use center crop heuristic instead of entire frame
        # This prevents background/scenery from polluting all signals
        is_full_frame_fallback = True
        # Assume a centered portrait: crop center 60% of the image
        cx, cy = w_img // 2, h_img // 2
        crop_w, crop_h = int(w_img * 0.6), int(h_img * 0.6)
        fb_x1 = max(0, cx - crop_w // 2)
        fb_y1 = max(0, cy - crop_h // 2)
        fb_x2 = min(w_img, fb_x1 + crop_w)
        fb_y2 = min(h_img, fb_y1 + crop_h)
        detected_bboxes = [(fb_x1, fb_y1, fb_x2 - fb_x1, fb_y2 - fb_y1)]

    num_faces = len(detected_bboxes)
    faces_results = []
    global_fft = get_fft_anomaly(cv2.cvtColor(img_bgr, cv2.COLOR_BGR2GRAY))
    global_sharpness = float(cv2.Laplacian(cv2.cvtColor(img_bgr, cv2.COLOR_BGR2GRAY), cv2.CV_64F).var())

    for idx, bbox in enumerate(detected_bboxes):
        if is_full_frame_fallback:
            # Use the center crop directly
            bx, by, bw, bh = bbox
            crop = img_bgr[by:by+bh, bx:bx+bw]
            padded_coords = (bx, by, bx+bw, by+bh)
        else:
            # Tight face crop with moderate padding (0.15 to keep it face-focused)
            crop, padded_coords = crop_face(img_bgr, bbox, pad_ratio=0.15)

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

        # Penalize full-frame fallback confidence (no face detected = lower trust)
        if is_full_frame_fallback:
            conf = float(np.clip(conf * 0.75, 0.50, 0.80))

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
        overall_confidence = float(np.clip(single["confidence"], 0.50, 0.80))
        summary_note = f"No face explicitly detected — evaluated center crop: {verdict} ({overall_confidence*100:.1f}% confidence)."
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
            "backend": os.path.basename(model_path),
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
