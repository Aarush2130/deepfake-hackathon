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
    max_dim = 800
    scale = 1.0
    if max(h_orig, w_orig) > max_dim:
        scale = max_dim / float(max(h_orig, w_orig))
        work_img = cv2.resize(img_bgr, (int(w_orig * scale), int(h_orig * scale)))
    else:
        work_img = img_bgr.copy()

    gray = cv2.equalizeHist(cv2.cvtColor(work_img, cv2.COLOR_BGR2GRAY))
    detected = []

    if "alt2" in cascades:
        f = cascades["alt2"].detectMultiScale(gray, scaleFactor=1.1, minNeighbors=4, minSize=(30, 30))
        if len(f) > 0:
            detected.extend(f)

    if len(detected) == 0 and "default" in cascades:
        f = cascades["default"].detectMultiScale(gray, scaleFactor=1.1, minNeighbors=4, minSize=(30, 30))
        if len(f) > 0:
            detected.extend(f)

    if len(detected) == 0 and "profile" in cascades:
        f = cascades["profile"].detectMultiScale(gray, scaleFactor=1.1, minNeighbors=3, minSize=(30, 30))
        if len(f) > 0:
            detected.extend(f)

    if len(detected) == 0:
        return []

    scaled_boxes = [(int(x / scale), int(y / scale), int(w / scale), int(h / scale)) for (x, y, w, h) in detected]
    return sorted(nms_boxes(scaled_boxes, overlap_thresh=0.35), key=lambda b: b[0])

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

def classify_crop(crop_bgr):
    if crop_bgr is None or crop_bgr.size == 0:
        return 0.15, 0.15
        
    gray_crop = cv2.cvtColor(crop_bgr, cv2.COLOR_BGR2GRAY)
    fft_score = get_fft_anomaly(gray_crop)
    fake_prob = fft_score

    if classifier is not None:
        try:
            crop_rgb = cv2.cvtColor(crop_bgr, cv2.COLOR_BGR2RGB)
            pil_img = Image.fromarray(crop_rgb)
            tensor_img = img_transform(pil_img).unsqueeze(0).to(device)
            
            with torch.no_grad():
                logits = classifier(tensor_img)
                probs = torch.softmax(logits, dim=1).cpu().numpy()[0]
                # Alphabetical classes: Index 0 = Fake, Index 1 = Real
                fake_prob = float(probs[0])
        except Exception as e:
            print(f"Custom Model Inference Error: {e}")
            fake_prob = fft_score

    return fake_prob, fft_score

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

    for idx, bbox in enumerate(detected_bboxes):
        crop, padded_coords = (img_bgr, (0, 0, w_img, h_img)) if is_full_frame_fallback else crop_face(img_bgr, bbox, pad_ratio=0.20)
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

    manipulated_faces = [f for f in faces_results if f["manipulation_score"] >= 0.50]
    
    if is_full_frame_fallback:
        single = faces_results[0]
        is_fake = single["manipulation_score"] >= 0.50
        verdict = "Manipulated (Deepfake Detected)" if is_fake else "Authentic Media"
        overall_manipulation = single["manipulation_score"]
        overall_confidence = single["confidence"]
        summary_note = f"Evaluated as full-frame portrait: {verdict} ({overall_confidence*100:.1f}% confidence)."
        display_face_count = 1
    elif manipulated_faces:
        worst_face = max(faces_results, key=lambda f: f["manipulation_score"])
        verdict = f"Manipulated (Deepfake in {len(manipulated_faces)}/{num_faces} Subjects)"
        overall_manipulation = worst_face["manipulation_score"]
        overall_confidence = worst_face["confidence"]
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

    return {
        "status": "FACES_ANALYZED",
        "face_count": display_face_count,
        "faces": faces_results,
        "verdict": verdict,
        "confidence": overall_confidence,
        "manipulation_score": overall_manipulation,
        "fft_score": global_fft,
        "heatmap_path": heatmap_path,
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
    if worst_frame is not None:
        cv2.imwrite(worst_heatmap_path, generate_annotated_heatmap(worst_frame, worst_faces))
        
    return {
        "status": "VIDEO_ANALYZED",
        "face_count": max(1, len(worst_faces)),
        "faces": worst_faces,
        "verdict": "Manipulated Video Track" if is_fake else "Authentic Video Track",
        "confidence": max_score if is_fake else (1.0 - avg_score),
        "manipulation_score": max_score,
        "fft_score": avg_score,
        "heatmap_path": worst_heatmap_path,
        "summary_note": f"Analyzed {len(scores)} sampled frames."
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