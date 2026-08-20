"""
VeriChain Forensic OS — Bulk Evaluation & Batch Benchmarking Engine
Processes thousands of images (e.g., 4,000+) in batch mode.

Features:
- Fast multi-image inference utilizing CUDA/GPU acceleration
- Supports ground-truth benchmarking (e.g., 'real' and 'fake' subfolders) or unlabelled flat directories
- Calculates Accuracy, Precision, Recall, F1-score, and Confusion Matrix
- Exports full itemized forensic logs to 'batch_evaluation_results.csv'
- Identifies and lists hardest edge-case misclassifications for inspection
"""

import os
import sys
import time
import glob
import argparse
import numpy as np
import pandas as pd
import cv2
import torch
from PIL import Image

# Ensure UTF-8 output on Windows consoles
if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8")
if hasattr(sys.stderr, "reconfigure"):
    sys.stderr.reconfigure(encoding="utf-8")

import engine

def collect_images(input_path, max_samples=None):
    """Discovers all valid image files in the directory tree with optional balanced sampling."""
    valid_exts = {".jpg", ".jpeg", ".png", ".webp", ".bmp", ".jfif"}
    image_entries = []

    # Check if folder has real/fake subfolders
    subdirs = [d for d in os.listdir(input_path) if os.path.isdir(os.path.join(input_path, d))]
    has_labels = any("fake" in d.lower() for d in subdirs) and any("real" in d.lower() for d in subdirs)

    if has_labels:
        print(f"[*] Detected labeled benchmark structure in '{input_path}'")
        fake_entries, real_entries = [], []
        for root, _, files in os.walk(input_path):
            folder_name = os.path.basename(root).lower()
            if any(term in folder_name for term in ["fake", "synth", "manipulated", "deepfake"]):
                for f in files:
                    if os.path.splitext(f)[1].lower() in valid_exts:
                        fake_entries.append({"file_path": os.path.join(root, f), "filename": f, "ground_truth": "Fake"})
            elif "real" in folder_name or "auth" in folder_name:
                for f in files:
                    if os.path.splitext(f)[1].lower() in valid_exts:
                        real_entries.append({"file_path": os.path.join(root, f), "filename": f, "ground_truth": "Real"})

        if max_samples and max_samples < (len(fake_entries) + len(real_entries)):
            per_class = max_samples // 2
            image_entries = fake_entries[:per_class] + real_entries[:per_class]
            print(f"[*] Balanced sampling applied: {len(image_entries):,} images ({per_class:,} Fake + {per_class:,} Real)")
        else:
            image_entries = fake_entries + real_entries
    else:
        print(f"[*] Scanning flat/unlabelled directory: '{input_path}'")
        for root, _, files in os.walk(input_path):
            for f in files:
                ext = os.path.splitext(f)[1].lower()
                if ext in valid_exts:
                    image_entries.append({
                        "file_path": os.path.join(root, f),
                        "filename": f,
                        "ground_truth": "Unknown"
                    })
        if max_samples and max_samples < len(image_entries):
            image_entries = image_entries[:max_samples]

    return image_entries

def run_bulk_evaluation(images, output_csv="batch_evaluation_results.csv", fast_mode=False):
    total_imgs = len(images)
    print(f"[*] Starting bulk evaluation on {total_imgs:,} images...")
    print(f"[*] Active Model: {engine.model_path} on {engine.device}")
    print(f"[*] Fast Mode (Pre-Cropped Face): {'ENABLED' if fast_mode else 'DISABLED'}")
    print("="*75)

    results = []
    start_time = time.time()

    for idx, entry in enumerate(images, 1):
        fpath = entry["file_path"]
        fname = entry["filename"]
        gt = entry["ground_truth"]

        try:
            img_bgr = cv2.imread(fpath)
            if img_bgr is None:
                results.append({
                    "filename": fname,
                    "ground_truth": gt,
                    "prediction": "ERROR_CORRUPTED",
                    "confidence": 0.0,
                    "manipulation_score": 0.0,
                    "neural_score": 0.0,
                    "fft_score": 0.0,
                    "ela_score": 0.0,
                    "face_count": 0,
                    "is_correct": False,
                    "status": "FAIL_READ"
                })
                continue

            if fast_mode:
                crop = img_bgr
                face_count = 1
            else:
                bboxes = engine.detect_faces(img_bgr)
                h_img, w_img = img_bgr.shape[:2]

                if len(bboxes) == 0:
                    cx, cy = w_img // 2, h_img // 2
                    cw, ch = int(w_img * 0.6), int(h_img * 0.6)
                    crop = img_bgr[max(0, cy - ch//2):min(h_img, cy + ch//2), max(0, cx - cw//2):min(w_img, cx + cw//2)]
                    face_count = 0
                else:
                    crop, _ = engine.crop_face(img_bgr, bboxes[0], pad_ratio=0.15)
                    face_count = len(bboxes)

            # Classify
            metrics = engine.classify_crop(crop)
            fake_prob = metrics["composite_prob"]
            pred_label = "Fake" if fake_prob >= 0.50 else "Real"
            conf = fake_prob if pred_label == "Fake" else (1.0 - fake_prob)

            is_correct = (pred_label == gt) if gt in ["Real", "Fake"] else None

            results.append({
                "filename": fname,
                "file_path": fpath,
                "ground_truth": gt,
                "prediction": pred_label,
                "confidence": round(float(conf), 4),
                "manipulation_score": round(float(fake_prob), 4),
                "neural_score": round(float(metrics["neural_prob"]), 4),
                "fft_score": round(float(metrics["fft_score"]), 4),
                "ela_score": round(float(metrics["ela_score"]), 4),
                "face_count": face_count,
                "is_correct": is_correct,
                "status": "SUCCESS"
            })

        except Exception as err:
            results.append({
                "filename": fname,
                "file_path": fpath,
                "ground_truth": gt,
                "prediction": "ERROR",
                "confidence": 0.0,
                "manipulation_score": 0.0,
                "neural_score": 0.0,
                "fft_score": 0.0,
                "ela_score": 0.0,
                "face_count": 0,
                "is_correct": False,
                "status": f"ERROR: {err}"
            })

        if idx % 50 == 0 or idx == total_imgs:
            elapsed = time.time() - start_time
            rate = idx / elapsed if elapsed > 0 else 0
            eta_sec = (total_imgs - idx) / rate if rate > 0 else 0
            print(f"[{idx:5d}/{total_imgs:5d}] ({idx/total_imgs*100:5.1f}%) | Speed: {rate:5.1f} img/s | ETA: {eta_sec:4.0f}s")

    # Convert to DataFrame
    df = pd.DataFrame(results)
    df.to_csv(output_csv, index=False)
    total_elapsed = time.time() - start_time

    print("\n" + "="*75)
    print("📊 BULK EVALUATION SUMMARY")
    print("="*75)
    print(f"⏱️ Total Time Elapsed: {total_elapsed:.1f} seconds ({total_imgs/total_elapsed:.1f} images/sec)")
    print(f"📁 Itemized CSV exported to: '{output_csv}'")

    # If ground truth was available, calculate benchmark metrics
    labeled_df = df[df["ground_truth"].isin(["Real", "Fake"])]
    if len(labeled_df) > 0:
        tp = len(labeled_df[(labeled_df["ground_truth"] == "Fake") & (labeled_df["prediction"] == "Fake")])
        fp = len(labeled_df[(labeled_df["ground_truth"] == "Real") & (labeled_df["prediction"] == "Fake")])
        tn = len(labeled_df[(labeled_df["ground_truth"] == "Real") & (labeled_df["prediction"] == "Real")])
        fn = len(labeled_df[(labeled_df["ground_truth"] == "Fake") & (labeled_df["prediction"] == "Real")])

        accuracy = (tp + tn) / len(labeled_df) * 100.0
        precision = tp / (tp + fp) * 100.0 if (tp + fp) > 0 else 0.0
        recall = tp / (tp + fn) * 100.0 if (tp + fn) > 0 else 0.0
        f1 = (2 * precision * recall) / (precision + recall) if (precision + recall) > 0 else 0.0

        print(f"\n🎯 BENCHMARK ACCURACY METRICS ({len(labeled_df):,} Labeled Images):")
        print(f"   - Overall Accuracy:  {accuracy:.2f}%")
        print(f"   - Precision (Fake):  {precision:.2f}%")
        print(f"   - Recall / Sens:     {recall:.2f}%")
        print(f"   - F1-Score:          {f1:.2f}%")
        print(f"\n🧩 CONFUSION MATRIX:")
        print(f"   - True Positives  (Deepfakes caught):    {tp:,}")
        print(f"   - True Negatives  (Authentic verified):  {tn:,}")
        print(f"   - False Positives (Real flagged fake):   {fp:,}")
        print(f"   - False Negatives (Deepfakes missed):    {fn:,}")

        # List misclassified samples
        misclassified = labeled_df[labeled_df["is_correct"] == False]
        if len(misclassified) > 0:
            print(f"\n⚠️ Misclassified Samples ({len(misclassified):,} total):")
            for _, r in misclassified.head(10).iterrows():
                print(f"   * '{r['filename']}' | GT={r['ground_truth']} | Pred={r['prediction']} (Score={r['manipulation_score']:.3f})")
            if len(misclassified) > 10:
                print(f"   ... and {len(misclassified) - 10} more in CSV.")
    else:
        fake_count = len(df[df["prediction"] == "Fake"])
        real_count = len(df[df["prediction"] == "Real"])
        print(f"\n📈 Unlabelled Distribution:")
        print(f"   - Predicted Deepfake:   {fake_count:,} ({fake_count/total_imgs*100:.1f}%)")
        print(f"   - Predicted Authentic:  {real_count:,} ({real_count/total_imgs*100:.1f}%)")

    print("="*75)

def main():
    parser = argparse.ArgumentParser(description="Bulk Deepfake Evaluation on 4,000+ images")
    parser.add_argument("--input_dir", type=str, required=True, help="Path to folder of images (or folder with 'real' & 'fake' subfolders)")
    parser.add_argument("--output_csv", type=str, default="batch_evaluation_results.csv", help="Destination CSV file for logs")
    parser.add_argument("--max_samples", type=int, default=None, help="Limit number of images to test (e.g. 4000)")
    parser.add_argument("--fast", action="store_true", help="Fast mode: skips face detector for pre-cropped face datasets")
    args = parser.parse_args()

    if not os.path.exists(args.input_dir):
        print(f"❌ Error: Input directory '{args.input_dir}' does not exist.")
        return

    images = collect_images(args.input_dir, max_samples=args.max_samples)
    if len(images) == 0:
        print(f"❌ No supported image files (.jpg, .png, .webp, .jpeg) found in '{args.input_dir}'.")
        return

    run_bulk_evaluation(images, output_csv=args.output_csv, fast_mode=args.fast)

if __name__ == "__main__":
    main()
