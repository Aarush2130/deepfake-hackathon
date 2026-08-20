"""Diagnose class index mapping - is class 0 = Fake or Real?"""
import engine
import cv2
import numpy as np
import glob

# Check for actual test images
imgs = glob.glob("*.jpg") + glob.glob("*.png") + glob.glob("*.jpeg")
print("Available images:", imgs[:10])

# Test 1: Uniform skin-tone (simulates real face texture with camera noise)
skin = np.zeros((224, 224, 3), dtype=np.uint8)
skin[:, :] = [180, 200, 230]  # warm skin tone BGR
skin = skin.astype(np.float32) + np.random.normal(0, 5, skin.shape)
skin = np.clip(skin, 0, 255).astype(np.uint8)

r1 = engine.classify_crop(skin)
print("\nSkin-tone patch (simulates real photo texture):")
print(f"  neural_prob = {r1['neural_prob']:.4f}")
print(f"  composite   = {r1['composite_prob']:.4f}")
print(f"  Verdict: {'FAKE' if r1['composite_prob'] >= 0.5 else 'REAL'}")

# Test 2: Perfectly smooth gradient (simulates AI-generated face)
y = np.linspace(0, 1, 224).reshape(-1, 1)
x = np.linspace(0, 1, 224).reshape(1, -1)
grad_xy = y * x * 200
grad_y = np.repeat(y * 180, 224, axis=1)
grad_x = np.repeat(x * 220, 224, axis=0)
smooth = np.stack([grad_xy, grad_y, grad_x], axis=-1).astype(np.uint8)

r2 = engine.classify_crop(smooth)
print("\nSmooth gradient (simulates AI generation):")
print(f"  neural_prob = {r2['neural_prob']:.4f}")
print(f"  composite   = {r2['composite_prob']:.4f}")
print(f"  Verdict: {'FAKE' if r2['composite_prob'] >= 0.5 else 'REAL'}")

# Test 3: If there are actual images, test one
if imgs:
    img = cv2.imread(imgs[0])
    if img is not None:
        crop = cv2.resize(img, (224, 224))
        r3 = engine.classify_crop(crop)
        print(f"\nActual image ({imgs[0]}):")
        print(f"  neural_prob = {r3['neural_prob']:.4f}")
        print(f"  composite   = {r3['composite_prob']:.4f}")
        print(f"  Verdict: {'FAKE' if r3['composite_prob'] >= 0.5 else 'REAL'}")

print("\n=== INTERPRETATION ===")
print("If BOTH skin patch AND smooth gradient show as FAKE -> class index may be inverted")
print("The model should classify noise/gradients as FAKE and real textures as REAL")
