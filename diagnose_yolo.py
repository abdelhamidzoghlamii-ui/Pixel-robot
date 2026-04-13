def diagnose_yolo(image_path="~/test.jpg", model_path="~/robot/yolov8n.onnx"):
    import onnxruntime as ort
    import numpy as np
    import os
    from PIL import Image

    print("=" * 50)
    print("YOLO DIAGNOSTIC")
    print("=" * 50)

    print("\n[1] Loading model...")
    sess = ort.InferenceSession(os.path.expanduser(model_path), providers=["CPUExecutionProvider"])
    inp = sess.get_inputs()[0]
    out = sess.get_outputs()[0]
    print(f"    Input  shape : {inp.shape}")
    print(f"    Output shape : {out.shape}")

    print("\n[2] Loading image...")
    img = Image.open(os.path.expanduser(image_path)).convert("RGB")
    print(f"    Image size   : {img.width}w x {img.height}h")

    print("\n[3] Running inference (naive 640x640 resize)...")
    img_resized = img.resize((640, 640), Image.BILINEAR)
    tensor = np.array(img_resized, dtype=np.float32) / 255.0
    tensor = np.transpose(tensor, (2, 0, 1))[np.newaxis]
    raw = sess.run(None, {inp.name: tensor})[0]
    print(f"    Raw output shape : {raw.shape}")

    print("\n[4] Top person scores (trying both layouts)...")
    if raw.shape[1] == 84:
        scores = raw[0, 4, :]
        layout = "(1, 84, 8400)"
    else:
        scores = raw[0, :, 4]
        layout = "(1, 8400, 84)"
    print(f"    Layout detected  : {layout}")
    top5 = np.sort(scores)[::-1][:5]
    print(f"    Top-5 person scores : {np.round(top5, 3)}")
    print(f"    Max person score    : {scores.max():.4f}")

    print("\n[5] Top-3 classes overall...")
    flat = raw[0]
    class_scores = flat[4:, :] if flat.shape[0] == 84 else flat[:, 4:].T
    max_per_class = class_scores.max(axis=1)
    top3 = np.argsort(max_per_class)[::-1][:3]
    coco = {0:"person",2:"car",15:"cat",16:"dog",39:"bottle",56:"chair",57:"couch"}
    for c in top3:
        print(f"    class {c:3d} ({coco.get(c, 'other'):8s}): {max_per_class[c]:.3f}")

    print("\n" + "=" * 50)

diagnose_yolo()
