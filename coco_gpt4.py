import os
import json
from PIL import Image
from tqdm import tqdm

# === 1. Define your class map ===
PPE_CLASSES = {
    0: "Hardhat",
    1: "Mask",
    2: "NO-Hardhat",
    3: "NO-Mask",
    4: "NO-Safety Vest",
    5: "Person",
    6: "Safety Cone",
    7: "Safety Vest",
    8: "Machinery",
    9: "Vehicle"
}

# === 2. Define paths ===
image_dir = "css-data/test/images"       # Path to folder containing images
label_dir = "css-data/test/labels"       # Path to folder containing YOLO .txt annotations
output_json = "instances_test.json"

# === 3. Prepare category mapping for COCO format ===
categories = [{"id": idx, "name": name, "supercategory": "none"} for idx, name in PPE_CLASSES.items()]

# === 4. Begin conversion ===
images = []
annotations = []
annotation_id = 1
image_id = 1

label_files = [f for f in os.listdir(label_dir) if f.endswith(".txt")]

for label_file in tqdm(label_files, desc="Converting YOLO to COCO"):
    image_filename = os.path.splitext(label_file)[0] + ".jpg"  # change to .png if needed
    image_path = os.path.join(image_dir, image_filename)
    label_path = os.path.join(label_dir, label_file)

    if not os.path.exists(image_path):
        print(f"⚠️ Skipping {label_file}: image not found ({image_filename})")
        continue

    with Image.open(image_path) as img:
        width, height = img.size

    images.append({
        "id": image_id,
        "file_name": image_filename,
        "width": width,
        "height": height
    })

    with open(label_path, "r") as f:
        for line in f:
            parts = line.strip().split()
            if len(parts) != 5:
                continue

            class_id, x_center, y_center, w, h = map(float, parts)
            class_id = int(class_id)

            # Convert normalized to absolute coordinates
            x_center *= width
            y_center *= height
            w *= width
            h *= height
            x = x_center - w / 2
            y = y_center - h / 2

            annotations.append({
                "id": annotation_id,
                "image_id": image_id,
                "category_id": class_id,
                "bbox": [x, y, w, h],
                "area": w * h,
                "iscrowd": 0
            })
            annotation_id += 1

    image_id += 1

# === 5. Write COCO JSON ===
coco_dict = {
    "images": images,
    "annotations": annotations,
    "categories": categories
}

with open(output_json, "w") as f:
    json.dump(coco_dict, f, indent=4)

print(f"\n✅ COCO annotations saved to: {output_json}")