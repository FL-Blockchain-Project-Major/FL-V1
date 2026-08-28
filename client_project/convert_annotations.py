"""
convert_annotations.py
========================
Converts VisDrone annotation format → YOLO format.

Usage:
    python convert_annotations.py --client client1
    python convert_annotations.py --client client2
    FL_CLIENT_ID=client3 python convert_annotations.py
"""

import argparse
import os
from pathlib import Path

from PIL import Image
from tqdm import tqdm


def parse_args():
    parser = argparse.ArgumentParser(
        description="VisDrone → YOLO annotation converter"
    )
    parser.add_argument(
        "--client",
        default=os.environ.get("FL_CLIENT_ID", "client1"),
        help="Client ID, e.g. client1 / client2 / client3",
    )
    return parser.parse_args()


def convert_box(image_size, box):
    """
    VisDrone: x, y, width, height (top-left corner, pixel absolute)
    YOLO:     x_center, y_center, width, height (normalized 0-1)
    """
    iw, ih = image_size
    x, y, w, h = box
    return (
        (x + w / 2) / iw,
        (y + h / 2) / ih,
        w / iw,
        h / ih,
    )


def convert(client_id: str):
    client_dir      = Path(f"data/{client_id}")
    images_dir      = client_dir / "images"
    annotations_dir = client_dir / "annotations"
    labels_dir      = client_dir / "labels"
    labels_dir.mkdir(parents=True, exist_ok=True)

    annotation_files = sorted(annotations_dir.glob("*.txt"))

    if not annotation_files:
        print(f"✘  No annotation files found in {annotations_dir}")
        return

    print(f"\nFound {len(annotation_files)} annotation files")
    converted = 0
    skipped   = 0

    for ann_file in tqdm(annotation_files, desc="Converting"):
        image_file = images_dir / f"{ann_file.stem}.jpg"
        if not image_file.exists():
            skipped += 1
            continue

        with Image.open(image_file) as img:
            img_size = img.size

        yolo_lines = []
        with open(ann_file) as f:
            for line in f:
                row = line.strip().split(",")
                if len(row) < 8:
                    continue
                x, y, w, h = int(row[0]), int(row[1]), int(row[2]), int(row[3])
                score       = int(row[4])
                class_id    = int(row[5])

                if score == 0 or w == 0 or h == 0:
                    continue

                class_id -= 1  # VisDrone 1-10 → YOLO 0-9
                xc, yc, nw, nh = convert_box(img_size, (x, y, w, h))
                yolo_lines.append(
                    f"{class_id} {xc:.6f} {yc:.6f} {nw:.6f} {nh:.6f}\n"
                )

        out_file = labels_dir / ann_file.name
        with open(out_file, "w") as f:
            f.writelines(yolo_lines)
        converted += 1

    print(f"\n{'='*50}")
    print(f"Conversion complete for {client_id}")
    print(f"  Converted : {converted}")
    print(f"  Skipped   : {skipped} (no matching image)")
    print(f"  Labels in : {labels_dir.resolve()}")
    print(f"{'='*50}\n")


if __name__ == "__main__":
    args = parse_args()
    convert(args.client)
