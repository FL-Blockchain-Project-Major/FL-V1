from pathlib import Path
from PIL import Image
from tqdm import tqdm


# =========================================================
# PATHS
# =========================================================

CLIENT_DIR = Path(__file__).resolve().parent / "data" / "client3"

IMAGES_DIR = CLIENT_DIR / "images"
ANNOTATIONS_DIR = CLIENT_DIR / "annotations"
LABELS_DIR = CLIENT_DIR / "labels"

# Create labels folder automatically
LABELS_DIR.mkdir(parents=True, exist_ok=True)


# =========================================================
# CONVERT BOUNDING BOX
# =========================================================

def convert_box(image_size, box):
    """
    VisDrone format:
    x, y, width, height

    YOLO format:
    x_center, y_center, width, height

    All YOLO values are normalized between 0 and 1.
    """

    image_width, image_height = image_size

    x, y, width, height = box

    x_center = (x + width / 2) / image_width
    y_center = (y + height / 2) / image_height

    normalized_width = width / image_width
    normalized_height = height / image_height

    return (
        x_center,
        y_center,
        normalized_width,
        normalized_height
    )


# =========================================================
# CONVERSION
# =========================================================

annotation_files = list(
    ANNOTATIONS_DIR.glob("*.txt")
)

print(f"\nAnnotations found: {len(annotation_files)}")

converted = 0

for annotation_file in tqdm(
    annotation_files,
    desc="Converting VisDrone annotations"
):

    # Example:
    # annotations/000001.txt
    # images/000001.jpg

    image_file = (
        IMAGES_DIR /
        f"{annotation_file.stem}.jpg"
    )

    # Skip if corresponding image doesn't exist
    if not image_file.exists():

        print(
            f"\nWARNING: Image not found:\n{image_file}"
        )

        continue

    # Get image dimensions
    with Image.open(image_file) as image:
        image_size = image.size

    yolo_lines = []

    # Read VisDrone annotation file
    with open(
        annotation_file,
        "r"
    ) as file:

        for line in file:

            row = line.strip().split(",")

            # Make sure annotation is valid
            if len(row) < 8:
                continue

            # -------------------------------------------------
            # VisDrone format:
            #
            # x, y, width, height,
            # score,
            # object_category,
            # truncation,
            # occlusion
            # -------------------------------------------------

            x = int(row[0])
            y = int(row[1])
            width = int(row[2])
            height = int(row[3])

            score = int(row[4])
            class_id = int(row[5])

            # Ignore regions
            if score == 0:
                continue

            # Convert VisDrone class IDs 1-10
            # into YOLO IDs 0-9
            class_id = class_id - 1

            # Convert bounding box
            x_center, y_center, width, height = convert_box(
                image_size,
                (x, y, width, height)
            )

            # YOLO format:
            # class x_center y_center width height

            yolo_lines.append(
                f"{class_id} "
                f"{x_center:.6f} "
                f"{y_center:.6f} "
                f"{width:.6f} "
                f"{height:.6f}\n"
            )

    # Save YOLO label file

    output_file = (
        LABELS_DIR /
        annotation_file.name
    )

    with open(
        output_file,
        "w"
    ) as file:

        file.writelines(yolo_lines)

    converted += 1


# =========================================================
# RESULTS
# =========================================================

print("\n" + "=" * 60)
print("CONVERSION COMPLETE")
print("=" * 60)

print(f"Annotation files found : {len(annotation_files)}")
print(f"YOLO label files created: {converted}")
print(f"\nLabels location:\n{LABELS_DIR.resolve()}")