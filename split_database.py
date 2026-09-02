from pathlib import Path
import random
import shutil

SOURCE_DIR = Path("VisDrone2019-DET-train/VisDrone2019-DET-train")
IMAGES_DIR = SOURCE_DIR / "images"
ANNOTATIONS_DIR = SOURCE_DIR / "annotations"
OUTPUT_DIR = Path(".")
NUM_CLIENTS = 3
SEED = 42
IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png"}
CLASS_IDS = set(range(10))


def convert_annotation(annotation_path, image_path, label_path):
    from PIL import Image

    width, height = Image.open(image_path).size
    labels = []
    for row in annotation_path.read_text(encoding="utf-8").splitlines():
        values = row.split(",")
        if len(values) < 6:
            continue
        x, y, box_width, box_height = map(int, values[:4])
        ignored = int(values[4])
        class_id = int(values[5]) - 1
        if ignored == 1 or class_id not in CLASS_IDS:
            continue
        labels.append(
            f"{class_id} {(x + box_width / 2) / width:.6f} "
            f"{(y + box_height / 2) / height:.6f} "
            f"{box_width / width:.6f} {box_height / height:.6f}\n"
        )
    label_path.write_text("".join(labels), encoding="utf-8")


if not IMAGES_DIR.exists() or not ANNOTATIONS_DIR.exists():
    raise FileNotFoundError(
        f"Expected source images and annotations under {SOURCE_DIR}"
    )

images = sorted(
    image for image in IMAGES_DIR.iterdir()
    if image.suffix.lower() in IMAGE_EXTENSIONS
    and (ANNOTATIONS_DIR / f"{image.stem}.txt").exists()
)
print(f"Total paired images found: {len(images)}")
random.seed(SEED)
random.shuffle(images)

for client_id in range(1, NUM_CLIENTS + 1):
    client_dir = OUTPUT_DIR / f"client{client_id}_project" / "data" / f"client{client_id}"
    shutil.rmtree(client_dir / "images", ignore_errors=True)
    shutil.rmtree(client_dir / "labels", ignore_errors=True)
    (client_dir / "images").mkdir(parents=True, exist_ok=True)
    (client_dir / "labels").mkdir(parents=True, exist_ok=True)

for index, image_path in enumerate(images):
    client_id = (index % NUM_CLIENTS) + 1
    client_dir = OUTPUT_DIR / f"client{client_id}_project" / "data" / f"client{client_id}"
    shutil.copy2(image_path, client_dir / "images" / image_path.name)
    convert_annotation(
        ANNOTATIONS_DIR / f"{image_path.stem}.txt",
        image_path,
        client_dir / "labels" / f"{image_path.stem}.txt",
    )

print("\nDataset split completed!\n")
for client_id in range(1, NUM_CLIENTS + 1):
    client_dir = OUTPUT_DIR / f"client{client_id}_project" / "data" / f"client{client_id}"
    client_images = list((client_dir / "images").glob("*"))
    client_labels = list((client_dir / "labels").glob("*.txt"))
    print(
        f"Client {client_id}: {len(client_images)} images, "
        f"{len(client_labels)} labels"
    )
