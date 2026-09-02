from pathlib import Path
import random
import shutil

PROJECT_ROOT = Path(__file__).resolve().parent
SOURCE_DIR = PROJECT_ROOT / "VisDrone2019-DET-train" / "VisDrone2019-DET-train"
IMAGES_DIR = SOURCE_DIR / "images"
ANNOTATIONS_DIR = SOURCE_DIR / "annotations"
OUTPUT_DIR = PROJECT_ROOT
CLIENTS_DIR = OUTPUT_DIR / "clients"
NUM_CLIENTS = 3
SEED = 42
IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png"}


if not IMAGES_DIR.exists() or not ANNOTATIONS_DIR.exists():
    raise FileNotFoundError(
        f"Expected source images and annotations under {SOURCE_DIR}"
    )

images = sorted(
    image
    for image in IMAGES_DIR.iterdir()
    if image.suffix.lower() in IMAGE_EXTENSIONS
    and (ANNOTATIONS_DIR / f"{image.stem}.txt").exists()
)
print(f"Total paired images found: {len(images)}")
random.seed(SEED)
random.shuffle(images)

for client_id in range(1, NUM_CLIENTS + 1):
    client_dir = CLIENTS_DIR / f"client{client_id}"
    shutil.rmtree(client_dir / "images", ignore_errors=True)
    shutil.rmtree(client_dir / "annotations", ignore_errors=True)
    shutil.rmtree(client_dir / "labels", ignore_errors=True)
    (client_dir / "annotations").mkdir(parents=True, exist_ok=True)
    (client_dir / "images").mkdir(parents=True, exist_ok=True)

for index, image_path in enumerate(images):
    client_id = (index % NUM_CLIENTS) + 1
    client_dir = CLIENTS_DIR / f"client{client_id}"
    annotation_path = ANNOTATIONS_DIR / f"{image_path.stem}.txt"
    shutil.copy2(image_path, client_dir / "images" / image_path.name)
    shutil.copy2(annotation_path, client_dir / "annotations" / annotation_path.name)

print("\nDataset split completed!\n")
for client_id in range(1, NUM_CLIENTS + 1):
    client_dir = CLIENTS_DIR / f"client{client_id}"
    client_images = list((client_dir / "images").glob("*"))
    client_annotations = list((client_dir / "annotations").glob("*.txt"))
    print(
        f"Client {client_id}: {len(client_images)} images, "
        f"{len(client_annotations)} annotations"
    )
