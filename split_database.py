from pathlib import Path
import random
import shutil
SOURCE_DIR = Path("C:\\Users\\KIIT0001\\Desktop\\PROJECT\\FL-V1\\VisDrone2019-DET-train\\VisDrone2019-DET-train")
IMAGES_DIR = SOURCE_DIR / "images"
ANNOTATIONS_DIR = SOURCE_DIR / "annotations"
OUTPUT_DIR = Path("clients")
NUM_CLIENTS = 3
SEED = 42
image_extensions = {".jpg", ".jpeg", ".png"}
images = [
    image
    for image in IMAGES_DIR.iterdir()
    if image.suffix.lower() in image_extensions
]
print(f"Total images found: {len(images)}")
random.seed(SEED)
random.shuffle(images)
for client_id in range(1, NUM_CLIENTS + 1):

    (OUTPUT_DIR / f"client{client_id}" / "images").mkdir(
        parents=True,
        exist_ok=True
    )
    (OUTPUT_DIR / f"client{client_id}" / "annotations").mkdir(
        parents=True,
        exist_ok=True
    )
for index, image_path in enumerate(images):
    client_id = (index % NUM_CLIENTS) + 1
    client_dir = OUTPUT_DIR / f"client{client_id}"
    shutil.copy2(
        image_path,
        client_dir / "images" / image_path.name
    )
    annotation_path = (
        ANNOTATIONS_DIR / f"{image_path.stem}.txt"
    )
    if annotation_path.exists():
        shutil.copy2(
            annotation_path,
            client_dir / "annotations" / annotation_path.name
        )
    else:
        print(f"WARNING: No annotation found for {image_path.name}")
print("\nDataset split completed!\n")
for client_id in range(1, NUM_CLIENTS + 1):
    client_images = list(
        (OUTPUT_DIR / f"client{client_id}" / "images").glob("*")
    )
    client_annotations = list(
        (OUTPUT_DIR / f"client{client_id}" / "annotations").glob("*")
    )
    print(
        f"Client {client_id}: "
        f"{len(client_images)} images, "
        f"{len(client_annotations)} annotations"
    )