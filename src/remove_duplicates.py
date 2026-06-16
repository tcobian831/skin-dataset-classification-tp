from pathlib import Path
import hashlib
import csv
from collections import defaultdict

DATASET_DIR = Path("data/Split_smol")
OUTPUT_CSV = Path("data/splits/duplicate_report.csv")

IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff", ".webp"}


def file_hash(path: Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def main():
    rows = []

    for split in ["train", "val"]:
        split_dir = DATASET_DIR / split

        for img_path in split_dir.rglob("*"):
            if img_path.is_file() and img_path.suffix.lower() in IMAGE_EXTENSIONS:
                rows.append({
                    "path": str(img_path),
                    "split_original": split,
                    "class": img_path.parent.name,
                    "hash": file_hash(img_path),
                })

    groups = defaultdict(list)
    for row in rows:
        groups[row["hash"]].append(row)

    report = []
    duplicate_groups = 0
    duplicate_images = 0

    for group_id, items in enumerate(groups.values(), start=1):
        items = sorted(items, key=lambda x: x["path"])
        is_duplicate_group = len(items) > 1

        if is_duplicate_group:
            duplicate_groups += 1
            duplicate_images += len(items) - 1

        for i, row in enumerate(items):
            report.append({
                **row,
                "duplicate_group_id": group_id if is_duplicate_group else "",
                "is_duplicate": is_duplicate_group and i > 0,
                "kept": i == 0,
            })

    OUTPUT_CSV.parent.mkdir(parents=True, exist_ok=True)

    with open(OUTPUT_CSV, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=[
            "path",
            "split_original",
            "class",
            "hash",
            "duplicate_group_id",
            "is_duplicate",
            "kept",
        ])
        writer.writeheader()
        writer.writerows(report)

    print(f"Total imágenes analizadas: {len(rows)}")
    print(f"Grupos con duplicados exactos: {duplicate_groups}")
    print(f"Imágenes duplicadas marcadas para excluir: {duplicate_images}")
    print(f"Reporte guardado en: {OUTPUT_CSV}")


if __name__ == "__main__":
    main()