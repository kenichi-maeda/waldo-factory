import argparse, random, csv
from pathlib import Path
from PIL import Image

# Deterministic runs if you set --seed
def yolo_box_from_paste(x, y, w, h, tile_size=128):
    cx = (x + w / 2) / tile_size
    cy = (y + h / 2) / tile_size
    nw = w / tile_size
    nh = h / tile_size
    return cx, cy, nw, nh

def paste_face_on_tile(tile_img, face_img, scale_range=(0.9, 1.1)):
    tile_w = tile_h = 128
    w, h = face_img.size
    scale = random.uniform(*scale_range)
    new_w, new_h = int(w * scale), int(h * scale)
    face = face_img.resize((new_w, new_h), Image.LANCZOS)

    # Random top-left so the face fits fully inside the tile
    max_x = tile_w - new_w
    max_y = tile_h - new_h
    x = random.randint(0, max_x)
    y = random.randint(0, max_y)

    # Paste with alpha
    tile_img.paste(face, (x, y), face)
    return x, y, new_w, new_h

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--crops_dir", type=Path, default=Path("data/crops_128"))
    ap.add_argument("--faces_dir", type=Path, default=Path("data/faces"))
    ap.add_argument("--out_images", type=Path, default=Path("data/dataset/images/train"))
    ap.add_argument("--out_labels", type=Path, default=Path("data/dataset/labels/train"))
    ap.add_argument("--spread_ids", type=str, default="1", help="Comma-separated spread folder names under crops_128")
    ap.add_argument("--label_frac", type=float, default=0.98, help="Fraction of tiles to receive a pasted face")
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()

    random.seed(args.seed)
    args.out_images.mkdir(parents=True, exist_ok=True)
    args.out_labels.mkdir(parents=True, exist_ok=True)

    # Class mapping (YOLO): 0 = Waldo (1-class)
    class_id = 0

    total_tiles = 0
    total_labeled = 0
    manifest_rows = []

    for sid in [s.strip() for s in args.spread_ids.split(",") if s.strip()]:
        spread_dir = args.crops_dir / sid
        face_path = args.faces_dir / f"{sid}_waldo.png"  # e.g., data/faces/1_waldo.png
        if not spread_dir.is_dir():
            print(f"[WARN] Missing spread dir: {spread_dir}")
            continue
        if not face_path.exists():
            print(f"[WARN] Missing face PNG for spread {sid}: {face_path}")
            continue

        face_img = Image.open(face_path).convert("RGBA")
        tiles = sorted(spread_dir.glob("*.png"))
        n_tiles = len(tiles)
        n_label = int(round(args.label_frac * n_tiles))

        # Randomly choose which tiles will get a face
        label_indices = set(random.sample(range(n_tiles), n_label))

        for i, tile_path in enumerate(tiles):
            tile = Image.open(tile_path).convert("RGBA")
            out_name = f"{sid}__{tile_path.stem}.png"
            out_img = args.out_images / out_name
            out_lbl = args.out_labels / f"{sid}__{tile_path.stem}.txt"

            if i in label_indices:
                x, y, w, h = paste_face_on_tile(tile, face_img)
                # Save composite as RGB (no alpha)
                tile.convert("RGB").save(out_img, quality=95)

                # Write YOLO label
                cx, cy, nw, nh = yolo_box_from_paste(x, y, w, h, tile_size=128)
                with open(out_lbl, "w") as f:
                    f.write(f"{class_id} {cx:.6f} {cy:.6f} {nw:.6f} {nh:.6f}\n")

                total_labeled += 1
                manifest_rows.append([sid, tile_path.name, out_name, x, y, w, h, cx, cy, nw, nh, 1])
            else:
                # Background: no label file
                tile.convert("RGB").save(out_img, quality=95)
                manifest_rows.append([sid, tile_path.name, out_name, "", "", "", "", "", "", "", "", 0])

            total_tiles += 1

    # Optional: write a simple manifest for bookkeeping
    man = args.out_images.parent / "manifest.csv"
    with open(man, "w", newline="") as f:
        csv.writer(f).writerow(
            ["spread", "src_tile", "out_image", "x", "y", "w", "h", "cx", "cy", "nw", "nh", "has_face"]
        )
        csv.writer(f).writerows(manifest_rows)

    print(f"[DONE] Tiles: {total_tiles} | Labeled (with Waldo): {total_labeled} "
          f"| Background: {total_tiles - total_labeled}")

if __name__ == "__main__":
    main()



# 1 1_r004_c005
# 2 2_r004_c000
# 3 3_r003_c011

# 5 5_r010_c004
# 6 6_r003_c013
# 7 7_r006_c006
# 8 8_r002_c009
# 9 9_r005_c001