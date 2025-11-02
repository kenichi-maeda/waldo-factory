import argparse, math
from pathlib import Path
from PIL import Image, ImageOps
import csv

def pad_to_multiple(img: Image.Image, tile: int, pad_color=(0, 0, 0)):
    w, h = img.size
    new_w = math.ceil(w / tile) * tile
    new_h = math.ceil(h / tile) * tile
    pad_right = new_w - w
    pad_bottom = new_h - h
    if pad_right == 0 and pad_bottom == 0:
        return img, (0, 0, 0, 0)
    return ImageOps.expand(img, border=(0, 0, pad_right, pad_bottom), fill=pad_color), (0, 0, pad_right, pad_bottom)

def crop_grid(img_path: Path, out_dir: Path, tile: int = 128, image_format="png"):
    img = Image.open(img_path).convert("RGB")
    padded, _ = pad_to_multiple(img, tile, pad_color=(0, 0, 0))  # zero padding at edges
    W, H = padded.size
    cols, rows = W // tile, H // tile

    page_dir = out_dir / img_path.stem
    page_dir.mkdir(parents=True, exist_ok=True)

    manifest_path = page_dir / "manifest.csv"
    with open(manifest_path, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["tile_path", "col", "row", "x", "y", "w", "h", "page_width", "page_height"])
        for r in range(rows):
            for c in range(cols):
                x, y = c * tile, r * tile
                tile_img = padded.crop((x, y, x + tile, y + tile))
                tile_name = f"{img_path.stem}_r{r:03d}_c{c:03d}.{image_format}"
                tile_path = page_dir / tile_name
                tile_img.save(tile_path, quality=95)
                writer.writerow([str(tile_path), c, r, x, y, tile, tile, W, H])
    return cols, rows, page_dir

def main():
    ap = argparse.ArgumentParser(description="Crop an image into a grid of 128x128 tiles with edge padding.")
    ap.add_argument("image", type=Path, help="Path to a single page image (PNG/JPG).")
    ap.add_argument("--out", type=Path, default=Path("data/crops_128"), help="Output directory for tiles.")
    ap.add_argument("--tile", type=int, default=128, help="Tile size (default: 128).")
    ap.add_argument("--format", type=str, default="png", choices=["png", "jpg", "jpeg"], help="Output image format.")
    args = ap.parse_args()

    args.out.mkdir(parents=True, exist_ok=True)
    cols, rows, page_dir = crop_grid(args.image, args.out, tile=args.tile, image_format=args.format)
    print(f"[OK] {args.image.name}: {cols*rows} tiles ({cols} cols x {rows} rows) → {page_dir}")

if __name__ == "__main__":
    main()
