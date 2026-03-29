"""
Synthetic word image generator using TextRecognitionDataGenerator (trdg).

Generates grayscale word images compatible with the IAM training pipeline.
Output is saved to data/synthetic/words/ with a flat structure.
A labels file (synthetic_labels.txt) is written alongside.

Usage:
    python generate_synthetic.py                  # generates 50k images from IAM vocab
    python generate_synthetic.py --count 10000    # custom count
    python generate_synthetic.py --count 0        # only re-write labels file

The generated images + labels can be loaded via load_synthetic_labels() and
passed into IAMWordDataset with words_root pointing to the synthetic folder.
"""

import os
import sys
import argparse
import random
from pathlib import Path

from PIL import Image

# ─── Paths ──────────────────────────────────────────────────────────
SCRIPT_DIR  = Path(__file__).resolve().parent
DATA_DIR    = SCRIPT_DIR.parent.parent / "data"
OUT_DIR     = DATA_DIR / "synthetic" / "words"
LABELS_FILE = DATA_DIR / "synthetic" / "synthetic_labels.txt"
IAM_WORDS   = DATA_DIR / "ascii" / "words.txt"

BACKGROUNDS = [1, 0]  # 1 = plain white, 0 = gaussian noise


def get_iam_vocab(words_txt: Path) -> list[str]:
    """Return unique words from IAM words.txt (only 'ok' entries)."""
    words = set()
    with open(words_txt, encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line or line.startswith("#"):
                continue
            parts = line.split()
            if len(parts) < 9 or parts[1] != "ok":
                continue
            words.add(" ".join(parts[8:]))
    return sorted(words)


def _patch_trdg_pillow() -> None:
    """
    trdg uses FreeTypeFont.getsize() which was removed in Pillow 10.
    Patch everywhere trdg references get_text_height.
    """
    try:
        from PIL import ImageFont

        def _get_text_height(image_font: ImageFont.FreeTypeFont, text: str) -> int:
            ascent, descent = image_font.getmetrics()
            bbox = image_font.getbbox(text)
            bbox_height = 0
            if bbox:
                bbox_height = bbox[3] - bbox[1]
            return max(ascent + descent, bbox_height)

        import trdg.utils as _trdg_utils
        _trdg_utils.get_text_height = _get_text_height

        # computer_text_generator does `from trdg.utils import get_text_height`
        # so we must patch its local reference too
        import trdg.computer_text_generator as _ctg
        _ctg.get_text_height = _get_text_height
    except Exception as e:
        print(f"[warn] trdg pillow patch failed: {e}")


def _add_padding(img: Image.Image, top=6, bottom=14, left=8, right=8) -> Image.Image:
    """Add constant white padding to avoid clipped ascenders/descenders."""
    new_w = img.width + left + right
    new_h = img.height + top + bottom
    padded = Image.new("L", (new_w, new_h), color=255)
    padded.paste(img, (left, top))
    return padded


def generate(count: int, vocab: list[str], out_dir: Path) -> dict[str, str]:
    """
    Generate `count` synthetic images and return {image_id: word}.
    Images are saved as <image_id>.png under out_dir.
    """
    _patch_trdg_pillow()
    from trdg.generators import GeneratorFromStrings

    out_dir.mkdir(parents=True, exist_ok=True)

    # Sample words (with replacement if count > vocab size)
    words_to_generate = random.choices(vocab, k=count)

    labels: dict[str, str] = {}
    idx = 0

    # Generate in batches of 500 to avoid large generator overhead
    batch_size = 500
    generated = 0

    while generated < count:
        batch = words_to_generate[generated : generated + batch_size]

        generator = GeneratorFromStrings(
            strings=batch,
            count=len(batch),
            blur=random.randint(0, 1),
            random_blur=True,
            background_type=random.choice(BACKGROUNDS),
            skewing_angle=2,
            random_skew=True,
            distorsion_type=0,           # no distortion by default
            image_mode="L",              # grayscale — matches IAM
            character_spacing=0,
            margins=(10, 10, 10, 10),    # generous margins so descenders aren't clipped
            fit=True,
            size=64,                     # large font — dataset transform resizes to 32×128
        )

        for img, label in generator:
            if img is None:
                continue
            img = _add_padding(img)
            image_id = f"syn-{idx:07d}"
            img_path = out_dir / f"{image_id}.png"
            img.save(str(img_path))
            labels[image_id] = label
            idx += 1

        generated += len(batch)
        if generated % 5000 == 0 or generated >= count:
            print(f"  Generated {generated}/{count} images...")

    return labels


def write_labels(labels: dict[str, str], labels_file: Path) -> None:
    labels_file.parent.mkdir(parents=True, exist_ok=True)
    with open(labels_file, "w", encoding="utf-8") as f:
        for image_id, word in labels.items():
            f.write(f"{image_id} {word}\n")
    print(f"Labels written to {labels_file}  ({len(labels)} entries)")


def load_synthetic_labels(labels_file: Path | None = None) -> dict[str, str]:
    """
    Load synthetic labels file and return {image_id: word}.
    Call this from main.py to merge with IAM data.
    """
    path = labels_file or LABELS_FILE
    if not path.exists():
        print(f"[synthetic] Labels file not found: {path}")
        return {}
    labels = {}
    with open(path, encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            parts = line.split(" ", 1)
            if len(parts) == 2:
                labels[parts[0]] = parts[1]
    print(f"[synthetic] Loaded {len(labels)} synthetic labels from {path}")
    return labels


def main():
    parser = argparse.ArgumentParser(description="Generate synthetic word images with trdg")
    parser.add_argument("--count",    type=int,  default=50_000, help="Number of images to generate")
    parser.add_argument("--out_dir",  type=str,  default=str(OUT_DIR),  help="Output directory for images")
    parser.add_argument("--labels",   type=str,  default=str(LABELS_FILE), help="Path to write labels file")
    parser.add_argument("--iam_words",type=str,  default=str(IAM_WORDS), help="Path to IAM words.txt")
    parser.add_argument("--seed",     type=int,  default=42)
    args = parser.parse_args()

    random.seed(args.seed)

    out_dir     = Path(args.out_dir)
    labels_file = Path(args.labels)

    print(f"Loading IAM vocabulary from {args.iam_words} ...")
    vocab = get_iam_vocab(Path(args.iam_words))
    print(f"Vocabulary size: {len(vocab)} unique words")

    if args.count > 0:
        print(f"Generating {args.count} synthetic images → {out_dir}")
        labels = generate(args.count, vocab, out_dir)
    else:
        # reload existing
        labels = {}
        print("count=0 → skipping generation, re-writing labels only")
        for p in sorted(out_dir.glob("syn-*.png")):
            # can't recover labels from filenames → skip
            pass

    write_labels(labels, labels_file)
    print("Done.")


if __name__ == "__main__":
    main()
