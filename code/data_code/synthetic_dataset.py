import os
import glob
import torch
from PIL import Image
from torch.utils.data import Dataset
import torchvision.transforms as transforms


class SyntheticHandwritingDataset(Dataset):
    """
    Generate synthetic handwriting images with TextRecognitionDataGenerator (TRDG).

    Place handwriting fonts (.ttf) in fonts/handwriting/.
    Suggested fonts (Google Fonts): Caveat, Kalam, Patrick Hand, Shadows Into Light, Indie Flower.

    If save_dir already contains a labels.txt from a previous run, the data is
    loaded from disk instead of being regenerated.
    """

    def __init__(self, words, encoder, count, fonts_dir=None, save_dir=None):
        self.encoder = encoder
        self.transform = transforms.Compose([
            transforms.Resize((32, 128)),
            transforms.ToTensor(),
        ])
        self.samples = []

        # Load from disk if data already exists
        if save_dir:
            labels_path = os.path.join(save_dir, "labels.txt")
            if os.path.exists(labels_path):
                self._load_from_disk(save_dir, labels_path)
                if self.samples:
                    print(f"[Synthetic] {len(self.samples)} Samples von Disk geladen ({save_dir})")
                    return
                else:
                    print("[Synthetic] labels.txt gefunden, aber keine Samples geladen. Neu generieren...")

        self._generate(words, encoder, count, fonts_dir, save_dir)

    def _load_from_disk(self, save_dir, labels_path):
        try:
            with open(labels_path, "r", encoding="utf-8") as f:
                for line in f:
                    line = line.strip()
                    if not line:
                        continue
                    parts = line.split("\t", 1)
                    if len(parts) != 2:
                        continue
                    filename, label = parts
                    img_path = os.path.join(save_dir, filename)
                    if os.path.exists(img_path):
                        img = Image.open(img_path).convert("L")
                        self.samples.append((img, label))
        except Exception as e:
            print(f"[Synthetic] Fehler beim Laden von Disk: {e}")
            self.samples = []

    def _generate(self, words, encoder, count, fonts_dir, save_dir):
        labels_file = None
        try:
            from trdg.generators import GeneratorFromStrings
            import trdg

            fonts = self._find_fonts(fonts_dir, trdg)

            if not fonts:
                print("[Synthetic] Keine Fonts gefunden. Synthetisches Dataset wird uebersprungen.")
                return

            print(f"[Synthetic] {len(fonts)} Font(s) gefunden. Generiere {count} Bilder...")

            # Nur Woerter verwenden, deren Zeichen im Encoder vorhanden sind
            valid_words = [
                w for w in words
                if w and all(c in encoder.char_to_idx for c in w)
            ]

            if not valid_words:
                print("[Synthetic] Keine gueltigen Woerter fuer synthetisches Dataset.")
                return

            if save_dir:
                os.makedirs(save_dir, exist_ok=True)
                labels_path = os.path.join(save_dir, "labels.txt")
                labels_file = open(labels_path, "w", encoding="utf-8")

            target_count = count
            accepted = 0
            total_generated = 0
            rejected_empty = 0
            rejected_invalid = 0
            max_attempts = 5
            attempt = 0

            while accepted < target_count and attempt < max_attempts:
                remaining = target_count - accepted
                gen_count = max(remaining * 2, 1000)

                generator = GeneratorFromStrings(
                    strings=valid_words,
                    count=gen_count,
                    fonts=fonts,
                    size=32,
                    image_mode="L",        # Graustufen wie IAM
                    distorsion_type=3,     # Zufallige Verzerrung fuer Handschrift-Optik
                    distorsion_orientation=0,
                    background_type=1,     # Weisser Hintergrund wie IAM-Scans
                    fit=True,              # Bild eng an Text croppen
                )

                accepted_before = accepted
                for _, (img, label) in enumerate(generator):
                    total_generated += 1
                    if img is None:
                        rejected_empty += 1
                        continue
                    if not label:
                        rejected_empty += 1
                        continue
                    if not all(c in encoder.char_to_idx for c in label):
                        rejected_invalid += 1
                        continue

                    if labels_file is not None:
                        filename = f"{accepted:07d}.png"
                        img_path = os.path.join(save_dir, filename)
                        img.save(img_path)
                        labels_file.write(f"{filename}\t{label}\n")

                    self.samples.append((img, label))
                    accepted += 1
                    if accepted >= target_count:
                        break

                if accepted == accepted_before:
                    break
                attempt += 1

            print(
                "[Synthetic] "
                f"{len(self.samples)} synthetische Samples generiert "
                f"(target={target_count}, generated={total_generated}, "
                f"rejected_empty={rejected_empty}, rejected_invalid={rejected_invalid})."
            )

        except ImportError:
            print("[Synthetic] trdg nicht installiert. pip install trdg")
        except Exception as e:
            print(f"[Synthetic] Fehler bei der Datengenerierung: {e}")
        finally:
            if labels_file is not None:
                labels_file.close()

    def _find_fonts(self, fonts_dir, trdg_module):
        fonts = []

        # 1. Eigener Handschrift-Font-Ordner im Projekt
        if fonts_dir and os.path.isdir(fonts_dir):
            fonts = glob.glob(os.path.join(fonts_dir, "*.ttf"))
            if fonts:
                return fonts

        # Projektverzeichnis: fonts/handwriting/
        project_fonts = os.path.join(
            os.path.dirname(__file__), "..", "..", "fonts", "handwriting"
        )
        project_fonts = os.path.normpath(project_fonts)
        if os.path.isdir(project_fonts):
            fonts = glob.glob(os.path.join(project_fonts, "*.ttf"))
            if fonts:
                return fonts

        # 2. Fallback: TRDG-eigene Latin-Fonts
        trdg_font_dir = os.path.join(os.path.dirname(trdg_module.__file__), "fonts", "latin")
        if os.path.isdir(trdg_font_dir):
            fonts = glob.glob(os.path.join(trdg_font_dir, "*.ttf"))

        return fonts

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        img, text = self.samples[idx]
        img = self.transform(img)
        encoded = torch.tensor(self.encoder.encode(text), dtype=torch.long)
        return img, encoded, len(encoded)
