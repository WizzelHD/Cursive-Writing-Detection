import io
import sys
import json
import base64
import traceback
from pathlib import Path

from fastapi import FastAPI, UploadFile, File
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
import uvicorn

# ─── PyTorch imports ────────────────────────────────────────────────
import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import transforms, models
from PIL import Image

# ─── Training-Code importieren ──────────────────────────────────────
CODE_DIR = Path(__file__).resolve().parent.parent.parent / "code"
sys.path.insert(0, str(CODE_DIR))

from data_code.iam_parser import parse_iam_words
from data_code.label_encode import LabelEncoder

app = FastAPI(title="Vision AI Backend")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# ─── Pfade ──────────────────────────────────────────────────────────
BASE_DIR = Path(__file__).resolve().parent
MODELS_DIR = BASE_DIR / "models"
DATA_DIR = CODE_DIR.parent / "data"

CRNN_PATH = MODELS_DIR / "crnn_best.pth"
RESNET_PATH = MODELS_DIR / "resnet_best.pth"

# ─── Globale Referenzen ────────────────────────────────────────────
resnet_model = None
crnn_model = None
encoder: LabelEncoder | None = None
idx_to_word: dict[int, str] = {}
device = torch.device("cpu")

if torch.backends.mps.is_available():
    device = torch.device("mps")
elif torch.cuda.is_available():
    device = torch.device("cuda")


# ════════════════════════════════════════════════════════════════════
# Modell-Architekturen (exakt wie im Training)
# ════════════════════════════════════════════════════════════════════

class CRNN(nn.Module):
    """Exakte Architektur aus code/encoder/crnn.py"""
    def __init__(self, num_classes):
        super(CRNN, self).__init__()
        self.cnn = nn.Sequential(
            nn.Conv2d(1, 64, 3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.MaxPool2d(2, 2),

            nn.Conv2d(64, 128, 3, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(),
            nn.MaxPool2d(2, 2),

            nn.Conv2d(128, 256, 3, padding=1),
            nn.BatchNorm2d(256),
            nn.ReLU(),

            nn.Conv2d(256, 256, 3, padding=1),
            nn.BatchNorm2d(256),
            nn.ReLU(),
        )
        # nach 2x Pooling: (1, 32, 128) -> (256, 8, 32)
        self.rnn = nn.LSTM(
            input_size=256 * 8,
            hidden_size=512,
            num_layers=2,
            bidirectional=True,
            batch_first=True,
            dropout=0.3
        )
        self.fc = nn.Linear(512 * 2, num_classes)
        self.log_softmax = nn.LogSoftmax(dim=2)

    def forward(self, x):
        x = self.cnn(x)
        batch, channels, height, width = x.size()
        x = x.permute(0, 3, 1, 2)
        x = x.contiguous().view(batch, width, channels * height)
        x, _ = self.rnn(x)
        x = self.fc(x)
        x = self.log_softmax(x)
        return x


class ResNetBaseline(nn.Module):
    """Exakte Architektur aus code/encoder/resnet.py"""
    def __init__(self, num_classes):
        super().__init__()
        self.model = models.resnet18(weights=None)
        self.model.conv1 = nn.Conv2d(
            1, 64, kernel_size=7, stride=2, padding=3, bias=False
        )
        self.model.maxpool = nn.Identity()
        self.model.fc = nn.Linear(self.model.fc.in_features, num_classes)

    def forward(self, x):
        return self.model(x)


# ════════════════════════════════════════════════════════════════════
# Preprocessing — exakt wie im Training (dataset.py)
# transforms.Resize((32, 128)) + transforms.ToTensor()
# ToTensor() konvertiert Graustufenbild zu [0, 1]
# ════════════════════════════════════════════════════════════════════

image_transform = transforms.Compose([
    transforms.Grayscale(num_output_channels=1),
    transforms.Resize((32, 128)),
    transforms.ToTensor(),
])





# ════════════════════════════════════════════════════════════════════
# Metadaten & Modelle laden
# ════════════════════════════════════════════════════════════════════

def load_metadata():
    """Lade Zeichensatz und Wort-Mapping aus den IAM-Daten."""
    global encoder, idx_to_word

    words_txt = DATA_DIR / "ascii" / "words.txt"
    if not words_txt.exists():
        print(f"[Vision AI] WARNUNG: {words_txt} nicht gefunden!")
        return 0, 0

    labels = parse_iam_words(str(words_txt))

    # CRNN: Zeichensatz
    all_text = "".join(labels.values())
    unique_chars = sorted(set(all_text))
    encoder = LabelEncoder(unique_chars)
    num_classes = len(unique_chars) + 1  # +1 für CTC blank

    # ResNet: Wort-Mapping
    unique_words = sorted(set(labels.values()))
    word_to_idx = {w: i for i, w in enumerate(unique_words)}
    idx_to_word = {i: w for w, i in word_to_idx.items()}
    num_word_classes = len(unique_words)

    print(f"[Vision AI] Zeichensatz: {len(unique_chars)} Zeichen + 1 blank = {num_classes} Klassen")
    print(f"[Vision AI] Wort-Klassen: {num_word_classes}")

    return num_classes, num_word_classes


def load_models():
    global resnet_model, crnn_model

    num_classes, num_word_classes = load_metadata()

    if CRNN_PATH.exists() and num_classes > 0:
        print(f"[Vision AI] Lade CRNN von {CRNN_PATH} ...")
        try:
            crnn_model = CRNN(num_classes)
            state_dict = torch.load(str(CRNN_PATH), map_location=device, weights_only=True)
            crnn_model.load_state_dict(state_dict)
            crnn_model.to(device)
            crnn_model.eval()
            print("[Vision AI] ✓ CRNN geladen!")
        except Exception as e:
            print(f"[Vision AI] ✗ Fehler beim Laden von CRNN: {e}")
            traceback.print_exc()
            crnn_model = None
    else:
        print(f"[Vision AI] CRNN nicht gefunden unter {CRNN_PATH}")

    if RESNET_PATH.exists() and num_word_classes > 0:
        print(f"[Vision AI] Lade ResNet von {RESNET_PATH} ...")
        try:
            resnet_model = ResNetBaseline(num_word_classes)
            state_dict = torch.load(str(RESNET_PATH), map_location=device, weights_only=True)
            resnet_model.load_state_dict(state_dict)
            resnet_model.to(device)
            resnet_model.eval()
            print("[Vision AI] ✓ ResNet geladen!")
        except Exception as e:
            print(f"[Vision AI] ✗ Fehler beim Laden von ResNet: {e}")
            traceback.print_exc()
            resnet_model = None
    else:
        print(f"[Vision AI] ResNet nicht gefunden unter {RESNET_PATH}")


# ════════════════════════════════════════════════════════════════════
# Inference
# ════════════════════════════════════════════════════════════════════

def ctc_greedy_decode(output: torch.Tensor) -> tuple[str, float]:
    """
    CTC Greedy Decoding.
    output shape: (batch=1, seq_len=32, num_classes=80) — log-softmax Werte
    """
    # output ist log_softmax, also exp() für Wahrscheinlichkeiten
    probs = output[0].exp()  # (seq_len, num_classes)
    max_probs, indices = probs.max(dim=1)

    result_chars = []
    confidences = []
    last_idx = -1

    for t in range(indices.size(0)):
        idx = indices[t].item()
        conf = max_probs[t].item()
        # 0 = CTC blank, skip blank und Wiederholungen
        if idx != 0 and idx != last_idx:
            result_chars.append(encoder.decode([idx]))
            confidences.append(conf)
        last_idx = idx

    text = "".join(result_chars)
    avg_conf = sum(confidences) / len(confidences) if confidences else 0.0
    return text, avg_conf


def run_resnet(image: Image.Image) -> dict:
    if resnet_model is None:
        return {"label": "ResNet nicht geladen", "confidence": 0.0, "topResults": []}

    img_tensor = image_transform(image).unsqueeze(0).to(device)

    with torch.no_grad():
        output = resnet_model(img_tensor)
        probs = F.softmax(output, dim=1)
        top5_conf, top5_idx = probs.topk(5, dim=1)

    top_results = []
    for i in range(5):
        idx = top5_idx[0, i].item()
        conf = top5_conf[0, i].item()
        word = idx_to_word.get(idx, f"Klasse {idx}")
        top_results.append({"label": word, "confidence": round(conf, 4)})

    return {
        "label": top_results[0]["label"],
        "confidence": top_results[0]["confidence"],
        "topResults": top_results,
    }


def run_crnn(image: Image.Image) -> dict:
    if crnn_model is None or encoder is None:
        return {"text": "CRNN nicht geladen", "confidence": 0.0}

    img_tensor = image_transform(image).unsqueeze(0).to(device)

    with torch.no_grad():
        output = crnn_model(img_tensor)  # (batch, seq_len, num_classes)

    text, conf = ctc_greedy_decode(output)

    return {
        "text": text if text else "(kein Text erkannt)",
        "confidence": round(conf, 4),
    }


# ════════════════════════════════════════════════════════════════════
# API Endpunkte
# ════════════════════════════════════════════════════════════════════

@app.on_event("startup")
async def startup():
    MODELS_DIR.mkdir(parents=True, exist_ok=True)
    load_models()


@app.get("/health")
async def health():
    return {
        "status": "ok",
        "resnet_loaded": resnet_model is not None,
        "crnn_loaded": crnn_model is not None,
        "device": str(device),
    }


@app.post("/analyze")
async def analyze(file: UploadFile = File(...)):
    """Nimmt ein Bild entgegen und gibt die Ergebnisse beider Modelle zurueck."""
    try:
        contents = await file.read()
        image = Image.open(io.BytesIO(contents))

        resnet_result = run_resnet(image)
        crnn_result = run_crnn(image)

        return JSONResponse(content={
            "resnet": resnet_result,
            "crnn": crnn_result,
        })
    except Exception as e:
        traceback.print_exc()
        return JSONResponse(
            status_code=500,
            content={"error": f"Analyse fehlgeschlagen: {str(e)}"},
        )


@app.post("/analyze/base64")
async def analyze_base64(body: dict):
    """
    Nimmt ein Base64-kodiertes Bild entgegen (fuer Kamera-Aufnahmen).
    Body: { "image": "data:image/jpeg;base64,..." }
    """
    try:
        image_data = body.get("image", "")
        if "," in image_data:
            image_data = image_data.split(",", 1)[1]

        image_bytes = base64.b64decode(image_data)
        image = Image.open(io.BytesIO(image_bytes))

        resnet_result = run_resnet(image)
        crnn_result = run_crnn(image)

        return JSONResponse(content={
            "resnet": resnet_result,
            "crnn": crnn_result,
        })
    except Exception as e:
        traceback.print_exc()
        return JSONResponse(
            status_code=500,
            content={"error": f"Analyse fehlgeschlagen: {str(e)}"},
        )


# ════════════════════════════════════════════════════════════════════
# Start
# ════════════════════════════════════════════════════════════════════

if __name__ == "__main__":
    print("=" * 60)
    print("  Vision AI - Python Backend")
    print("=" * 60)
    print(f"  Modell-Ordner:  {MODELS_DIR}")
    print(f"  IAM Daten:      {DATA_DIR}")
    print(f"  Device:         {device}")
    print(f"  Server:         http://localhost:8000")
    print("=" * 60)
    uvicorn.run(app, host="0.0.0.0", port=8000)
