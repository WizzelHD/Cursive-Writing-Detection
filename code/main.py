import os
import random
import torch
import torch.optim as optim
from torch.utils.data import DataLoader, random_split, ConcatDataset
from data_code.iam_parser import parse_iam_words
from data_code.dataset import IAMWordDataset, SyntheticWordDataset, crnn_collate_fn, resnet_collate_fn
from data_code.label_encode import LabelEncoder
from data_code.word_classification_dataset import WordClassificationDataset
from data_code.synthetic_dataset import SyntheticHandwritingDataset
from data_code.generate_synthetic import load_synthetic_labels
from encoder.crnn import CRNN
from encoder.resnet import ResNetBaseline
from training.train_crnn import train_crnn
from training.train_resnet import train_resnet
from metrics import cer_for_loader, evaluate_resnet
from evaluation import run_confusion_analysis, run_final_report

os.environ["PYTORCH_ENABLE_MPS_FALLBACK"] = "1"

# ─── Synthetic augmentation toggle ──────────────────────────────────────────
# Set to True to include synthetic trdg images in the CRNN training split.
# Run  python data_code/generate_synthetic.py  first to create the images.
USE_SYNTHETIC = True
SYNTHETIC_WORDS_ROOT = "../data/synthetic/words"
SYNTHETIC_LABELS_FILE = "../data/synthetic/synthetic_labels.txt"

def set_seed(seed: int = 42):
    random.seed(seed)
    torch.manual_seed(seed)


def main():
    set_seed(42)

    # Device selection: Metal (Apple Silicon) > CUDA (NVIDIA) > CPU
    if torch.backends.mps.is_available():
        device = torch.device("mps")
        print("Using device: mps (Metal Performance Shaders)")
    elif torch.cuda.is_available():
        device = torch.device("cuda")
        torch.backends.cudnn.benchmark = False
        print("Using device: cuda")
    else:
        device = torch.device("cpu")
        print("Using device: cpu")

    os.makedirs("../pth", exist_ok=True)

    # IAM labels
    labels = parse_iam_words("../data/ascii/words.txt")


    # Character Encoder (CRNN)
    all_text = "".join(labels.values())
    unique_chars = sorted(set(all_text))
    encoder = LabelEncoder(unique_chars)
    num_classes = len(unique_chars) + 1  # + blank for CTC


    # Dataset Split
    dataset = IAMWordDataset(
        labels_dict=labels,
        words_root="../data/words",
        encoder=encoder
    )

    train_size = int(0.8 * len(dataset))
    val_size = int(0.1 * len(dataset))
    test_size = len(dataset) - train_size - val_size

    generator = torch.Generator().manual_seed(42)

    train_dataset, val_dataset, test_dataset = random_split(
        dataset,
        [train_size, val_size, test_size],
        generator=generator
    )

    # Synthetische Handschrift-Daten zum Training hinzufügen
    word_list_for_synth = list(labels.values())
    synthetic_dataset = SyntheticHandwritingDataset(
        words=word_list_for_synth,
        encoder=encoder,
        count=50000,
        save_dir="../data/synthetic"
    )

    if len(synthetic_dataset) > 0:
        combined_train = ConcatDataset([train_dataset, synthetic_dataset])
        print(f"[INFO] Training mit IAM ({len(train_dataset)}) + Synthetic ({len(synthetic_dataset)}) = {len(combined_train)} Samples")
    else:
        combined_train = train_dataset
        print(f"[INFO] Training nur mit IAM ({len(train_dataset)}) Samples (kein synthetisches Dataset)")

    train_loader = DataLoader(
        combined_train,
        batch_size=64,
        shuffle=True,
        collate_fn=crnn_collate_fn,
        num_workers=8,
        pin_memory=True
    )

    # ── Optionally merge synthetic data into CRNN train split ──────────────
    if USE_SYNTHETIC:
        from pathlib import Path
        syn_labels = load_synthetic_labels(Path(SYNTHETIC_LABELS_FILE))
        if syn_labels:
            syn_dataset = SyntheticWordDataset(
                labels_dict=syn_labels,
                words_root=SYNTHETIC_WORDS_ROOT,
                encoder=encoder,
            )
            combined_train = ConcatDataset([train_dataset, syn_dataset])
            print(f"[synthetic] Combined train: {len(train_dataset)} IAM + {len(syn_dataset)} synthetic = {len(combined_train)} total")
            train_loader = DataLoader(
                combined_train,
                batch_size=64,
                shuffle=True,
                collate_fn=crnn_collate_fn,
                num_workers=8,
                pin_memory=True
            )
        else:
            print("[synthetic] No labels found — skipping synthetic augmentation.")

    val_loader = DataLoader(
        val_dataset,
        batch_size=64,
        shuffle=False,
        collate_fn=crnn_collate_fn,
        num_workers=0
    )

    test_loader = DataLoader(
        test_dataset,
        batch_size=64,
        shuffle=False,
        collate_fn=crnn_collate_fn,
        num_workers=0
    )

    # ----------- CRNN TRAINING ------------------
    print("\n========== Training CRNN ==========\n")

    crnn_model = CRNN(num_classes).to(device)

    optimizer = optim.Adam(crnn_model.parameters(), lr=1e-4)

    scheduler = optim.lr_scheduler.ReduceLROnPlateau(
        optimizer,
        mode="min",
        factor=0.5,
        patience=2
    )

    crnn_ckpt_path = "../pth/crnn_last.pth"
    crnn_best_path = "../pth/crnn_best.pth"

    crnn_model, train_losses, val_cers = train_crnn(
        model=crnn_model,
        train_loader=train_loader,
        val_loader=val_loader,
        optimizer=optimizer,
        scheduler=scheduler,
        encoder=encoder,
        device=device,
        ckpt_path=crnn_ckpt_path,
        best_model_path=crnn_best_path,
        epochs =40
    )


    # ----------- CRNN EVALUATION ----------------

    print("\n========== Evaluating CRNN ==========\n")

    crnn_model.load_state_dict(torch.load(crnn_best_path, map_location=device, weights_only=True))
    crnn_model.eval()

    test_cer = cer_for_loader(crnn_model, test_loader, encoder, device)
    print(f"Final CRNN Test CER: {test_cer:.4f}")

    run_confusion_analysis(crnn_model, test_loader, encoder, device)


    # ----------- RESNET TRAINING ----------------

    print("\n========== Training ResNet ==========\n")

    word_list = list(labels.values())
    unique_words = sorted(set(word_list))
    word_to_idx = {w: i for i, w in enumerate(unique_words)}
    num_word_classes = len(unique_words)

    resnet_train_dataset = WordClassificationDataset(train_dataset, word_to_idx)
    resnet_val_dataset = WordClassificationDataset(val_dataset, word_to_idx)
    resnet_test_dataset = WordClassificationDataset(test_dataset, word_to_idx)

    if len(synthetic_dataset) > 0:
        resnet_synthetic_dataset = WordClassificationDataset(synthetic_dataset, word_to_idx)
        resnet_combined_train = ConcatDataset([resnet_train_dataset, resnet_synthetic_dataset])
        print(f"[INFO] ResNet Training mit IAM ({len(resnet_train_dataset)}) + Synthetic ({len(resnet_synthetic_dataset)}) = {len(resnet_combined_train)} Samples")
    else:
        resnet_combined_train = resnet_train_dataset
        print(f"[INFO] ResNet Training nur mit IAM ({len(resnet_train_dataset)}) Samples")

    resnet_train_loader = DataLoader(
        resnet_combined_train,
        batch_size=64,
        shuffle=True,
        num_workers=0,
        collate_fn=resnet_collate_fn
    )

    resnet_val_loader = DataLoader(
        resnet_val_dataset,
        batch_size=64,
        shuffle=False,
        num_workers=0,
        collate_fn=resnet_collate_fn
    )

    resnet_test_loader = DataLoader(
        resnet_test_dataset,
        batch_size=64,
        shuffle=False,
        num_workers=0,
        collate_fn=resnet_collate_fn
    )
    resnet_model = ResNetBaseline(num_word_classes).to(device)

    resnet_ckpt_path = "../pth/resnet_last.pth"
    resnet_best_path = "../pth/resnet_best.pth"

    optimizer = torch.optim.Adam(resnet_model.parameters(), lr=1e-4)

    resnet_scheduler = optim.lr_scheduler.ReduceLROnPlateau(
        optimizer,
        mode="max",
        factor=0.5,
        patience=2
    )

    resnet_model, _, _ = train_resnet(
        model=resnet_model,
        train_loader=resnet_train_loader,
        val_loader=resnet_val_loader,
        device=device,
        ckpt_path=resnet_ckpt_path,
        best_model_path=resnet_best_path,
        optimizer=optimizer,
        scheduler=resnet_scheduler,
        epochs=40
    )


    # ----------- RESNET EVALUATION --------------

    print("\n========== Evaluating ResNet ==========\n")

    resnet_model.load_state_dict(torch.load(resnet_best_path, map_location=device, weights_only=True))
    resnet_model.eval()

    resnet_acc = evaluate_resnet(resnet_model, resnet_test_loader, device)
    print(f"Final ResNet Test Accuracy: {resnet_acc:.4f}")


    idx_to_word = {v: k for k, v in word_to_idx.items()}

    run_final_report(
        crnn_model,
        resnet_model,
        test_loader,
        resnet_test_loader,
        encoder,
        idx_to_word,
        device
    )

    print("\nPipeline finished successfully.")


if __name__ == "__main__":
    main()



