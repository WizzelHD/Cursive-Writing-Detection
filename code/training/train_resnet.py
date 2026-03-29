import torch
import torch.nn as nn
import os
from plotting import plot_training_curves 
from metrics import evaluate_resnet

def train_resnet(
    model,
    train_loader,
    val_loader,
    optimizer,
    device,
    ckpt_path,
    best_model_path,
    epochs,
    scheduler=None,
    patience=8
):
    """
    Trainiert ein ResNet-Modell zur Wortklassifikation.

    Das Modell wird mit CrossEntropyLoss optimiert und nach jeder Epoche
    auf dem Validierungsdatensatz evaluiert (Word Accuracy).

    Funktionen:
    - Speichert Checkpoints (letzter Stand + bestes Modell)
    - Unterstützt Resume-Training bei vorhandenem Checkpoint
    - Aktualisiert Trainings- und Validierungsmetriken
    - Implementiert Early Stopping basierend auf Validation Accuracy
    """
    criterion = nn.CrossEntropyLoss()

    start_epoch = 0
    best_acc = 0.0
    epochs_without_improvement = 0

    train_accs = []
    val_accs = []

    if os.path.exists(ckpt_path):
        try:
            print(f"[INFO] Found ResNet checkpoint: {ckpt_path}. Resuming...")
            ckpt = torch.load(ckpt_path, map_location=device, weights_only=False)
            model.load_state_dict(ckpt["model"])
            optimizer.load_state_dict(ckpt["optimizer"])
            if scheduler is not None and "scheduler" in ckpt:
                scheduler.load_state_dict(ckpt["scheduler"])
            start_epoch = ckpt["epoch"] + 1
            best_acc = ckpt["best_acc"]
            train_accs = ckpt.get("train_accs", [])
            val_accs = ckpt.get("val_accs", [])
            epochs_without_improvement = ckpt.get("epochs_without_improvement", 0)
            if start_epoch >= epochs:
                print("[INFO] Training already finished. Regenerating plots...")

                try:
                    train_errors = [1 - a for a in train_accs]
                    val_errors = [1 - a for a in val_accs]
                    plot_training_curves(train_errors, val_errors, prefix="resnet")
                except Exception as e:
                    print(f"[WARN] Plot regeneration failed: {e}")

                return model, train_accs, val_accs
            print(f"[INFO] Resuming from Epoch {start_epoch}, Best Val Acc so far: {best_acc*100:.2f}%")
        except Exception as e:
            print(f"[WARN] Checkpoint load failed: {e}. Starting fresh.")

    for epoch in range(start_epoch, epochs):
        print(f"\n--- ResNet Epoch {epoch + 1}/{epochs} ---")
        model.train()
        running_loss = 0.0
        correct = 0
        total = 0

        for batch_idx, batch in enumerate(train_loader):
            if batch is None:
                continue
            
            images, targets = batch
            images = images.to(device)
            targets = targets.to(device)

            optimizer.zero_grad()
            outputs = model(images)
            loss = criterion(outputs, targets)
            loss.backward()
            optimizer.step()

            # Statistiken berechnen
            running_loss += loss.item()
            preds = torch.argmax(outputs, dim=1)
            correct += (preds == targets).sum().item()
            total += targets.size(0)

            if batch_idx % 50 == 0:
                if device.type == "cuda":
                    torch.cuda.synchronize()


            if batch_idx % 100 == 0:
                current_loss = running_loss / (batch_idx + 1)
                current_acc = (correct / total) * 100 if total > 0 else 0
                print(f"Batch {batch_idx:4d} / {len(train_loader)} | "
                      f"Loss: {current_loss:.4f} | "
                      f"Train Acc: {current_acc:.2f}%")

        if total == 0: 
            print("[WARN] No valid data in this epoch.")
            continue 

        train_acc = correct / total
        print(f"[INFO] Epoch {epoch+1} finished. Evaluating on validation set...")
        val_acc = evaluate_resnet(model, val_loader, device)

        if scheduler is not None:
            scheduler.step(val_acc)

        train_accs.append(train_acc)
        val_accs.append(val_acc)

        print(f"RESULT: Epoch {epoch+1} | Train Acc: {train_acc*100:.2f}% | Val Acc: {val_acc*100:.2f}%")

        # checkpoint
        ckpt_data = {
            "epoch": epoch,
            "model": model.state_dict(),
            "optimizer": optimizer.state_dict(),
            "best_acc": best_acc,
            "epochs_without_improvement": epochs_without_improvement,
            "train_accs": train_accs,
            "val_accs": val_accs
        }
        if scheduler is not None:
            ckpt_data["scheduler"] = scheduler.state_dict()
        torch.save(ckpt_data, ckpt_path)

        # plot
        try:
            train_errors = [1 - a for a in train_accs]
            val_errors = [1 - a for a in val_accs] 
            plot_training_curves(train_errors, val_errors, prefix="resnet")
        except Exception as e:
            print(f"[WARN] Could not update plots: {e}")

        # bestes model
        if val_acc > best_acc:
            print(f"New Best Model! (Val Acc improved from {best_acc*100:.2f}% to {val_acc*100:.2f}%) ***")
            best_acc = val_acc
            epochs_without_improvement = 0
            torch.save(model.state_dict(), best_model_path)
        else:
            epochs_without_improvement += 1
            print(f"[IDLE] No improvement for {epochs_without_improvement}/{patience} epochs.")

        if epochs_without_improvement >= patience:
            print("Early stopping triggered. ResNet training complete.")
            break

    return model, train_accs, val_accs

