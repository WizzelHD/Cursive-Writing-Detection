import torch
import torch.nn as nn
from metrics import cer_for_loader
from plotting import plot_training_curves
import os

def train_crnn(
    model,
    train_loader,
    val_loader,
    optimizer,
    scheduler,
    encoder,
    device,
    ckpt_path,
    best_model_path,
    epochs,
    patience=8
):
    """
    Trainiert ein CRNN-Modell zur zeichenbasierten Handschrifterkennung.

    Das Modell wird mit CTC-Loss optimiert und nach jeder Epoche
    anhand der Character Error Rate (CER) auf dem Validierungs-
    datensatz evaluiert.

    Funktionen:
    - Speichert Checkpoints (letzter Stand + bestes Modell)
    - Unterstützt Resume-Training bei vorhandenem Checkpoint
    - Passt die Lernrate mittels Scheduler (ReduceLROnPlateau) an
    - Aktualisiert Trainingsverlust und Validation CER
    - Implementiert Early Stopping basierend auf Validation CER
    """

    criterion = nn.CTCLoss(blank=0, zero_infinity=True)
    
    # CTC loss is not supported on MPS, so we always compute it on CPU
    ctc_device = torch.device("cpu") if device.type == "mps" else device
    best_val_cer = float("inf")
    epochs_without_improvement = 0
    start_epoch = 0
    train_losses = []
    val_cers = []

    # Resume from checkpoint
    if ckpt_path and os.path.exists(ckpt_path):
        try:
            ckpt = torch.load(ckpt_path, map_location=device, weights_only=False)

            model.load_state_dict(ckpt["model"])
            optimizer.load_state_dict(ckpt["optimizer"])
            scheduler.load_state_dict(ckpt["scheduler"])

            best_val_cer = ckpt["best_val_cer"]
            start_epoch = ckpt["epoch"] + 1

            train_losses = ckpt.get("train_losses", [])
            val_cers = ckpt.get("val_cers", [])
            epochs_without_improvement = ckpt.get("epochs_without_improvement", 0)

            if start_epoch >= epochs:
                print("[INFO] CRNN training already finished. Regenerating plots...")

                try:
                    plot_training_curves(train_losses, val_cers, prefix="crnn")
                except Exception as e:
                    print(f"[WARN] Plot regeneration failed: {e}")

                return model, train_losses, val_cers
        except Exception as e:
            print(f"Checkpoint load failed: {e}")

    # training
    for epoch in range(start_epoch, epochs):
        print(f"Starting epoch {epoch+1}")
        model.train()
        running_loss = 0.0

        for batch_idx, batch in enumerate(train_loader):
            if batch is None: continue
            
            images, labels_batch, lengths = batch
            images, labels_batch, lengths = images.to(device), labels_batch.to(device), lengths.to(device)

            optimizer.zero_grad()
            outputs = model(images).permute(1, 0, 2)

            input_lengths = torch.full(
                size=(images.size(0),),
                fill_value=outputs.size(0),
                dtype=torch.long,
                device=device
            )

            # CTC loss computation: move to CPU if using MPS (CTC not supported on MPS)
            if device.type == "mps":
                outputs_cpu = outputs.to("cpu")
                labels_cpu = labels_batch.to("cpu")
                input_lengths_cpu = input_lengths.to("cpu")
                lengths_cpu = lengths.to("cpu")
                loss = criterion(outputs_cpu, labels_cpu, input_lengths_cpu, lengths_cpu)
                loss = loss.to(device)
            else:
                loss = criterion(outputs, labels_batch, input_lengths, lengths)
            
            if torch.isinf(loss) or torch.isnan(loss):
                continue

            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 5.0)
            optimizer.step()

            running_loss += loss.item()

            if batch_idx % 50 == 0:
                if device.type == "cuda":
                    torch.cuda.synchronize()

            if batch_idx % 100 == 0:
                print(f"Batch {batch_idx} / {len(train_loader)} (Loss: {loss.item():.4f})")

        avg_loss = running_loss / len(train_loader)
        val_cer = cer_for_loader(model, val_loader, encoder, device)

        scheduler.step(val_cer)

        train_losses.append(avg_loss)
        val_cers.append(val_cer)

        print(f"\n[CRNN] Epoch {epoch+1}/{epochs}")
        print(f"Train Loss: {avg_loss:.4f}")
        print(f"Val CER: {val_cer:.4f}")

        # save last checkpoint
        torch.save({
            "epoch": epoch,
            "model": model.state_dict(),
            "optimizer": optimizer.state_dict(),
            "scheduler": scheduler.state_dict(),
            "best_val_cer": best_val_cer,
            "epochs_without_improvement": epochs_without_improvement,
            "train_losses": train_losses,
            "val_cers": val_cers
        }, ckpt_path)

        # plot after each epoch
        try:
            plot_training_curves(train_losses, val_cers, prefix="crnn")
        except Exception as e:
            print(f"[WARN] Plot update failed: {e}")

        # save best model
        if val_cer < best_val_cer:
            best_val_cer = val_cer
            epochs_without_improvement = 0
            torch.save(model.state_dict(), best_model_path)
            print("New best CRNN model saved.")
        else:
            epochs_without_improvement += 1

        # early stop
        if epochs_without_improvement >= patience:
            print("Early stopping triggered.")
            break

    try:
        plot_training_curves(train_losses, val_cers, prefix="crnn")
    except Exception as e:
        print(f"[WARN] Plot update failed: {e}")

    return model, train_losses, val_cers