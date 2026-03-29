import matplotlib.pyplot as plt
import numpy as np
import os
import matplotlib.colors as mcolors

def plot_training_curves(losses, metrics, save_dir="../graphs", prefix="model"):
    """
    Erstellt getrennte Trainingskurven für Loss und Metriken.
    prefix: 'crnn' oder 'resnet' sorgt dafür, dass nichts überschrieben wird.
    """
    os.makedirs(save_dir, exist_ok=True)
    
    # Dynamische Beschriftung je nach Modell-Typ
    if prefix.lower() == "crnn":
        metric_label = "Validation CER (%)"
        title_suffix = "CRNN"
        train_label = "Train Loss"
        train_ylabel = "Loss"
        train_title = f"Training Loss - {title_suffix}"
    elif prefix.lower() == "resnet":
        metric_label = "Validation Error (1-Acc) (%)"
        title_suffix = "ResNet"
        train_label = "Train Error (1-Acc)"
        train_ylabel = "Error Rate"
        train_title = f"Training Error (1-Acc) - {title_suffix}"
    else:
        raise ValueError(f"Unsupported prefix in plot_training_curves: {prefix}")       

    # 1. Plot für den Loss
    plt.figure(figsize=(8, 6))
    plt.plot(range(1, len(losses) + 1), losses, label=train_label, color='tab:blue', linewidth=2)
    plt.xlabel("Epoch")
    plt.ylabel(train_ylabel)
    plt.title(train_title)
    plt.legend()
    plt.grid(True, linestyle='--', alpha=0.7)
    plt.tight_layout()
    plt.savefig(os.path.join(save_dir, f"{prefix}_train_loss.png"), dpi=200)
    plt.close()

    # 2. Plot für die Metrik (CER oder Error Rate)
    metrics_pct = np.asarray(metrics) * 100.0
    plt.figure(figsize=(8, 6))
    plt.plot(range(1, len(metrics_pct) + 1), metrics_pct, label=metric_label, color='tab:orange', linewidth=2)
    plt.xlabel("Epoch")
    plt.ylabel(metric_label)
    plt.title(f"{metric_label} Progress - {title_suffix}")
    plt.legend()
    plt.grid(True, linestyle='--', alpha=0.7)
    plt.tight_layout()
    plt.savefig(os.path.join(save_dir, f"{prefix}_val_metric.png"), dpi=200)
    plt.close()

    print(f"Saved {prefix} training curves to {save_dir}")

def plot_confusion_heatmap(confusion_list, top_k=10, prefix="crnn", save_dir="../graphs"):
    """
    Erstellt eine Heatmap der häufigsten Verwechslungen.
    """
    os.makedirs(save_dir, exist_ok=True)
    save_path = os.path.join(save_dir, f"{prefix}_top_confusions_heatmap.png")
    
    top = confusion_list[:top_k]
    gt_chars = sorted(set(gt for _, gt, _ in top))
    pred_chars = sorted(set(pred for _, _, pred in top))

    gt_index = {c: i for i, c in enumerate(gt_chars)}
    pred_index = {c: i for i, c in enumerate(pred_chars)}

    matrix = np.zeros((len(gt_chars), len(pred_chars)))
    for count, gt_char, pred_char in top:
        matrix[gt_index[gt_char], pred_index[pred_char]] = count

    plt.figure(figsize=(10, 8))
    im = plt.imshow(matrix, cmap="magma", aspect="auto")
    plt.colorbar(im, label="Count")

    plt.xticks(range(len(pred_chars)), ["∅" if c == "<del>" else c for c in pred_chars], rotation=45)
    plt.yticks(range(len(gt_chars)), gt_chars)

    plt.xlabel("Predicted")
    plt.ylabel("Ground Truth")
    plt.title(f"Top {top_k} {prefix.upper()} Confusions")

    norm = mcolors.Normalize(vmin=matrix.min(), vmax=matrix.max())
    cmap = plt.get_cmap("magma")

    for i in range(len(gt_chars)):
        for j in range(len(pred_chars)):
            value = matrix[i, j]
            if value > 0:
                rgba = cmap(norm(value))
                luminance = 0.3*rgba[0] + 0.6*rgba[1] + 0.1*rgba[2]
                text_color = "black" if luminance > 0.5 else "white"

                plt.text(j, i, str(int(value)),
                        ha="center", va="center",
                        color=text_color,
                        fontsize=8)

    plt.tight_layout()
    plt.savefig(save_path, dpi=300)
    plt.close()
    print(f"Saved {prefix} confusion heatmap to {save_path}")

def plot_model_comparison(crnn_cer, resnet_acc, save_path="../graphs/model_comparison.png"):
    """
    Vergleicht CRNN (CER) mit ResNet (Error Rate).
    """
    crnn_error = crnn_cer * 100
    resnet_error = (1 - resnet_acc) * 100

    models = ["CRNN (CER)", "ResNet (Word Error)"]
    errors = [crnn_error, resnet_error]

    plt.figure(figsize=(8, 6))
    plt.bar(models, errors, color=['skyblue', 'salmon'])

    plt.ylabel("Error Rate (%)")
    plt.title("Final Model Comparison")

    for i, v in enumerate(errors):
        plt.text(i, v + 0.5, f"{v:.2f}%", ha="center", fontweight='bold')

    plt.ylim(0, max(errors) * 1.3)
    plt.grid(axis='y', linestyle='--', alpha=0.7)
    plt.tight_layout()
    plt.savefig(save_path, dpi=300)
    plt.close()
    print(f"Saved final comparison plot to {save_path}")

def plot_top_confusions(confusion_list, top_k=10, prefix="crnn", save_dir="../graphs"):
    """
    Balkendiagramm der häufigsten Verwechslungen.
    """
    os.makedirs(save_dir, exist_ok=True)
    save_path = os.path.join(save_dir, f"{prefix}_top_confusions_bar.png")

    top = confusion_list[:top_k]
    labels = [f"{gt}→{('∅' if pr=='<del>' else pr)}" for count, gt, pr in top]
    counts = [count for count, gt, pr in top]

    x = np.arange(len(labels))

    width = max(12, top_k * 0.2)
    plt.figure(figsize=(width, 8))

    plt.bar(x, counts, color='teal', width=0.9)

    plt.xticks(x, labels, rotation=90, fontsize=7)
    plt.xlim(-0.5, len(labels) - 0.5)

    plt.xlabel("Ground Truth → Prediction")
    plt.ylabel("Count")
    plt.title(f"Top {top_k} Confusions ({prefix.upper()})")

    plt.grid(axis='y', linestyle='--', alpha=0.6)
    plt.tight_layout()
    plt.savefig(save_path, dpi=300)
    plt.close()

    print(f"Saved {prefix} confusion bar chart to {save_path}")

def plot_accuracy_comparison(crnn_word_acc, resnet_word_acc, save_path="../graphs/accuracy_comparison.png"):
    """
    Vergleicht CRNN und ResNet anhand der Word Accuracy auf dem Testset.
    Erwartet Accuracy als Wert zwischen 0 und 1.
    """
    os.makedirs(os.path.dirname(save_path), exist_ok=True)

    crnn_acc_pct = crnn_word_acc * 100
    resnet_acc_pct = resnet_word_acc * 100

    models = ["CRNN (Word Acc)", "ResNet (Word Acc)"]
    accs = [crnn_acc_pct, resnet_acc_pct]

    plt.figure(figsize=(8, 6))
    plt.bar(models, accs, color=["skyblue", "salmon"])

    plt.ylabel("Word Accuracy (%)")
    plt.title("Final Model Comparison")

    for i, v in enumerate(accs):
        plt.text(i, v + 0.5, f"{v:.2f}%", ha="center", fontweight="bold")

    # Wichtig: gleiche Skala, damit es nicht “verzerrt” wirkt
    plt.ylim(0, 100)
    plt.grid(axis="y", linestyle="--", alpha=0.7)
    plt.tight_layout()
    plt.savefig(save_path, dpi=300)
    plt.close()

    print(f"Saved accuracy comparison plot to {save_path}")
