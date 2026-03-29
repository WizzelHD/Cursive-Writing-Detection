from metrics import compute_confusion_matrix
import plotting as pt
from metrics import (
    cer_for_loader,
    word_accuracy_for_loader,
    evaluate_resnet,
    resnet_cer_for_loader
)
from plotting import plot_model_comparison, plot_accuracy_comparison


def run_confusion_analysis(model, loader, encoder, device):
    """
    Führt eine vollständige Zeichen-Fehleranalyse durch.
    Berechnet die Confusion-Matrix mittels Levenshtein-Alignment,
    gibt die häufigsten Verwechslungen aus und erzeugt
    entsprechende Visualisierungen (Bar-Plot und Heatmap).
    """
    print("\nComputing Character Confusion Matrix...")

    confusion = compute_confusion_matrix(model, loader, encoder, device)

    confusion_list = []

    for gt_char in confusion:
        for pred_char in confusion[gt_char]:
            if gt_char != pred_char:
                confusion_list.append(
                    (confusion[gt_char][pred_char], gt_char, pred_char)
                )

    confusion_list.sort(reverse=True)

    pt.plot_top_confusions(confusion_list, top_k=50)
    pt.plot_confusion_heatmap(confusion_list, top_k=100)


def run_final_report(
    crnn_model,
    resnet_model,
    test_loader,
    resnet_test_loader,
    encoder,
    idx_to_word,
    device
):
    """
    Führt die finale Evaluation beider Modelle durch.
    Berechnet Word Accuracy und Character Error Rate (CER)
    für CRNN und ResNet, gibt die Ergebnisse aus und
    erstellt einen direkten Modellvergleich.
    """

    print("\n========== FINAL EVALUATION ==========\n")

    # CRNN Metrics
    crnn_cer = cer_for_loader(crnn_model, test_loader, encoder, device)
    crnn_word_acc = word_accuracy_for_loader(crnn_model, test_loader, encoder, device)

    print(f"CRNN Word Accuracy: {crnn_word_acc*100:.2f}%")
    print(f"CRNN CER: {crnn_cer*100:.2f}%\n")

    # ResNet Metrics
    resnet_word_acc = evaluate_resnet(
        resnet_model,
        resnet_test_loader,
        device
    )

    resnet_cer = resnet_cer_for_loader(
        resnet_model,
        resnet_test_loader,
        idx_to_word,
        device
    )

    print(f"ResNet Word Accuracy: {resnet_word_acc*100:.2f}%")
    print(f"ResNet CER: {resnet_cer*100:.2f}%")


    # Plot comparison
    plot_model_comparison(crnn_cer, resnet_word_acc)
    plot_accuracy_comparison(crnn_word_acc, resnet_word_acc)

    return {
        "crnn_word_acc": crnn_word_acc,
        "crnn_cer": crnn_cer,
        "resnet_word_acc": resnet_word_acc,
        "resnet_cer": resnet_cer
    }
