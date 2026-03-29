import torch
from collections import defaultdict
import numpy as np

def greedy_decode(outputs, encoder):
    """
    Führt ein einfaches CTC-Greedy-Decoding durch.
    Entfernt Blank-Tokens (Index 0) und doppelte aufeinanderfolgende Zeichen
    und wandelt die Indexsequenz in Text um.
    """
    # outputs: (T, N, C)
    pred = torch.argmax(outputs, dim=2)  # (T, N)
    pred = pred.permute(1, 0)           # (N, T)

    decoded_texts = []
    for seq in pred:
        prev = None
        kept = []
        for idx in seq:
            idx = idx.item()
            if idx != 0 and idx != prev:
                kept.append(idx)
            prev = idx
        decoded_texts.append(encoder.decode(kept))
    return decoded_texts

# Verfahren für Abstand von vorhergesagtem Wort zum eigentlichen
def levenshtein_distance(s1, s2):
    """
    Berechnet die Levenshtein-Distanz zwischen zwei Strings.
    Misst die minimale Anzahl von Einfügungen, Löschungen und
    Ersetzungen zur Transformation von s1 in s2.
    """
    if len(s1) < len(s2):
        return levenshtein_distance(s2, s1)
    if len(s2) == 0:
        return len(s1)

    previous_row = list(range(len(s2) + 1))
    for i, c1 in enumerate(s1):
        current_row = [i + 1]
        for j, c2 in enumerate(s2):
            ins = previous_row[j + 1] + 1
            dele = current_row[j] + 1
            sub = previous_row[j] + (c1 != c2)
            current_row.append(min(ins, dele, sub))
        previous_row = current_row
    return previous_row[-1]

def cer_for_loader(model, loader, encoder, device):
    """
    Berechnet die Character Error Rate (CER) über einen gesamten DataLoader.
    CER = (Summe der Edit-Distanzen) / (Gesamtanzahl der Zeichen im Ground Truth).
    """
    model.eval()
    total_distance = 0
    total_chars = 0

    with torch.no_grad():
        for batch in loader:
            if batch is None:
                continue
            images, labels_batch, lengths = batch
            images = images.to(device)
            labels_batch = labels_batch.to(device)
            lengths = lengths.to(device)

            outputs = model(images).permute(1, 0, 2)  # (T, N, C)
            preds = greedy_decode(outputs, encoder)

            for i in range(len(preds)):
                # Richtige Wort (Label)
                gt = encoder.decode(labels_batch[i][:lengths[i]].tolist())
                # Vorhergesagtes Wort
                pr = preds[i]
                # Wie stark unterscheiden gt und pr
                # Einfügen, Löschen und ersetzen
                # Auto -> Ato (1)
                total_distance += levenshtein_distance(gt, pr)
                # Anteil der falschen Stellen
                total_chars += max(1, len(gt))

    return total_distance / total_chars if total_chars > 0 else 0.0

def compute_confusion_matrix(model, loader, encoder, device):
    """
    Erzeugt eine Zeichen-zu-Zeichen-Confusion-Matrix auf Basis
    eines Levenshtein-Alignments zwischen Ground Truth und Vorhersage.
    """
    model.eval()

    # Matrix: GT → Pred
    confusion = defaultdict(lambda: defaultdict(int))

    with torch.no_grad():
        for batch in loader:
            if batch is None:
                continue
            images, labels_batch, lengths = batch
            images = images.to(device)
            labels_batch = labels_batch.to(device)
            lengths = lengths.to(device)

            outputs = model(images).permute(1, 0, 2)
            preds = greedy_decode(outputs, encoder)

            for i in range(len(preds)):
                gt = encoder.decode(labels_batch[i][:lengths[i]].tolist())
                pr = preds[i]

                align_and_count(gt, pr, confusion)

    return confusion

def align_and_count(gt, pr, confusion):
    """
    Levenshtein Alignment mit Backtracking,
    damit wir wissen welches Zeichen mit welchem verglichen wurde.
    """

    m, n = len(gt), len(pr)
    dp = np.zeros((m+1, n+1), dtype=int)

    for i in range(m+1):
        dp[i][0] = i
    for j in range(n+1):
        dp[0][j] = j

    for i in range(1, m+1):
        for j in range(1, n+1):
            cost = 0 if gt[i-1] == pr[j-1] else 1
            dp[i][j] = min(
                dp[i-1][j] + 1,      # deletion
                dp[i][j-1] + 1,      # insertion
                dp[i-1][j-1] + cost  # substitution
            )

    # Backtracking
    i, j = m, n
    while i > 0 and j > 0:
        if gt[i-1] == pr[j-1]:
            confusion[gt[i-1]][pr[j-1]] += 1
            i -= 1
            j -= 1
        elif dp[i][j] == dp[i-1][j-1] + 1:
            confusion[gt[i-1]][pr[j-1]] += 1
            i -= 1
            j -= 1
        elif dp[i][j] == dp[i-1][j] + 1:
            confusion[gt[i-1]]["<del>"] += 1
            i -= 1
        else:
            confusion["<ins>"][pr[j-1]] += 1
            j -= 1

def word_accuracy_for_loader(model, loader, encoder, device):
    """
    Berechnet die Wortgenauigkeit (Word Accuracy) für ein CRNN-Modell.
    Ein Wort zählt als korrekt, wenn die gesamte Zeichenfolge exakt übereinstimmt.
    """
    model.eval()
    correct = 0
    total = 0

    with torch.no_grad():
        for images, labels_batch, lengths in loader:

            images = images.to(device)
            labels_batch = labels_batch.to(device)
            lengths = lengths.to(device)

            outputs = model(images).permute(1, 0, 2)
            preds = greedy_decode(outputs, encoder)

            for i in range(len(preds)):
                gt = encoder.decode(labels_batch[i][:lengths[i]].tolist())
                pr = preds[i]

                if gt == pr:
                    correct += 1

                total += 1

    return correct / total if total > 0 else 0.0

def resnet_cer_for_loader(model, loader, idx_to_word, device):

    """
    Berechnet die Character Error Rate (CER) für das ResNet-Modell,
    indem vorhergesagte Wortklassen in Strings umgewandelt und
    mittels Levenshtein-Distanz verglichen werden.
    """
    model.eval()

    total_distance = 0
    total_chars = 0

    with torch.no_grad():
        for images, targets in loader:
            images = images.to(device)
            targets = targets.to(device)

            outputs = model(images)
            preds = torch.argmax(outputs, dim=1)

            for i in range(len(preds)):
                gt_word = idx_to_word[targets[i].item()]
                pred_word = idx_to_word[preds[i].item()]

                total_distance += levenshtein_distance(gt_word, pred_word)
                total_chars += max(1, len(gt_word))

    return total_distance / total_chars

def evaluate_resnet(model, test_loader, device):
    """
    Berechnet die Wortgenauigkeit für das ResNet-Modell.
    Vergleicht vorhergesagte Wortklassen direkt mit den Zielklassen.
    """
    model.eval()
    correct = 0
    total = 0
    with torch.no_grad():
        for batch in test_loader:
            if batch is None: continue
            images, targets = batch
            images = images.to(device)
            targets = targets.to(device)
            outputs = model(images)
            preds = torch.argmax(outputs, dim=1)
            correct += (preds == targets).sum().item()
            total += targets.size(0)
    return correct / total if total > 0 else 0
