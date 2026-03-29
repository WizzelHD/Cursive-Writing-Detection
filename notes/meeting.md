# Meeting Protokolle – Projekt: Automatische Erkennung handgeschriebener Schreibschrift

---

# Meeting 1 – Projektstart & Konzept

**Teilnehmer:** Emre Temel, Erik Lang

## Fortschritt

- Projektidee finalisiert: Handschrift-Erkennung mit IAM Dataset
- Recherche zu CRNN + CTC-Loss durchgeführt
- IAM Dataset Struktur analysiert
- Datensatz heruntergeladen

## Probleme

- Unklarheit über Struktur der IAM words.txt
- Verständnis von CTC-Loss noch nicht vollständig

## Planung

- Parser für words.txt implementieren
- Dataset-Klasse für Word-Level erstellen
- Testen, ob Bilder korrekt geladen werden
- Nächstes Meeting: funktionierender DataLoader

---

# Meeting 2 – Datenpipeline

## Fortschritt

- IAM Parser implementiert
- Bildpfad-Generierung korrekt umgesetzt
- PyTorch Dataset + DataLoader erstellt
- Labels werden korrekt geladen
- Encoder für Zeichenmapping erstellt

## Probleme

- Fehlerhafte / beschädigte Bilder im IAM Dataset
- Verständnis für CTC Target Format (input_lengths vs target_lengths)
- (Adapter Problem zwischen unterschiedlichen Formaten, CTC-Loss, CNN, LSTM)

## Planung

- CRNN Architektur implementieren
- CTC-Loss integrieren
- Erster Trainingslauf mit wenigen Epochen
- Format Anpassung durch permute

---

# Meeting 3 – Erstes Training

## Fortschritt

- CRNN implementiert (CNN + BiLSTM)
- CTC-Loss korrekt integriert
- Training läuft stabil
- Erste Loss-Kurve erzeugt
- Greedy Decoding implementiert

## Probleme

- Anfangs sehr hohe CER
- Modell verwechselt ähnliche Zeichen (z. B. I und l)

## Planung

- Mehr Epochen trainieren
- Learning Rate anpassen
- Validation CER pro Epoche einbauen

---

# Meeting 4 – Optimierung & Auswertung

## Fortschritt

- Training auf 20 Epochen erweitert
- Learning Rate reduziert
- ReduceLROnPlateau Scheduler integriert
- Validation CER Tracking implementiert
- Trainingskurven gespeichert

## Probleme

- Epochen sehr ressourcenaufwändig
- Modell hat Schwierigkeiten bei Sonderzeichen
- Überanpassung ab ca. Epoche 14 beobachtet

## Planung

- Fehleranalyse implementieren
- Worst-Case Beispiele extrahieren
- Architektur für Präsentation aufbereiten
- Lösung finden für Ressourcen

---

# Meeting 5

## Fortschritt

- Fehleranalyse durchgeführt
- Worst Prediction Samples gespeichert
- Trainingskurven finalisiert
- Dokumentation / README überarbeitet
- Präsentationsstruktur vorbereitet
- Ressourcenlösung -> Checkpoints für jede Woche
- Final Test CER berechnet (~12%)

## Erkenntnisse

- CRNN + CTC funktioniert robust für Wortebene
- Schwächen bei:
  - Einzelzeichen
  - Satzzeichen
- Scheduler verbessert Generalisierung

## Wissenschaftliche Einordnung

- Architektur orientiert sich an Shi et al. (2016)
- Training mit CTC gemäß Graves et al. (2006)
- Evaluation mittels Character Error Rate (CER), Standardmetrik in OCR

## Planung

- App bauen (mobile?)

---

# Meeting 6

## Fortschritt

- Posterstruktur überarbeitet und inhaltlich aktualisiert
- Kernaussagen geschärft (CRNN vs. ResNet, CER/Accuracy, Fehleranalyse)
- Relevante Visualisierungen für das Poster ausgewählt
- README sprachlich und strukturell bereinigt

## Probleme

- Inhalte mussten für das Poster stärker verdichtet werden
- Fokus zwischen Methodik-Details und Ergebnisdarstellung musste ausbalanciert werden

## Planung

- App-Konzept konkretisieren (Zielplattform, Features, UI-Flow)
- Erste App-Version implementieren
- Modellinferenz in die App integrieren und testen

---

# Meeting 7

## Fortschritt

- Web-Demo fertiggestellt (Next.js Frontend + FastAPI Backend)
- Synthetischer Datensatz integriert (TRDG, 8 Handschrift-Fonts, 50.000 Bilder)
- Synthetische Daten werden persistent gespeichert und bei erneutem Start von Disk geladen
- ResNet wird nun ebenfalls mit synthetischen Daten trainiert
- Plot-Update nach jeder Epoche für CRNN nachgezogen (war nur am Ende)
- README aktualisiert: Abschnitt synthetischer Datensatz, combined_comparison.png, Ergebnisse vor/nach
- program_flow.txt als Präsentationsunterlage erstellt

## Erkenntnisse

- Synthetische Daten hatten nur marginalen Effekt durch Domain Gap (gerenderte Fonts vs. echte Handschrift)
- Web-Demo erkennt Handy-Fotos durch eigene IAM-style Preprocessing-Pipeline

## Planung

- Poster drucken

---
