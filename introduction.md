# Maschinelles Sehen – Projektabgabe

**Repository:** https://gitlab.bht-berlin.de/medieninformatik-master-se1/se2/machinelles-sehen-projekt

---

## Inhalt der README.md

| Abschnitt                                                    | Inhalt                  |
| ------------------------------------------------------------ | ----------------------- |
| Projektübersicht & Problemstellung                           | Projektidee             |
| Verwendeter Datensatz & Synthetischer Datensatz              | Datensatz               |
| Related Work                                                 | Literatur & Quellen     |
| Methodik, Architektur, Training Setup                        | Vorgehen                |
| Ergebnisse, Trainingsverlauf, Modellvergleich, Fehleranalyse | Ergebnisse & Auswertung |
| Poster                                                       | Poster                  |

---

## Wichtige Dateien

```
machinelles-sehen-projekt/
│
├── README.md                               ← Hauptdokumentation
├── requirements.txt                        ← Abhängigkeiten
│
├── presentation/
│   └── Poster.pdf                          ← Poster
│
├── graphs/                                 ← Ergebnis-Grafiken (Plots, Heatmaps)
│
├── code/
│   ├── main.py                             ← Pipeline-Einstiegspunkt
│   ├── metrics.py                          ← CER, Accuracy, Levenshtein
│   ├── evaluation.py                       ← Confusion-Analyse & Report
│   ├── plotting.py                         ← Trainingsplots
│   ├── encoder/
│   │   ├── crnn.py                         ← CRNN-Architektur
│   │   └── resnet.py                       ← ResNet-18-Architektur
│   ├── training/
│   │   ├── train_crnn.py                   ← Trainingslogik CRNN
│   │   └── train_resnet.py                 ← Trainingslogik ResNet-18
│   └── data_code/
│       ├── iam_parser.py                   ← IAM-Datensatz Parser
│       ├── dataset.py                      ← PyTorch Dataset (IAM)
│       ├── label_encode.py                 ← Zeichenalphabet & Encoding
│       ├── synthetic_dataset.py            ← Synthetisches Dataset
│       ├── generate_synthetic.py           ← Datengenerierung (TRDG)
│       └── word_classification_dataset.py  ← Dataset für ResNet-18
│
└── web_demo/W
    ├── backend/
    │   └── server.py                       ← FastAPI Inference-Backend
    └── app/                                ← Next.js Frontend
```

---

Autoren: Emre Temel, Erik Lang
