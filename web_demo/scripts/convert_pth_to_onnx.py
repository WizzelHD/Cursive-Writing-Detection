"""
Konvertiere deine PyTorch .pth Modelle zu .onnx Format
fuer die Verwendung im Browser mit ONNX Runtime Web.

Anleitung:
1. pip install torch torchvision onnx
2. Passe die Modell-Definitionen unten an (ResNet / CRNN)
3. python scripts/convert_pth_to_onnx.py

Die .onnx Dateien werden in public/models/ gespeichert.
Lade sie dann ueber die Weboberflaeche in den Browser.
"""

import os
import torch
import torch.onnx

# ============================================================
# PASSE DIESEN ABSCHNITT AN DEINE MODELLE AN
# ============================================================

def load_resnet_model(pth_path: str) -> torch.nn.Module:
    """
    Lade dein ResNet Modell.
    Passe dies an deine Modell-Architektur an.
    
    Beispiel mit torchvision:
        import torchvision.models as models
        model = models.resnet18(num_classes=ANZAHL_KLASSEN)
        model.load_state_dict(torch.load(pth_path, map_location="cpu"))
        return model
    """
    import torchvision.models as models

    # Beispiel: ResNet18 mit 1000 Klassen (ImageNet)
    # Aendere num_classes auf die Anzahl deiner Klassen
    model = models.resnet18(num_classes=1000)
    state_dict = torch.load(pth_path, map_location="cpu", weights_only=True)
    model.load_state_dict(state_dict)
    model.eval()
    return model


def load_crnn_model(pth_path: str) -> torch.nn.Module:
    """
    Lade dein CRNN Modell.
    Passe dies an deine Modell-Architektur an.
    
    Beispiel:
        from your_crnn_module import CRNN
        model = CRNN(imgH=32, nc=1, nclass=37, nh=256)
        model.load_state_dict(torch.load(pth_path, map_location="cpu"))
        return model
    """
    # ============================================
    # ERSETZE DIESEN BLOCK MIT DEINER CRNN KLASSE
    # ============================================
    # Beispiel einer einfachen CRNN Architektur:
    import torch.nn as nn

    class BidirectionalLSTM(nn.Module):
        def __init__(self, nIn, nHidden, nOut):
            super().__init__()
            self.rnn = nn.LSTM(nIn, nHidden, bidirectional=True)
            self.embedding = nn.Linear(nHidden * 2, nOut)

        def forward(self, input):
            recurrent, _ = self.rnn(input)
            T, b, h = recurrent.size()
            t_rec = recurrent.view(T * b, h)
            output = self.embedding(t_rec)
            output = output.view(T, b, -1)
            return output

    class CRNN(nn.Module):
        def __init__(self, imgH, nc, nclass, nh):
            super().__init__()
            ks = [3, 3, 3, 3, 3, 3, 2]
            ps = [1, 1, 1, 1, 1, 1, 0]
            ss = [1, 1, 1, 1, 1, 1, 1]
            nm = [64, 128, 256, 256, 512, 512, 512]
            cnn = nn.Sequential()
            def convRelu(i, batchNormalization=False):
                nIn = nc if i == 0 else nm[i - 1]
                nOut = nm[i]
                cnn.add_module(f"conv{i}", nn.Conv2d(nIn, nOut, ks[i], ss[i], ps[i]))
                if batchNormalization:
                    cnn.add_module(f"batchnorm{i}", nn.BatchNorm2d(nOut))
                cnn.add_module(f"relu{i}", nn.ReLU(True))
            convRelu(0)
            cnn.add_module("pooling0", nn.MaxPool2d(2, 2))
            convRelu(1)
            cnn.add_module("pooling1", nn.MaxPool2d(2, 2))
            convRelu(2, True)
            convRelu(3)
            cnn.add_module("pooling2", nn.MaxPool2d((2, 2), (2, 1), (0, 1)))
            convRelu(4, True)
            convRelu(5)
            cnn.add_module("pooling3", nn.MaxPool2d((2, 2), (2, 1), (0, 1)))
            convRelu(6, True)
            self.cnn = cnn
            self.rnn = nn.Sequential(
                BidirectionalLSTM(512, nh, nh),
                BidirectionalLSTM(nh, nh, nclass),
            )

        def forward(self, input):
            conv = self.cnn(input)
            b, c, h, w = conv.size()
            assert h == 1, f"Height must be 1, got {h}"
            conv = conv.squeeze(2)
            conv = conv.permute(2, 0, 1)
            output = self.rnn(conv)
            return output

    # nclass=37: 0-9 + a-z + blank(CTC)
    model = CRNN(imgH=32, nc=1, nclass=37, nh=256)
    state_dict = torch.load(pth_path, map_location="cpu", weights_only=True)
    model.load_state_dict(state_dict)
    model.eval()
    return model


# ============================================================
# KONVERTIERUNG - Normalerweise musst du hier nichts aendern
# ============================================================

def convert_resnet(pth_path: str, output_path: str):
    print(f"Lade ResNet von {pth_path}...")
    model = load_resnet_model(pth_path)

    # ResNet erwartet 224x224 RGB Bilder
    dummy_input = torch.randn(1, 3, 224, 224)

    print(f"Exportiere nach {output_path}...")
    torch.onnx.export(
        model,
        dummy_input,
        output_path,
        export_params=True,
        opset_version=11,
        do_constant_folding=True,
        input_names=["input"],
        output_names=["output"],
        dynamic_axes={
            "input": {0: "batch_size"},
            "output": {0: "batch_size"},
        },
    )
    print(f"ResNet ONNX gespeichert: {output_path}")


def convert_crnn(pth_path: str, output_path: str):
    print(f"Lade CRNN von {pth_path}...")
    model = load_crnn_model(pth_path)

    # CRNN erwartet 32x100 Graustufenbilder (HxW)
    dummy_input = torch.randn(1, 1, 32, 100)

    print(f"Exportiere nach {output_path}...")
    torch.onnx.export(
        model,
        dummy_input,
        output_path,
        export_params=True,
        opset_version=11,
        do_constant_folding=True,
        input_names=["input"],
        output_names=["output"],
        dynamic_axes={
            "input": {0: "batch_size"},
            "output": {1: "batch_size"},
        },
    )
    print(f"CRNN ONNX gespeichert: {output_path}")


if __name__ == "__main__":
    # Pfade zu deinen .pth Dateien (passe diese an!)
    RESNET_PTH = "models/resnet.pth"
    CRNN_PTH = "models/crnn.pth"

    # Output-Verzeichnis
    os.makedirs("public/models", exist_ok=True)

    if os.path.exists(RESNET_PTH):
        convert_resnet(RESNET_PTH, "public/models/resnet.onnx")
    else:
        print(f"WARNUNG: {RESNET_PTH} nicht gefunden. Ueberspringe ResNet.")

    if os.path.exists(CRNN_PTH):
        convert_crnn(CRNN_PTH, "public/models/crnn.onnx")
    else:
        print(f"WARNUNG: {CRNN_PTH} nicht gefunden. Ueberspringe CRNN.")

    print("\nFertig! Lade die .onnx Dateien ueber die Weboberflaeche in den Browser.")
