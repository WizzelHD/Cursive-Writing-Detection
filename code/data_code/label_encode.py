class LabelEncoder:
    """
    Kodiert und dekodiert Zeichenfolgen für das CRNN-Modell.

    Implementiert eine Zuordnung:
    - Zeichen → Index (für Training)
    - Index → Zeichen (für Decoding)

    Wichtige Eigenschaft:
    - Index 0 ist für das CTC-Blank-Label reserviert.
    """
    def __init__(self, characters):
        self.characters = characters
        self.char_to_idx = {char: idx + 1 for idx, char in enumerate(characters)}
        self.idx_to_char = {idx + 1: char for idx, char in enumerate(characters)}
        self.blank = 0

    def encode(self, text):
        return [self.char_to_idx[char] for char in text]

    def decode(self, indices):
        return "".join(
            [self.idx_to_char[i] for i in indices if i != self.blank]
        )
