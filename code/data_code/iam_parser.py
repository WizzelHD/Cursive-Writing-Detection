def parse_iam_words(ascii_path: str) -> dict:
    """
    Parses IAM words.txt and returns a dictionary:
    {image_id: transcription}
    """

    labels = {}

    with open(ascii_path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()

            # skip comments and empty lines
            if not line or line.startswith("#"):
                continue

            parts = line.split()

            image_id = parts[0]
            status = parts[1]

            # use correctly segmented words
            if status != "ok":
                continue

            # 8th column
            transcription = " ".join(parts[8:])

            labels[image_id] = transcription

    return labels
