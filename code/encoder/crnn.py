import torch.nn as nn


class CRNN(nn.Module):
    def __init__(self, num_classes):
        super(CRNN, self).__init__()

        # CNN 
        self.cnn = nn.Sequential(
            nn.Conv2d(1, 64, 3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.MaxPool2d(2, 2),

            nn.Conv2d(64, 128, 3, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(),
            nn.MaxPool2d(2, 2),

            nn.Conv2d(128, 256, 3, padding=1),
            nn.BatchNorm2d(256),
            nn.ReLU(),

            nn.Conv2d(256, 256, 3, padding=1),
            nn.BatchNorm2d(256),
            nn.ReLU(),
        )

        # nach 2x Pooling:
        # Input: (1, 32, 128)
        # -> (256, 8, 32)

        self.rnn = nn.LSTM(
            input_size=256 * 8,
            hidden_size=512,
            num_layers=2,
            bidirectional=True,
            batch_first=True,
            dropout=0.3
        )

        self.fc = nn.Linear(512 * 2, num_classes)
        self.log_softmax = nn.LogSoftmax(dim=2)

    def forward(self, x):
        x = self.cnn(x)

        batch, channels, height, width = x.size()

        x = x.permute(0, 3, 1, 2)
        x = x.contiguous().view(batch, width, channels * height)

        x, _ = self.rnn(x)

        x = self.fc(x)
        x = self.log_softmax(x)

        return x