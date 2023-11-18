import torch.nn as nn

__all__ = ["CNN1D"]


class CNN1D(nn.Module):
    def __init__(self):
        super(CNN1D, self).__init__()
        self.conv = nn.Sequential(
            nn.Conv1d(1, 32, 3, 1, padding="same"),
            nn.BatchNorm1d(32),
            nn.LeakyReLU(inplace=True),
            nn.MaxPool1d(2, 1),
            nn.Conv1d(32, 64, 3, 1, padding="same"),
            nn.BatchNorm1d(64),
            nn.LeakyReLU(inplace=True),
            nn.MaxPool1d(2, 2),
            nn.Conv1d(64, 128, 3, 1, padding="same"),
            nn.BatchNorm1d(128),
            nn.LeakyReLU(inplace=True),
            nn.MaxPool1d(2, 1),
            nn.Conv1d(128, 64, 3, 1, padding="same"),
            nn.BatchNorm1d(64),
            nn.LeakyReLU(inplace=True),
            nn.MaxPool1d(2, 1),
        )
        self.classifier_1 = nn.Sequential(nn.Linear(64, 3), )
        self.last_feature = None

    def forward(self, x):
        x = self.conv(x)
        x = x.mean([-1])
        self.last_feature = x.view(x.size(0), -1)
        x = x.view(x.size(0), -1)
        x = self.classifier_1(x)
        return x