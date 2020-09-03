# -*- coding: utf-8 -*-
import torch.nn as nn
from torchsummary import summary


class BidirectionalLSTM(nn.Module):

    def __init__(self, input_size, hidden_size, output_size):
        super(BidirectionalLSTM, self).__init__()

        self.rnn = nn.LSTM(input_size, hidden_size, bidirectional=True)
        self.embedding = nn.Linear(hidden_size * 2, output_size)

    def forward(self, input):
        """
        Args:
            input: visual feature [seq_len x batch_size x input_size]
        Return: 
            output: contextual feature [seq_len x batch_size x output_size]
        """
        
        recurrent, _ = self.rnn(input)
        seq_len, batch, hidden_size = recurrent.size()
        t_rec = recurrent.view(seq_len*batch, hidden_size)

        output = self.embedding(t_rec)  # [seq_len * batch, output_size]
        output = output.view(seq_len, batch, -1)

        return output


class CRNN(nn.Module):

    def __init__(self, imgH, channels, nclass, hidden_state):
        super(CRNN, self).__init__()
        assert imgH % 16 == 0, 'Image height has to be a multiple of 16'

        cnn = nn.Sequential(
            nn.Conv2d(channels, 64, 3, 1, 1), nn.ReLU(True),
            nn.MaxPool2d(2, 2),  # 64x16x50
            nn.Conv2d(64, 128, 3, 1, 1), nn.ReLU(True),
            nn.MaxPool2d(2, 2), # 128x8x25
            nn.Conv2d(128, 256, 3, 1, 1), nn.ReLU(True),
            nn.Conv2d(256, 256, 3, 1, 1), nn.ReLU(True),
            nn.MaxPool2d((2, 1), (2, 1)),   # 256x4x25
            nn.Conv2d(256, 512, 3, 1, 1, bias=False),
            nn.BatchNorm2d(512), nn.ReLU(True),
            nn.Conv2d(512, 512, 3, 1, 1, bias=False),
            nn.BatchNorm2d(512), nn.ReLU(True),
            nn.MaxPool2d((2, 1), (2, 1)),   # 512x2x25
            nn.Conv2d(512, 512, 2, 1, 0), nn.ReLU(True) #512x1x24
            )

        self.cnn = cnn
        self.rnn = nn.Sequential(
            BidirectionalLSTM(512, hidden_state, hidden_state),
            BidirectionalLSTM(hidden_state, hidden_state, nclass))

    def forward(self, input):
        # feature extraction
        conv = self.cnn(input)
        b, c, h, w = conv.size()
        assert h == 1, "the height of conv must be 1"
        conv = conv.squeeze(2)
        conv = conv.permute(2, 0, 1)  # [w, b, c]

        # sequence modelings
        output = self.rnn(conv)

        return output
    
if __name__ == "__main__":
    model = CRNN(32, 1, 37, 256)
    print(model)
    summary(model, (1, 32, 100))