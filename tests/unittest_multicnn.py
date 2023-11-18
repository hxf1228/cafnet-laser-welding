from src.models.multicnn import MultiCNN
import torch

if __name__ == '__main__':
    input1 = torch.randn(32, 1, 1024)
    input2 = torch.randn(32, 1, 1024)
    model = MultiCNN()
    prediction = model(input1, input2)
    print(prediction.shape)
