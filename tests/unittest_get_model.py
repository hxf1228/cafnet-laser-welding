import torch
from src.models.get_model import get_model

model = torch.nn.DataParallel(get_model('cnn1d').cuda())
