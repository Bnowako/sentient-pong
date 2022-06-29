import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import os


class QNet(nn.Module):
    def __init__(self, input_size, hidden_size, output_size):
        super().__init__()
        self.linear1 = nn.Linear(input_size, hidden_size)
        self.linear2 = nn.Linear(hidden_size, output_size)

    def forward(self, x):
        x = F.relu(self.linear1(x))
        x = self.linear2(x)
        return x
    #
    # def save(self):
    #     model_scripted = torch.jit.script(self)  # Export to TorchScript
    #     model_scripted.save('snake_ai_model.pt')

    def get_model(self):
        return torch.load(f'./model/model.pth')
