import torch.nn as nn
from src.set_transformer import SetTransformer
from torch import cat


class HPONetwork(nn.Module):
    def __init__(self, dim_input, num_outputs,
                 dim_output, emb_output=1, ln=False):
        super(HPONetwork, self).__init__()
        self.embedding = nn.Sequential(
            SetTransformer(dim_input, num_outputs, emb_output),
            nn.Flatten(),
        )
        self.linear1 = nn.Linear(dim_input+num_outputs, dim_output)

    def forward(self, x_bar, improve, c):
        improve = improve.unsqueeze(1)
        out = self.embedding(c)
        out = cat((x_bar, improve, out), dim=1)
        out = self.linear1(out)

        return out
