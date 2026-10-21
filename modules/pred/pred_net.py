import torch
import torch.optim as optim

from torch import nn
from torch.nn import functional as F


class PredNetwork(nn.Module):
    def __init__(self,args, num_inputs, hidden_dim, num_outputs, device, lr=3e-4):
        super(PredNetwork, self).__init__()

        def weights_init_(m):
            if isinstance(m, nn.Linear):
                torch.nn.init.xavier_uniform_(m.weight, gain=1)
                torch.nn.init.constant_(m.bias, 0)

        self.device = device
        self.hideen_dim = hidden_dim
        self.linear1 = nn.Linear(num_inputs, hidden_dim)
        self.linear2 = nn.Linear(hidden_dim, hidden_dim)
        self.linear3 = nn.Linear(hidden_dim, num_outputs)
        self.args=args
        self.apply(weights_init_)
        self.lr = lr

        self.optimizer = optim.Adam(self.parameters(), lr=self.lr)

    def forward(self, input):
        h1 = F.relu(self.linear1(input))
        h2 = F.relu(self.linear2(h1))
        return self.linear3(h2)

    def log_p(self, input, outcome):
        with torch.no_grad():
            mean = self.forward(input)
            if self.args.log_likelihood == 'abs':
                return -(mean-outcome).abs()
            else:
                return -nn.MSELoss(reduction='none')(mean, outcome)

    def update(self, input, output, mask):
        if mask.sum() > 0:
            loss = F.mse_loss(self.forward(input), output, reduction='none')
            loss = loss.sum(dim=-1, keepdim=True)
            loss = (loss * mask).sum() / mask.sum()

            self.optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(self.parameters(), 1.)
            self.optimizer.step()

            return loss.to('cpu').detach().item()

        return None