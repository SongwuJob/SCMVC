import torch
import torch.nn as nn

class ContrastiveLoss(nn.Module):
    def __init__(self, batch_size, temperature, device):
        super(ContrastiveLoss, self).__init__()
        self.batch_size = batch_size
        self.temperature = temperature
        self.device = device

    def forward(self, h_i, h_j, weight=None):
        N =self.batch_size
        similarity_matrix = torch.matmul(h_i, h_j.T) / self.temperature
        positives = torch.diag(similarity_matrix)
        mask = torch.ones((N, N)).to(self.device)
        mask = mask.fill_diagonal_(0)

        nominator = torch.exp(positives)
        denominator = (mask.bool()) * torch.exp(similarity_matrix)

        loss_partial = -torch.log(nominator / torch.sum(denominator, dim=1))
        loss = torch.sum(loss_partial) / N
        loss = weight * loss if weight is not None else loss

        return loss
