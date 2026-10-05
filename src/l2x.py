
# Se define una clase que se realiza la selección de características en base a lo expuesto en el paper L2X

import torch
import torch.nn as nn

class Sample_Concrete(nn.Module):
    """
    Class for sample Concrete / Gumbel-Softmax variables.
    """
    def __init__(self, k, tau0 = 0.5):
        super(Sample_Concrete, self).__init__()
        self.tau0 = tau0
        self.k = k

    def forward(self, logits):
        # logits: [BATCH_SIZE, d]
        shape = logits.size()
        logits = logits.view(logits.size(0), -1)
        logits_ = logits.view(logits.size(0), 1, -1) # [BATCH_SIZE, 1, d]

        uniform = torch.rand((logits_.size(0), self.k, logits_.size(2)), device = logits_.device)
        uniform = uniform.clip(torch.finfo(torch.float32).tiny, 1.0)

        gumbel = - torch.log(- torch.log(uniform))
        noisy_logits = (gumbel + logits_) / self.tau0
        samples = torch.softmax(noisy_logits, dim=-1)
        samples, _ = torch.max(samples, dim=1)

        # Explanation Stage output.
        threshold = torch.topk(logits, self.k, dim=-1, sorted=True)[0][:,-1].view(-1, 1)
        discrete_logits = (logits >= threshold).float()

        return samples.view(shape) if self.training else discrete_logits.view(shape)
