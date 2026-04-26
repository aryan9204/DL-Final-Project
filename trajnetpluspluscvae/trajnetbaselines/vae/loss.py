"""Loss helpers for the CVAE model."""

import torch


class KLDLoss(torch.nn.Module):
    """KL divergence between diagonal Gaussian posterior and prior."""

    def __init__(self):
        super().__init__()

    def forward(self, posterior_mu, posterior_log_var, prior_mu=None, prior_log_var=None):
        if prior_mu is None:
            prior_mu = torch.zeros_like(posterior_mu)
        if prior_log_var is None:
            prior_log_var = torch.zeros_like(posterior_log_var)

        posterior_var = torch.exp(posterior_log_var)
        prior_var = torch.exp(prior_log_var)

        kl = prior_log_var - posterior_log_var
        kl += (posterior_var + (posterior_mu - prior_mu) ** 2) / prior_var
        kl -= 1.0
        kl = 0.5 * kl.sum(dim=1)
        return kl.mean()
