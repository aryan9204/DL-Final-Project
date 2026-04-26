from importlib.util import module_from_spec, spec_from_file_location
from pathlib import Path

import torch


LOSS_PATH = Path(__file__).resolve().parents[1] / 'trajnetbaselines' / 'vae' / 'loss.py'
SPEC = spec_from_file_location('trajnetpluspluscvae_vae_loss', LOSS_PATH)
LOSS_MODULE = module_from_spec(SPEC)
SPEC.loader.exec_module(LOSS_MODULE)
KLDLoss = LOSS_MODULE.KLDLoss


def test_kld_zero_for_matching_standard_normals():
    criterion = KLDLoss()
    mu = torch.zeros(3, 4)
    log_var = torch.zeros(3, 4)

    loss = criterion(mu, log_var, mu, log_var)

    assert torch.isclose(loss, torch.tensor(0.0))


def test_kld_positive_for_shifted_posterior():
    criterion = KLDLoss()
    posterior_mu = torch.ones(2, 3)
    posterior_log_var = torch.zeros(2, 3)
    prior_mu = torch.zeros(2, 3)
    prior_log_var = torch.zeros(2, 3)

    loss = criterion(posterior_mu, posterior_log_var, prior_mu, prior_log_var)

    assert loss > 0
