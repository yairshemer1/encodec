import torch
from torch import nn


class DiscriminatorAdversarialLoss(nn.Module):
    def __init__(self):
        super().__init__()

    def discriminator_loss(self, y_r, y_gen):
        return torch.mean(torch.relu(1 - y_r) + torch.relu(1 + y_gen))

    def forward(self, discs_y_r, discs_y_gen):
        return torch.mean(torch.stack(([self.discriminator_loss(y_r, y_gen) for y_r, y_gen in zip(discs_y_r, discs_y_gen)])))


class GeneratorAdversarialLoss(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, y_disc_gen):
        loss_per_discriminator = torch.stack(([torch.mean(torch.clamp(1-subdisc_y, min=0)) for subdisc_y in y_disc_gen]))
        return torch.mean(loss_per_discriminator)


class FeatureMatchingLoss(nn.Module):
    def __init__(self):
        super().__init__()

    def layer_feature_matching_loss(self, layer_fmap_r, layer_fmap_gen):
        return torch.abs(layer_fmap_r - layer_fmap_gen).mean() / torch.clamp(torch.abs(layer_fmap_r).mean(), min=1e-2)

    def feature_matching_loss_per_layer(self, disc_fmap_r, disc_fmap_gen):
        return torch.stack(([self.layer_feature_matching_loss(lfm_r.detach(), lfm_gen)
                          for lfm_r, lfm_gen in zip(disc_fmap_r, disc_fmap_gen)]))

    def feature_matching_loss_per_disc(self, fmap_r, fmap_gen):
        return torch.stack(([self.feature_matching_loss_per_layer(disc_fmap_r, disc_fmap_gen)
                          for disc_fmap_r, disc_fmap_gen in zip(fmap_r, fmap_gen)]))

    def forward(self, fmap_r, fmap_gen):
        return torch.mean(self.feature_matching_loss_per_disc(fmap_r, fmap_gen))


def loss_form_sanity_check(loss_tensor):
    assert loss_tensor.dim() == 0 and loss_tensor > 0
    pass
