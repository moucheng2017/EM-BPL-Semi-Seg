import math
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch
from torch import nn
from torch.autograd import Variable
from torch import einsum
import numpy as np
from scipy.ndimage import distance_transform_edt

################################################################################################
# A generalised and flexible k-l loss between
# By MCX
# manual:
# 1. by setting up all flags as 0, this kld_loss learns kl distance between two arbitrary Gaussians
# 2. if you want to double confirm, you can try set up mu2 (prior) as 0 and std2 (prior) as 1,
# then you get exactly the same implementation of the kl loss from the original VAE
#################################################################################################


def kld_loss(raw_output,
             mu1,
             logvar1,
             mu2=0.5,
             std2=0.125,
             flag_mu1=0,
             flag_std1=0,
             flag_mu2=0,
             flag_std2=1):
    '''
    Args:
        raw_output: raw digits output of predictions
        mu1: mean of posterior
        logvar1: log variance of posterior
        mu2: mean of prior
        std2: standard deviation of prior
        flag_mu1: flag for mean of post, 0: directly learn mean of posterior, 1: estimate the mean of posterior from raw output which is existing prediction confidence
        flag_std1: flag for std of post, 0: directly learn log var of posterior; 1: estimate the standard deviation of posterior from the mean of posterior, hard constraint, given than we know threshold need to be in 0 and 1; 2: use standard deviation of the prior directly
        flag_mu2: flag for mean of prior, 0: use the predefined mean of prior; 1: dynamic mean of prior, estimated from raw output which is based on existing prediction confidence
        flag_std2: flag for sd of prior, 0: use predefined standard deviation of prior; 1: estimate the standard deviation of prior from the mean of the prior

    Returns:
        loss
        predicted threshold for binary pseudo labelling
    '''

    gamma = 2. # number of sigmas

    if flag_mu1 == 0:
        # learn the mean of posterior, separately
        mu1 = F.relu(mu1, inplace=True)
    elif flag_mu1 == 1:
        # learn the mean of posterior, from current predictions
        mu1 = torch.sigmoid(raw_output)
        mu1 = mu1.mean()
    elif flag_mu1 == 2:
        mu1 = mu2
    else:
        raise NotImplementedError

    if flag_std1 == 0:
        # learn the variance of posterior
        log_sigma1 = 0.5*logvar1
        var1 = torch.exp(logvar1)
    elif flag_std1 == 1:
        # DO NOT learn the posterior variance, direct estimation from posterior mean
        std_upper = (1 - mu1) / gamma  # mean + 2*sigma <= 1.0
        std_lower = (mu1 - 0.0) / gamma  # mean - 2*sigma >= 0.0
        sigma1 = min(std_lower, std_upper)
        var1 = sigma1**2
        log_sigma1 = 0.5*math.log(var1)
    elif flag_std1 == 2:
        # DO NOT learn the posterior variance, use the prior variance
        var1 = std2**2
        log_sigma1 = 0.5*math.log(var1)
    else:
        raise NotImplementedError

    if flag_mu2 == 0:
        # mean of prior
        mu2 = mu2
    elif flag_mu2 == 1:
        # dynamic mean of prior, according to the current predictions
        mu2 = torch.sigmoid(raw_output)
        mu2 = mu2.mean()
    else:
        raise NotImplementedError

    if flag_std2 == 0:
        # standard deviation of prior
        sigma2 = std2
    elif flag_std2 == 1:
        # estimation of standard deviation of prior from mean of prior
        prior_std_upper = (1 - mu2) / gamma  # mean + 2*sigma <= 1.0
        prior_std_lower = (mu2 - 0.0) / gamma  # mean - 2*sigma >= 0.0
        sigma2 = min(prior_std_lower, prior_std_upper)
    else:
        raise NotImplementedError

    var2 = sigma2**2
    log_sigma2 = math.log(sigma2)

    loss = log_sigma2 - log_sigma1 + 0.5 * (var1 + (mu1 - mu2)**2) / var2 - 0.5
    loss = torch.mean(torch.sum(loss, dim=-1), dim=0)

    std = torch.exp(0.5 * logvar1)
    eps = torch.randn_like(std)
    threshold = eps * std + mu1

    if threshold.mean() < (mu2 - gamma * sigma2) or threshold.mean() > (mu2 + gamma * sigma2):
        threshold = mu2 * torch.ones_like(logvar1).cuda()

    return loss, threshold.mean()


if __name__ == "__main__":
    raw_output = torch.rand(4, 2, 128, 128, 128).cuda()

    mu = F.relu(torch.rand(4, 128)).cuda()
    logvar = torch.rand(4, 128).cuda()

    mu_prior = 0
    sigma_prior = 1

    # If all flags are zero, our implementation becoems the standard KL loss popular in existing VAE with N(0, 1) prior:
    loss, t = kld_loss(raw_output=raw_output,
                       mu1=mu,
                       logvar1=logvar,
                       mu2=mu_prior,
                       std2=sigma_prior,
                       flag_mu1=0,
                       flag_std1=0,
                       flag_mu2=0,
                       flag_std2=0
                       )

    # standard KL implementation from original vae paper with prior N(0, 1):
    loss_ = - 0.5 * torch.mean(torch.sum(1 + logvar - mu.pow(2) - logvar.exp(), dim=-1), dim=0)

    # difference between ours and standard univariate gaussian N(0, 1), should be zero:
    print(loss-loss_) # should be zero or close, depending whether there is a numeral issue with Pytorch
    print(t)
    print('\n')

    # with dynamic prior estimated from the existing prediction, should output zero:
    # there is a numeral issue with Pytorch implementation of torch.sigmoid, that the loss2 is not gonna be exactly zero
    # but if torch.sigmoid is changed to F.sigmoid, loss2 will be exactly zero
    loss2, t = kld_loss(raw_output=raw_output,
                        mu1=mu,
                        logvar1=logvar,
                        mu2=mu_prior,
                        std2=sigma_prior,
                        flag_mu1=1,
                        flag_std1=1,
                        flag_mu2=1,
                        flag_std2=1
                        )
    print(loss2) # should be zero or close, depending whether there is a numeral issue with Pytorch
    print(t)
    print('\n')

    # with dynamic prior estimated from the existing prediction, should output zero:
    loss3, t = kld_loss(raw_output=raw_output,
                        mu1=mu,
                        logvar1=logvar,
                        mu2=mu_prior,
                        std2=sigma_prior,
                        flag_mu1=1,
                        flag_std1=2,
                        flag_mu2=1,
                        flag_std2=0
                        )
    print(loss3) # should be zero or close, depending whether there is a numeral issue with Pytorch
    print(t)
    print('\n')

    # with dynamic prior estimated from the existing prediction, should output arbitrary positive value:
    loss4, t = kld_loss(raw_output=raw_output,
                        mu1=mu,
                        logvar1=logvar,
                        mu2=mu_prior,
                        std2=sigma_prior,
                        flag_mu1=0,
                        flag_std1=0,
                        flag_mu2=1,
                        flag_std2=1
                        )
    print(loss4) # should be a positive value
    print(t)
    print('\n')




