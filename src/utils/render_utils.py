import torch.nn as nn
import torch

class Density(nn.Module):
    def __init__(self, beta=0.1):
        super().__init__()
        self.beta = nn.Parameter(torch.tensor(beta))

    def forward(self, sdf, beta=None):
        return self.density_func(sdf, beta=beta)


class LaplaceDensity(Density):  # alpha * Laplace(loc=0, scale=beta).cdf(-sdf)
    def __init__(self, beta=0.1, beta_min=0.0001):
        super().__init__(beta=beta)
        # self.beta_min = torch.tensor(beta_min).cuda()
        self.beta_min = beta_min

    def density_func(self, sdf, beta=None):
        if beta is None:
            beta = self.get_beta()

        alpha = 1 / beta
        return alpha * (0.5 + 0.5 * sdf.sign() * torch.expm1(-sdf.abs() / beta))

    def get_beta(self):
        beta = self.beta.abs() + self.beta_min
        return beta

class ModifyLaplaceDensity(Density):  # alpha * Laplace(loc=0, scale=beta).cdf(-sdf)
    def __init__(self, beta=0.1, bias=5.0, beta_min=0.0001):
        super().__init__(beta=beta)
        # self.beta_min = torch.tensor(beta_min).cuda()
        self.beta_min = beta_min
        self.bias = bias

    def density_func(self, sdf, beta=None):
        if beta is None:
            beta = self.get_beta()

        alpha = 1 / beta
        return alpha * (0.5 + 0.5 * (sdf - self.bias).sign() * torch.expm1(-(sdf - self.bias).abs() / beta))

    def get_beta(self):
        beta = self.beta.abs() + self.beta_min
        return beta

if __name__ == "__main__":
    den = ModifyLaplaceDensity(beta=0.01, bias=-1.0)
    a = torch.linspace(-2., 0., 8)
    b = den(a)
    print(1)