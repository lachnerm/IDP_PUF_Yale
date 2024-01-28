import torch.nn as nn


class GeneratorExperimental(nn.Module):
    def __init__(self, ns, c_bits, c_weight):
        super().__init__()
        self.c_bits = c_bits
        self.c_weight = c_weight
        self.init_dim = 64
        self.challenge = nn.Linear(
            c_bits, self.init_dim ** 2 * c_weight
        )

        self.main = nn.Sequential(
            nn.ConvTranspose2d(
                c_weight, ns * 8, 3, 2, 1, output_padding=(1,), bias=False
            ),
            nn.BatchNorm2d(ns * 8),
            nn.GELU(),

            nn.ConvTranspose2d(ns * 8, ns * 4, 3, 1, 1, bias=False),
            nn.BatchNorm2d(ns * 4),
            nn.GELU(),

            nn.ConvTranspose2d(
                ns * 4, ns * 2, 3, 2, 1, output_padding=(1,), bias=False
            ),
            nn.BatchNorm2d(ns * 2),
            nn.GELU(),

            nn.ConvTranspose2d(ns * 2, ns, 3, 1, 1, bias=False),
            nn.BatchNorm2d(ns),
            nn.GELU(),

            nn.ConvTranspose2d(ns, 1, 3, 2, 1, output_padding=(1,), bias=False),
            nn.Tanh()
        )

    def forward(self, c_input):
        c_input = c_input.view(-1, self.c_bits)
        c_input = self.challenge(c_input)
        c_input = c_input.view(-1, self.c_weight, self.init_dim, self.init_dim)

        return self.main(c_input).squeeze(dim=1)


class GeneratorExperimentalKyungduk(nn.Module):
    # Creates speckle patterns of size 128x128
    def __init__(self, ns, c_bits, c_weight):
        super().__init__()
        self.c_bits = c_bits
        self.c_weight = c_weight
        self.init_dim = 16
        self.challenge = nn.Linear(
            c_bits, self.init_dim ** 2 * c_weight
        )

        self.main = nn.Sequential(
            nn.ConvTranspose2d(
                c_weight, ns * 4, 3, 2, 1, output_padding=(1,), bias=False
            ),
            nn.BatchNorm2d(ns * 4),
            nn.GELU(),

            nn.ConvTranspose2d(
                ns * 4, ns * 2, 3, 2, 1, output_padding=(1,), bias=False
            ),
            nn.BatchNorm2d(ns * 2),
            nn.GELU(),

            nn.ConvTranspose2d(ns * 2, 1, 3, 2, 1, output_padding=(1,), bias=False),
            nn.Tanh()
        )

    def forward(self, c_input):
        c_input = c_input.view(-1, self.c_bits)
        c_input = self.challenge(c_input)
        c_input = c_input.view(-1, self.c_weight, self.init_dim, self.init_dim)
        out = self.main(c_input)

        return out.squeeze()


class GeneratorExperimentalKyungdukCropped(nn.Module):
    # Creates speckle patterns of size 128x128
    def __init__(self, ns, c_bits, c_weight):
        super().__init__()
        self.c_bits = c_bits
        self.c_weight = c_weight
        self.init_dim = 12
        self.challenge = nn.Linear(
            c_bits, self.init_dim ** 2 * c_weight
        )

        self.main = nn.Sequential(
            nn.ConvTranspose2d(
                c_weight, ns * 4, 3, 2, 1, output_padding=(1,), bias=False
            ),
            nn.BatchNorm2d(ns * 4),
            nn.GELU(),

            nn.ConvTranspose2d(ns * 4, ns * 2, 3, 1, 1, bias=False),
            nn.BatchNorm2d(ns * 2),
            nn.GELU(),

            nn.ConvTranspose2d(ns * 2, 1, 3, 2, 1, output_padding=(1,), bias=False),
            nn.Tanh()
        )

    def forward(self, c_input):
        c_input = c_input.view(-1, self.c_bits)
        c_input = self.challenge(c_input)
        c_input = c_input.view(-1, self.c_weight, self.init_dim, self.init_dim)
        out = self.main(c_input)

        return out.squeeze()