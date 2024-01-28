import torch.nn as nn


class GeneratorExperimentalBitwise(nn.Module):
    def __init__(self, ns, c_bits, c_weight, r_bits):
        super().__init__()
        self.challenge_bits = c_bits
        self.c_weight = c_weight
        self.r_bits = r_bits
        self.init_dim = 32
        self.challenge = nn.Linear(c_bits, self.init_dim ** 2 * self.c_weight)

        self.main = nn.Sequential(
            nn.ConvTranspose2d(
                c_weight, ns * 8, 3, 2, 1, output_padding=(1,), bias=False
            ),
            nn.BatchNorm2d(ns * 8),
            nn.GELU(),

            nn.ConvTranspose2d(
                ns * 8, ns * 8, 3, 2, 1, output_padding=(1,), bias=False
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

            nn.ConvTranspose2d(
                ns, self.r_bits, 3, 2, 1, output_padding=(1,), bias=False
            ),
        )

    def forward(self, c_input):
        c_input = c_input.view(-1, self.challenge_bits)
        c_input = self.challenge(c_input)
        c_input = c_input.view(-1, self.c_weight, self.init_dim, self.init_dim)

        return self.main(c_input).transpose(1, 3)
