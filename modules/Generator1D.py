import torch.nn as nn


class Generator1D(nn.Module):
    def __init__(self, ns, challenge_bits, c_weight):
        super().__init__()
        self.challenge_bits = challenge_bits
        self.c_weight = c_weight
        self.init_dim = 17
        self.challenge = nn.Linear(
            challenge_bits, self.init_dim * c_weight
        )

        self.main = nn.Sequential(
            nn.ConvTranspose1d(
                c_weight, ns * 8, 3, 2, 1, output_padding=(1,), bias=False
            ),
            nn.BatchNorm1d(ns * 8),
            nn.LeakyReLU(),

            nn.ConvTranspose1d(
                ns * 8, ns * 4, 3, 2, 1, output_padding=(1,), bias=False
            ),
            nn.BatchNorm1d(ns * 4),
            nn.LeakyReLU(),

            nn.ConvTranspose1d(
                ns * 4, ns * 2, 3, 2, 1, output_padding=(1,), bias=False
            ),
            nn.BatchNorm1d(ns * 2),
            nn.LeakyReLU(),

            nn.ConvTranspose1d(
                ns * 2, ns, 3, 2, 0, output_padding=(1,), bias=False
            ),
            nn.BatchNorm1d(ns),
            nn.LeakyReLU(),

            nn.ConvTranspose1d(
                ns, 1, 4, 2, 1, output_padding=(1,), bias=False
            ),
            nn.Tanh()
        )

    def forward(self, challenge_input):
        challenge_input = challenge_input.view(-1, self.challenge_bits)
        challenge_input = self.challenge(challenge_input)
        challenge_input = challenge_input.view(
            -1, self.c_weight, self.init_dim
        )

        return self.main(challenge_input).squeeze()
