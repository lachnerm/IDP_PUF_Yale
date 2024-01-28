import math
import torch
import torch.nn as nn
from pytorch_lightning.core.lightning import LightningModule

from modules.GeneratorExperimental import GeneratorExperimental as GE
from modules.GeneratorExperimental import GeneratorExperimentalKyungduk as GEK
from modules.GeneratorExperimental import GeneratorExperimentalKyungdukCropped as GEKC
from modules.Generator_experimental_bitwise import \
    GeneratorExperimentalBitwise as GEB
from utils import board, utils
from utils.utils import (
    calc_nrmse, calc_fhd, dec2bin_pytorch, calc_ssim, calc_pc,
    calc_MAE
)


class PUFGenerator(LightningModule):
    def __init__(self, hparams, c_bits, name, denormalize, is_bitwise=False,
                 store_path="", do_log=True, r_bits=14, pref_bit=None):
        super().__init__()
        self.hparams.update(hparams)
        self.name = name
        self.is_bitwise = pref_bit
        self.pref_bit = 5
        self.is_single_bit = False
        self.store_path = store_path
        self.do_log = do_log

        self.challenges = []
        self.preds = []

        self.c_bits = c_bits
        self.train_log = {}
        self.val_log = {}
        self.test_log = {}

        if type(denormalize) == list:
            self.train_denormalize = denormalize[0]
            self.test_denormalize = denormalize[1]
        else:
            self.train_denormalize = denormalize
            self.test_denormalize = denormalize

        bits = int(math.sqrt(c_bits))
        if c_bits == bits ** 2:
            self.challenge_shape = (bits, bits)
        # Case with additional challenge bits containing other meta-data
        else:
            self.challenge_shape = False
        self.response_shape = (512, 512)
        self.response_shape = (128, 128)
        # self.response_shape = (48, 48)
        if is_bitwise:
            self.generator = GEB(
                hparams["ns"], c_bits, hparams["c_weight"], r_bits
            )
            self.criterion = nn.BCEWithLogitsLoss()
        else:
            # We use Kyungduk data from now on
            # self.generator = GE(hparams["ns"], c_bits, hparams["c_weight"])
            self.generator = GEK(hparams["ns"], c_bits, hparams["c_weight"])
            # self.generator = GEKC(hparams["ns"], c_bits, hparams["c_weight"])
            self.criterion = nn.MSELoss()
        self.generator.apply(utils.weights_init)
        if pref_bit is None:
            self.loss = self.reg_loss
        else:
            self.loss = self.bit_weight_loss

    def reg_loss(self, real_response, gen_response):
        loss = self.criterion(gen_response, real_response)
        return loss

    def bit_weight_loss(self, real_response, gen_response):
        loss1 = self.criterion(gen_response, real_response)
        loss2 = self.criterion(
            gen_response[:, :, :, self.pref_bit],
            real_response[:, :, :, self.pref_bit]
        )
        return loss1 + loss2

    def combined_loss(self, real_response, gen_response):
        loss1 = self.reg_loss(real_response, gen_response)
        loss2 = nn.BCEWithLogitsLoss()(
            dec2bin_pytorch(gen_response)[:, :, :, self.pref_bit].float(),
            dec2bin_pytorch(real_response)[:, :, :, self.pref_bit].float()
        )
        return loss1 + loss2

    def on_train_epoch_start(self):
        if self.do_log:
            self.grad_fig, self.gen_grad = board.create_gradient_figure(
                "Generator")

    def training_step(self, batch, batch_idx):
        challenge, real_response = batch
        gen_response = self.generator(challenge).squeeze()
        loss = self.loss(real_response, gen_response)

        if self.do_log and batch_idx == 0:
            real_response = self.train_denormalize(real_response)
            gen_response = self.train_denormalize(gen_response)

            self.train_log["challenge"] = challenge.detach()
            self.train_log["real_response"] = real_response.detach()
            self.train_log["gen_response"] = gen_response.detach()

        return {'loss': loss}

    def training_epoch_end(self, outputs):
        if self.do_log:
            gen_avg_loss = torch.stack(
                [output["loss"] for output in outputs]).mean()

            self.logger.experiment.add_figure("Gradients", self.grad_fig,
                                              self.current_epoch)
            self.logger.experiment.add_scalar("Training Loss",
                                              gen_avg_loss,
                                              self.current_epoch)

    def validation_step(self, batch, batch_idx):
        challenge, real_response = batch
        gen_response = self.generator(challenge)

        if self.is_bitwise:
            gen_response = torch.sigmoid(gen_response).round()

        real_response = self.test_denormalize(real_response)
        gen_response = self.test_denormalize(gen_response)

        if batch_idx == 0:
            self.val_log["challenge"] = challenge.detach()
            self.val_log["real_response"] = real_response.detach()
            self.val_log["gen_response"] = gen_response.detach()

        return self.calc_metrics(real_response, gen_response)

    def test_step(self, batch, batch_idx):
        challenge, real_response = batch
        gen_response = self.generator(challenge)

        if self.is_bitwise:
            gen_response = torch.sigmoid(gen_response).round()

        real_response = self.test_denormalize(real_response)
        gen_response = self.test_denormalize(gen_response)

        if self.do_log and batch_idx == 0:
            self.test_log["challenge"] = challenge
            self.test_log["real_response"] = real_response
            self.test_log["gen_response"] = gen_response

        if self.store_path:
            self.challenges.append(challenge.int())
            self.preds.append(gen_response.squeeze())

        return self.calc_metrics(real_response, gen_response)

    def validation_epoch_end(self, outputs):
        epoch = self.current_epoch
        fhd, single_bit_fhds, pc, nrmse, ssim, mae = self.get_metrics(outputs)

        self.logger.experiment.add_scalar(
            f"Validation FHD", fhd.mean(), epoch
        )
        self.logger.experiment.add_scalars(
            f"Validation Bitwise FHD", {
                name: value.mean() for name, value in single_bit_fhds.items()
            }, epoch
        )
        self.logger.experiment.add_scalar(
            f"Validation PC", pc.mean(), epoch
        )
        self.logger.experiment.add_scalar(
            f"Validation NRMSE", nrmse.mean(), epoch
        )
        self.logger.experiment.add_scalar(
            f"Validation SSIM", ssim.mean(), epoch
        )
        self.logger.experiment.add_scalar(
            f"Validation MAE", mae.mean(), epoch
        )
        self.log_gen_output_figure(self.val_log, "Validation")

    def test_epoch_end(self, outputs):
        fhd, single_bit_fhds, pc, nrmse, ssim, mae = self.get_metrics(outputs)
        bit1_fhd = single_bit_fhds["bit1"]
        bit2_fhd = single_bit_fhds["bit2"]
        bit3_fhd = single_bit_fhds["bit3"]
        bit4_fhd = single_bit_fhds["bit4"]
        bit5_fhd = single_bit_fhds["bit5"]
        self.results = {
            "FHD": fhd.tolist(),
            "FHD_bit1": bit1_fhd.tolist(),
            "FHD_bit2": bit2_fhd.tolist(),
            "FHD_bit3": bit3_fhd.tolist(),
            "FHD_bit4": bit4_fhd.tolist(),
            "FHD_bit5": bit5_fhd.tolist(),
            "PC": pc.tolist(),
            "NRMSE": nrmse.tolist(),
            "SSIM": ssim.tolist(),
            "MAE": mae.tolist()
        }

        if self.store_path:
            utils.store_preds(
                torch.cat(self.challenges),
                torch.cat(self.preds),
                self.store_path
            )

    def calc_metrics(self, real, gen):
        real = real.squeeze()
        gen = gen.squeeze()

        fhd = calc_fhd(real, gen, do_gabor=not self.is_single_bit)
        pc = calc_pc(real, gen)
        nrmse = calc_nrmse(real, gen)
        ssim = calc_ssim(real, gen)
        mae = calc_MAE(real, gen)

        if not self.is_single_bit:
            real = dec2bin_pytorch(real)
            gen = dec2bin_pytorch(gen)
            bit1_fhd = utils.calc_bit_n_FHD(real, gen, 1)
            bit2_fhd = utils.calc_bit_n_FHD(real, gen, 2)
            bit3_fhd = utils.calc_bit_n_FHD(real, gen, 3)
            bit4_fhd = utils.calc_bit_n_FHD(real, gen, 4)
            bit5_fhd = utils.calc_bit_n_FHD(real, gen, 5)
        else:
            bit1_fhd = torch.zeros(real.shape[0], device=self.device)
            bit2_fhd = bit1_fhd
            bit3_fhd = bit1_fhd
            bit4_fhd = bit1_fhd
            bit5_fhd = bit1_fhd

        return {
            'bit1_fhd': bit1_fhd,
            'bit2_fhd': bit2_fhd,
            'bit3_fhd': bit3_fhd,
            'bit4_fhd': bit4_fhd,
            'bit5_fhd': bit5_fhd,
            'pc': pc,
            'nrmse': nrmse,
            'fhd': fhd,
            'ssim': ssim,
            'mae': mae
        }

    def get_metrics(self, outputs):
        fhd = torch.cat([output["fhd"] for output in outputs])
        pc = torch.cat([output["pc"] for output in outputs])
        bit1_fhd = torch.cat([output["bit1_fhd"] for output in outputs])
        bit2_fhd = torch.cat([output["bit2_fhd"] for output in outputs])
        bit3_fhd = torch.cat([output["bit3_fhd"] for output in outputs])
        bit4_fhd = torch.cat([output["bit4_fhd"] for output in outputs])
        bit5_fhd = torch.cat([output["bit5_fhd"] for output in outputs])
        nrmse = torch.cat([output["nrmse"] for output in outputs])
        ssim = torch.cat([output["ssim"] for output in outputs])
        mae = torch.cat([output["mae"] for output in outputs])

        single_bit_fhds = {
            "bit1": bit1_fhd,
            "bit2": bit2_fhd,
            "bit3": bit3_fhd,
            "bit4": bit4_fhd,
            "bit5": bit5_fhd
        }
        return fhd, single_bit_fhds, pc, nrmse, ssim, mae

    def backward(self, trainer, loss, optimizer_idx):
        super().backward(trainer, loss, optimizer_idx)
        return
        if self.do_log and self.current_epoch % 5 == 0 and self.global_step % 10 == 0:
            board.plot_grad_flow(
                self.generator.named_parameters(), self.gen_grad
            )

    def configure_optimizers(self):
        optimizer = torch.optim.Adam(
            self.generator.parameters(),
            self.hparams.lr,
            (self.hparams.beta1, self.hparams.beta2)
        )
        return optimizer

    # Generator output figure should only be called during Validation or Testing
    def log_gen_output_figure(self, log, log_str):
        gen_output_figure = board.get_gen_output_figure_2d(log["challenge"],
                                                           log["real_response"],
                                                           log["gen_response"],
                                                           self.challenge_shape,
                                                           self.response_shape)
        self.logger.experiment.add_figure(
            f"{log_str} Real vs. Generated Output", gen_output_figure,
            self.current_epoch)
        return gen_output_figure
