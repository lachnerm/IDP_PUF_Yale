import torch
import torch.nn as nn
from pytorch_lightning.core.lightning import LightningModule

from modules.Generator1D import Generator1D
from utils import board, utils
from utils.utils import calc_pc, calc_nrmse


class PUFGenerator1D(LightningModule):
    def __init__(self, hparams, challenge_bits, name, denormalize,
                 store_path="", do_log=True):
        super().__init__()
        hparams["activation"] = "LeakyReLU"
        self.hparams.update(hparams)
        self.name = name
        self.store_path = store_path
        self.do_log = do_log

        self.challenges = []
        self.preds = []

        self.challenge_bits = challenge_bits
        self.train_log = {}
        self.val_log = {}
        self.test_log = {}

        self.denormalize = denormalize

        self.challenge_shape = (4, 3)
        self.response_size = 549
        self.response_shape = (23, 24)

        self.generator = Generator1D(self.hparams.ns, challenge_bits,
                                     self.hparams.c_weight)
        self.generator.apply(utils.weights_init)
        self.save_hyperparameters()

    def loss_function(self, real_response, gen_response):
        criterion = nn.L1Loss()
        loss = criterion(real_response, gen_response)
        return loss

    def on_train_epoch_start(self):
        if self.do_log:
            self.grad_fig, self.gen_grad = board.create_gradient_figure(
                "Generator")

    def training_step(self, batch, batch_idx):
        challenge, real_response = batch
        gen_response = self.generator(challenge)
        loss = self.loss_function(real_response, gen_response)

        if self.do_log and batch_idx == 0:
            real_response = self.denormalize(real_response)
            gen_response = self.denormalize(gen_response)

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

        real_response = self.denormalize(real_response)
        gen_response = self.denormalize(gen_response)

        if batch_idx == 0:
            self.val_log["challenge"] = challenge.detach()
            self.val_log["real_response"] = real_response.detach()
            self.val_log["gen_response"] = gen_response.detach()

        return self.calc_metrics(real_response, gen_response)

    def test_step(self, batch, batch_idx):
        challenge, real_response = batch
        gen_response = self.generator(challenge)

        real_response = self.denormalize(real_response)
        gen_response = self.denormalize(gen_response)

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
        pc, nrmse, abs = self.get_metrics(outputs)

        self.logger.experiment.add_scalar(
            f"Validation NRMSE (euclidean)", nrmse.mean(), epoch
        )
        self.logger.experiment.add_scalar(
            f"Validation PC", pc.mean(), epoch
        )
        self.logger.experiment.add_scalar(
            f"Validation Abs", abs.mean(), epoch
        )
        self.log_gen_output_figure(self.val_log, "Validation")

    def test_epoch_end(self, outputs):
        pc, nrmse, abs = self.get_metrics(outputs)
        self.results = {
            "PC": pc.tolist(),
            "NRMSE": nrmse.tolist(),
            "Abs": abs.tolist()
        }

        if self.store_path:
            utils.store_preds(
                torch.cat(self.challenges),
                torch.cat(self.preds),
                self.store_path
            )

    def calc_metrics(self, real, gen):
        pc = calc_pc(real, gen)
        nrmse = calc_nrmse(real, gen)
        abs = torch.cdist(real, gen, p=2.0).flatten()
        return {
            'pc': pc,
            'nrmse': nrmse,
            'abs': abs
        }

    def get_metrics(self, outputs):
        pc = torch.cat([output["pc"] for output in outputs])
        nrmse = torch.cat([output["nrmse"] for output in outputs])
        abs = torch.cat([output["abs"] for output in outputs])

        return pc, nrmse, abs

    def backward(self, trainer, loss, optimizer_idx):
        super().backward(trainer, loss, optimizer_idx)
        if self.do_log and self.trainer.global_step % 20 == 0:
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
        gen_output_figure = board.get_gen_output_figure_1d(log["challenge"],
                                                           log["real_response"],
                                                           log["gen_response"],
                                                           self.challenge_shape,
                                                           self.response_shape)
        self.logger.experiment.add_figure(
            f"{log_str} Real vs. Generated Output", gen_output_figure,
            self.current_epoch)
        return gen_output_figure
