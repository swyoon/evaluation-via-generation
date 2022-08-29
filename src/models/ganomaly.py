"""GANomaly
code from:
    https://github.com/samet-akcay/ganomaly/
"""
# pylint: disable=C0301,E1101,W0622,C0103,R0902,R0915

import os
import time

##
from collections import OrderedDict

import numpy as np
import torch.nn as nn
import torch.optim as optim
import torch.utils.data
import torchvision.utils as vutils
from tqdm import tqdm

from optimizers import get_optimizer


class NetD(nn.Module):
    """
    DISCRIMINATOR NETWORK
    """

    def __init__(self, encoder):
        super(NetD, self).__init__()
        model = encoder
        layers = list(model.net.children())
        self.feature = nn.Sequential(*layers[:-2])
        self.classifier = nn.Sequential(*layers[-2:])  # last layer and the sigmoid

    def forward(self, x):
        features = self.feature(x)
        classifier = self.classifier(features)
        return classifier, features


class NetG(nn.Module):
    """
    GENERATOR NETWORK
    """

    def __init__(self, encoder1, decoder, encoder2):
        super(NetG, self).__init__()
        self.encoder1 = encoder1
        self.decoder = decoder
        self.encoder2 = encoder2

    def forward(self, x):
        latent_i = self.encoder1(x)
        gen_imag = self.decoder(latent_i)
        latent_o = self.encoder2(gen_imag)
        return gen_imag, latent_i, latent_o


def weights_init(mod):
    """
    Custom weights initialization called on netG, netD and netE
    :param m:
    :return:
    """
    classname = mod.__class__.__name__
    if classname.find("Conv") != -1:
        mod.weight.data.normal_(0.0, 0.02)
    elif classname.find("BatchNorm") != -1:
        mod.weight.data.normal_(1.0, 0.02)
        mod.bias.data.fill_(0)


class BaseModel:
    """Base Model for ganomaly"""

    def __init__(self, opt, dataloader):
        ##
        # Seed for deterministic behavior
        self.seed(opt.manualseed)

        # Initalize variables.
        self.opt = opt
        self.visualizer = Visualizer(opt)
        self.dataloader = dataloader
        self.trn_dir = os.path.join(self.opt.outf, self.opt.name, "train")
        self.tst_dir = os.path.join(self.opt.outf, self.opt.name, "test")
        self.device = torch.device("cuda:0" if self.opt.device != "cpu" else "cpu")

    ##
    def set_input(self, input: torch.Tensor):
        """Set input and ground truth
        Args:
            input (FloatTensor): Input data for batch i.
        """
        with torch.no_grad():
            self.input.resize_(input[0].size()).copy_(input[0])
            self.gt.resize_(input[1].size()).copy_(input[1])
            self.label.resize_(input[1].size())

            # Copy the first batch as the fixed input.
            if self.total_steps == self.opt.batchsize:
                self.fixed_input.resize_(input[0].size()).copy_(input[0])

    ##
    def seed(self, seed_value):
        """Seed

        Arguments:
            seed_value {int} -- [description]
        """
        # Check if seed is default value
        if seed_value == -1:
            return

        # Otherwise seed all functionality
        import random

        random.seed(seed_value)
        torch.manual_seed(seed_value)
        torch.cuda.manual_seed_all(seed_value)
        np.random.seed(seed_value)
        torch.backends.cudnn.deterministic = True

    ##
    def get_errors(self):
        """Get netD and netG errors.
        Returns:
            [OrderedDict]: Dictionary containing errors.
        """

        errors = OrderedDict(
            [
                ("err_d", self.err_d.item()),
                ("err_g", self.err_g.item()),
                ("err_g_adv", self.err_g_adv.item()),
                ("err_g_con", self.err_g_con.item()),
                ("err_g_enc", self.err_g_enc.item()),
            ]
        )

        return errors

    ##
    def get_current_images(self):
        """Returns current images.
        Returns:
            [reals, fakes, fixed]
        """

        reals = self.input.data
        fakes = self.fake.data
        fixed = self.netg(self.fixed_input)[0].data

        return reals, fakes, fixed

    ##
    def save_weights(self, epoch):
        """Save netG and netD weights for the current epoch.
        Args:
            epoch ([int]): Current epoch number.
        """

        weight_dir = os.path.join(self.opt.outf, self.opt.name, "train", "weights")
        if not os.path.exists(weight_dir):
            os.makedirs(weight_dir)

        torch.save(
            {"epoch": epoch + 1, "state_dict": self.netg.state_dict()},
            "%s/netG.pth" % (weight_dir),
        )
        torch.save(
            {"epoch": epoch + 1, "state_dict": self.netd.state_dict()},
            "%s/netD.pth" % (weight_dir),
        )

    ##
    def train_one_epoch(self):
        """Train the model for one epoch."""

        self.netg.train()
        epoch_iter = 0
        for data in tqdm(
            self.dataloader["train"], leave=False, total=len(self.dataloader["train"])
        ):
            self.total_steps += self.opt.batchsize
            epoch_iter += self.opt.batchsize

            self.set_input(data)
            # self.optimize()
            self.optimize_params()

            if self.total_steps % self.opt.print_freq == 0:
                errors = self.get_errors()
                if self.opt.display:
                    counter_ratio = float(epoch_iter) / len(
                        self.dataloader["train"].dataset
                    )
                    self.visualizer.plot_current_errors(
                        self.epoch, counter_ratio, errors
                    )

            if self.total_steps % self.opt.save_image_freq == 0:
                reals, fakes, fixed = self.get_current_images()
                self.visualizer.save_current_images(self.epoch, reals, fakes, fixed)
                if self.opt.display:
                    self.visualizer.display_current_images(reals, fakes, fixed)

        print(
            ">> Training model %s. Epoch %d/%d"
            % (self.name, self.epoch + 1, self.opt.niter)
        )
        # self.visualizer.print_current_errors(self.epoch, errors)

    ##
    def train(self):
        """Train the model"""

        ##
        # TRAIN
        self.total_steps = 0
        best_auc = 0

        # Train for niter epochs.
        print(">> Training model %s." % self.name)
        for self.epoch in range(self.opt.iter, self.opt.niter):
            # Train for one epoch
            self.train_one_epoch()
            res = self.test()
            if res[self.opt.metric] > best_auc:
                best_auc = res[self.opt.metric]
                self.save_weights(self.epoch)
            self.visualizer.print_current_performance(res, best_auc)
        print(">> Training model %s.[Done]" % self.name)

    ##
    def test(self):
        """Test GANomaly model.
        Args:
            dataloader ([type]): Dataloader for the test set
        Raises:
            IOError: Model weights not found.
        """
        with torch.no_grad():
            # Load the weights of netg and netd.
            if self.opt.load_weights:
                path = "./output/{}/{}/train/weights/netG.pth".format(
                    self.name.lower(), self.opt.dataset
                )
                pretrained_dict = torch.load(path)["state_dict"]

                try:
                    self.netg.load_state_dict(pretrained_dict)
                except IOError:
                    raise IOError("netG weights not found")
                print("   Loaded weights.")

            self.opt.phase = "test"

            # Create big error tensor for the test set.
            self.an_scores = torch.zeros(
                size=(len(self.dataloader["test"].dataset),),
                dtype=torch.float32,
                device=self.device,
            )
            self.gt_labels = torch.zeros(
                size=(len(self.dataloader["test"].dataset),),
                dtype=torch.long,
                device=self.device,
            )
            self.latent_i = torch.zeros(
                size=(len(self.dataloader["test"].dataset), self.opt.nz),
                dtype=torch.float32,
                device=self.device,
            )
            self.latent_o = torch.zeros(
                size=(len(self.dataloader["test"].dataset), self.opt.nz),
                dtype=torch.float32,
                device=self.device,
            )

            # print("   Testing model %s." % self.name)
            self.times = []
            self.total_steps = 0
            epoch_iter = 0
            for i, data in enumerate(self.dataloader["test"], 0):
                self.total_steps += self.opt.batchsize
                epoch_iter += self.opt.batchsize
                time_i = time.time()
                self.set_input(data)
                self.fake, latent_i, latent_o = self.netg(self.input)

                error = torch.mean(torch.pow((latent_i - latent_o), 2), dim=1)
                time_o = time.time()

                self.an_scores[
                    i * self.opt.batchsize : i * self.opt.batchsize + error.size(0)
                ] = error.reshape(error.size(0))
                self.gt_labels[
                    i * self.opt.batchsize : i * self.opt.batchsize + error.size(0)
                ] = self.gt.reshape(error.size(0))
                self.latent_i[
                    i * self.opt.batchsize : i * self.opt.batchsize + error.size(0), :
                ] = latent_i.reshape(error.size(0), self.opt.nz)
                self.latent_o[
                    i * self.opt.batchsize : i * self.opt.batchsize + error.size(0), :
                ] = latent_o.reshape(error.size(0), self.opt.nz)

                self.times.append(time_o - time_i)

                # Save test images.
                if self.opt.save_test_images:
                    dst = os.path.join(self.opt.outf, self.opt.name, "test", "images")
                    if not os.path.isdir(dst):
                        os.makedirs(dst)
                    real, fake, _ = self.get_current_images()
                    vutils.save_image(
                        real, "%s/real_%03d.eps" % (dst, i + 1), normalize=True
                    )
                    vutils.save_image(
                        fake, "%s/fake_%03d.eps" % (dst, i + 1), normalize=True
                    )

            # Measure inference time.
            self.times = np.array(self.times)
            self.times = np.mean(self.times[:100] * 1000)

            # Scale error vector between [0, 1]
            self.an_scores = (self.an_scores - torch.min(self.an_scores)) / (
                torch.max(self.an_scores) - torch.min(self.an_scores)
            )
            # auc, eer = roc(self.gt_labels, self.an_scores)
            auc = evaluate(self.gt_labels, self.an_scores, metric=self.opt.metric)
            performance = OrderedDict(
                [("Avg Run Time (ms/batch)", self.times), (self.opt.metric, auc)]
            )

            if self.opt.display_id > 0 and self.opt.phase == "test":
                counter_ratio = float(epoch_iter) / len(self.dataloader["test"].dataset)
                self.visualizer.plot_performance(self.epoch, counter_ratio, performance)
            return performance


##
class Ganomaly(nn.Module):
    """GANomaly Class"""

    def __init__(
        self, encoder1, decoder, encoder2, discriminator, w_adv=1, w_con=50, w_enc=1
    ):
        super().__init__()
        # Create and initialize networks.
        self.netg = NetG(encoder1, decoder, encoder2)
        self.netd = NetD(discriminator)
        # self.netg.apply(weights_init)
        # self.netd.apply(weights_init)

        self.l_adv = nn.MSELoss()
        self.l_con = nn.L1Loss()
        self.l_enc = nn.MSELoss()
        self.l_bce = nn.BCELoss()

        self.w_adv = w_adv
        self.w_con = w_con
        self.w_enc = w_enc
        self.own_optimizer = True
        #     self.optimizer_d = optim.Adam(self.netd.parameters(), lr=self.opt.lr, betas=(self.opt.beta1, 0.999))
        #     self.optimizer_g = optim.Adam(self.netg.parameters(), lr=self.opt.lr, betas=(self.opt.beta1, 0.999))

    def get_optimizer(self, opt_cfg):
        opt_g, sch_g = get_optimizer(opt_cfg["g"], self.netg.parameters())
        opt_d, sch_d = get_optimizer(opt_cfg["d"], self.netd.parameters())
        d_opt = {"g": opt_g, "d": opt_d}
        return d_opt, {"g": sch_g, "d": sch_d}

    def backward_g(self, x, fake, latent_o, latent_i):
        """Backpropagate through netG"""
        err_g_adv = self.l_adv(self.netd(x)[1], self.netd(x)[1])
        err_g_con = self.l_con(fake, x)
        err_g_enc = self.l_enc(latent_o, latent_i)
        err_g = err_g_adv * self.w_adv + err_g_con * self.w_con + err_g_enc * self.w_enc
        err_g.backward(retain_graph=True)
        return err_g

    def backward_d(self, pred_real, pred_fake, real_label, fake_label):
        """Backpropagate through netD"""
        # Real - Fake Loss
        err_d_real = self.l_bce(pred_real.flatten(), real_label)
        err_d_fake = self.l_bce(pred_fake.flatten(), fake_label)

        # NetD Loss & Backward-Pass
        err_d = (err_d_real + err_d_fake) * 0.5
        err_d.backward()
        return err_d

    def reinit_d(self):
        """Re-initialize the weights of netD"""
        for layer in self.netd.feature.children():
            if hasattr(layer, "reset_parameters"):
                layer.reset_parameters()
        for layer in self.netd.classifier.children():
            if hasattr(layer, "reset_parameters"):
                layer.reset_parameters()
        print("   Reloading net d")

    def train_step(self, x, optimizer, y=None):
        """Forwardpass, Loss Computation and Backwardpass."""
        optimizer_g, optimizer_d = optimizer["g"], optimizer["d"]
        # Forward-pass
        fake, latent_i, latent_o = self.netg(x)
        pred_real, feat_real = self.netd(x)
        pred_fake, feat_fake = self.netd(fake.detach())
        real_label = torch.ones((len(x),), dtype=torch.float32, device=x.device)
        fake_label = torch.zeros((len(x),), dtype=torch.float32, device=x.device)

        # Backward-pass
        # netg
        optimizer_g.zero_grad()
        err_g = self.backward_g(x, fake, latent_o, latent_i)
        optimizer_g.step()

        # netd
        optimizer_d.zero_grad()
        err_d = self.backward_d(pred_real, pred_fake, real_label, fake_label)
        optimizer_d.step()

        # this part is not mentioned in the original paper, but seems crucial in training.
        if err_d.item() < 1e-5:
            self.reinit_d()

        d_result = {
            "loss": err_g.item() + err_d.item(),
            "ganomaly/err_g_": err_g.item(),
            "ganomaly/err_d_": err_d.item(),
        }
        return d_result

    def forward(self, x):
        fake, latent_i, latent_o = self.netg(x)
        error = torch.mean(torch.pow((latent_i - latent_o), 2).view(len(x), -1), dim=1)
        return error

    def predict(self, x):
        return self.forward(x)

    def validation_step(self, x, y=None):
        d_result = {"loss": 0.0}
        return d_result
