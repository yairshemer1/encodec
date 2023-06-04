# Copyright (c) Facebook, Inc. and its affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.
# author: adiyoss
import random

import wandb
import json
import logging
from pathlib import Path
import os
import time

import torch
import torch.nn.functional as F

from . import augment, distrib, pretrained
from .balancer import Balancer
from .evaluate import evaluate
from .model import discriminator_loss, feature_loss, generator_loss
from .stft_loss import MultiResolutionMelLoss
from .utils import bold, copy_state, pull_metric, serialize_model, swap_state, LogProgress

logger = logging.getLogger(__name__)


class Solver(object):
    def __init__(self, data, model, msd, optimizer_gen, optimizer_disc, args):
        self.tr_loader = data['tr_loader']
        self.cv_loader = data['cv_loader']
        self.tt_loader = data['tt_loader']
        self.model = model
        self.msd = msd
        self.dmodel = distrib.wrap(model)
        self.balancer = Balancer(weights={"l_t_wave": args.l_t_wave,
                                          "l_f_mel": args.l_f_mel,
                                          "l_feat_features": args.l_feat_features,
                                          "l_g_gen": args.l_g_gen})
        self.optimizer_gen = optimizer_gen
        self.optimizer_disc = optimizer_disc
        self.scheduler_gen = torch.optim.lr_scheduler.ExponentialLR(optimizer_gen, gamma=args.lr_decay, last_epoch=-1)
        self.scheduler_disc = torch.optim.lr_scheduler.ExponentialLR(optimizer_disc, gamma=args.lr_decay, last_epoch=-1)

        # Training config
        self.device = args.device
        self.epochs = args.epochs

        # Checkpoints
        self.continue_from = args.continue_from
        self.eval_every = args.eval_every
        self.checkpoint = args.checkpoint
        if self.checkpoint:
            self.checkpoint_file = Path(args.checkpoint_file)
            self.best_file = Path(args.best_file)
            logger.debug("Checkpoint will be saved to %s", self.checkpoint_file.resolve())
        self.history_file = args.history_file

        self.best_state = None
        self.restart = args.restart
        self.history = []  # Keep track of loss
        self.samples_dir = args.samples_dir  # Where to save samples
        self.num_prints = args.num_prints  # Number of times to log per epoch
        self.args = args
        self.multi_res_mel_loss = MultiResolutionMelLoss().to(self.device)
        self._reset()

    def _serialize(self):
        package = {}
        package['model'] = serialize_model(self.model)
        package['msd'] = serialize_model(self.msd)
        package['optimizer_gen'] = self.optimizer_gen.state_dict()
        package['optimizer_disc'] = self.optimizer_disc.state_dict()
        package['scheduler_gen'] = self.scheduler_gen.state_dict()
        package['scheduler_disc'] = self.scheduler_disc.state_dict()
        package['history'] = self.history
        package['best_state'] = self.best_state
        package['args'] = self.args
        tmp_path = str(self.checkpoint_file) + ".tmp"
        torch.save(package, tmp_path)
        # renaming is sort of atomic on UNIX (not really true on NFS)
        # but still less chances of leaving a half written checkpoint behind.
        os.rename(tmp_path, self.checkpoint_file)

        # Saving only the latest best model.
        model = package['model']
        model['state'] = self.best_state
        tmp_path = str(self.best_file) + ".tmp"
        torch.save(model, tmp_path)
        os.rename(tmp_path, self.best_file)

    def _reset(self):
        """_reset."""
        load_from = None
        load_best = False
        keep_history = True
        # Reset
        if self.checkpoint and self.checkpoint_file.exists() and not self.restart:
            load_from = self.checkpoint_file
        elif self.continue_from:
            load_from = self.continue_from
            load_best = self.args.continue_best
            keep_history = False

        if load_from:
            logger.info(f'Loading checkpoint model: {load_from}')
            package = torch.load(load_from, 'cpu')
            if load_best:
                self.msd.load_state_dict(package['msd'])
                self.model.load_state_dict(package['best_state'])
            else:
                self.msd.load_state_dict(package['msd']['state'])
                self.model.load_state_dict(package['model']['state'])
            if 'optimizer_gen' in package and 'optimizer_disc' in package and not load_best:
                self.optimizer_gen.load_state_dict(package['optimizer_gen'])
                self.optimizer_disc.load_state_dict(package['optimizer_disc'])
                self.scheduler_gen.load_state_dict(package['scheduler_gen'])
                self.scheduler_disc.load_state_dict(package['scheduler_disc'])
            if keep_history:
                self.history = package['history']
            self.best_state = package['best_state']
        continue_pretrained = self.args.continue_pretrained
        if continue_pretrained:
            logger.info("Fine tuning from pre-trained model %s", continue_pretrained)
            model = getattr(pretrained, self.args.continue_pretrained)()
            self.model.load_state_dict(model.state_dict())

    def train(self):
        if self.args.save_again:
            self._serialize()
            return
        # Optimizing the model
        if self.history:
            logger.info("Replaying metrics from previous run")
        for epoch, metrics in enumerate(self.history):
            info = " ".join(f"{k.capitalize()}={v:.5f}" for k, v in metrics.items())
            logger.info(f"Epoch {epoch + 1}: {info}")

        for epoch in range(len(self.history), self.epochs):
            # Train one epoch
            self.model.train()
            self.msd.train()
            start = time.time()
            logger.info('-' * 70)
            logger.info("Training...")
            train_loss = self._run_one_epoch(epoch)
            logger.info(
                bold(f'Train Summary | End of Epoch {epoch + 1} | '
                     f'Time {time.time() - start:.2f}s | Train Loss {train_loss:.5f}'))

            if self.cv_loader:
                # Cross validation
                logger.info('-' * 70)
                logger.info('Cross validation...')
                self.model.eval()
                with torch.no_grad():
                    valid_loss = self._run_one_epoch(epoch, cross_valid=True)
                logger.info(
                    bold(f'Valid Summary | End of Epoch {epoch + 1} | '
                         f'Time {time.time() - start:.2f}s | Valid Loss {valid_loss:.5f}'))
            else:
                valid_loss = 0

            best_loss = min(pull_metric(self.history, 'valid') + [valid_loss])
            metrics = {'train': train_loss, 'valid': valid_loss, 'best': best_loss,
                       'lr_gen': self.scheduler_gen.get_last_lr()[0], 'lr_disc': self.scheduler_disc.get_last_lr()[0]}
            # Save the best model
            if valid_loss == best_loss:
                logger.info(bold('New best valid loss %.4f'), valid_loss)
                self.best_state = copy_state(self.model.state_dict())

            # evaluate and enhance samples every 'eval_every' argument number of epochs
            # also evaluate on last epoch
            self.tt_loader.epoch = epoch
            if ((epoch + 1) % self.eval_every == 0 or epoch == self.epochs - 1) and self.tt_loader:
                # Evaluate on the testset
                logger.info('-' * 70)
                logger.info('Evaluating on the test set...')
                # We switch to the best known model for testing
                with swap_state(self.model, self.best_state):
                    evaluate(args=self.args, model=self.model, data_loader=self.tt_loader)

            if self.args.wandb:
                wandb.log(metrics, step=epoch)
            self.history.append(metrics)
            info = " | ".join(f"{k.capitalize()} {v:.05g}" for k, v in metrics.items())
            logger.info('-' * 70)
            logger.info(bold(f"Overall Summary | Epoch {epoch + 1} | {info}"))

            if distrib.rank() == 0:
                json.dump(self.history, open(self.history_file, "w"), indent=2)
                # Save model each epoch
                if self.checkpoint:
                    self._serialize()
                    logger.debug("Checkpoint saved to %s", self.checkpoint_file.resolve())

            self.scheduler_disc.step()
            self.scheduler_gen.step()

    def _run_one_epoch(self, epoch, cross_valid=False):
        total_loss = 0
        data_loader = self.tr_loader if not cross_valid else self.cv_loader

        data_loader.epoch = epoch
        label = ["Train", "Valid"][cross_valid]
        name = label + f" | Epoch {epoch + 1}"
        logprog = LogProgress(logger, data_loader, updates=self.num_prints, name=name)
        for i, data in enumerate(logprog):
            y = data.to(self.device)
            y_pred = self.dmodel(y)
            # apply a loss function after each layer
            with torch.autograd.set_detect_anomaly(True):
                y_pred_detach = torch.clone(y_pred).detach()
                disc_loss = self.disc_step(y=y, y_pred=y_pred_detach, cross_valid=cross_valid, epoch=epoch)
                gen_loss = self.generator_step(y=y, y_pred=y_pred, cross_valid=cross_valid, epoch=epoch)
                total_loss = gen_loss + disc_loss

            logprog.update(loss=format(total_loss / (i + 1), ".5f"))

        return distrib.average([total_loss / (i + 1)], i + 1)[0]

    def generator_step(self, y, y_pred, epoch, cross_valid=False):
        mel_loss, l1_mel_loss, l2_mel_loss = self.multi_res_mel_loss(y_pred.squeeze(1), y.squeeze(1))
        wav_loss = F.l1_loss(y_pred, y)
        signal_loss = wav_loss + mel_loss
        y_ds_hat_r, y_ds_hat_g, fmap_s_r, fmap_s_g = self.msd(y, y_pred)
        loss_fm_s = feature_loss(fmap_s_r, fmap_s_g)
        loss_gen_s = generator_loss(y_ds_hat_g)
        loss_gen_all = loss_gen_s + loss_fm_s + signal_loss
        losses = {"l_t_wave": wav_loss,
                  "l_f_mel": mel_loss,
                  "l_feat_features": loss_fm_s,
                  "l_g_gen": loss_gen_s,}
        if self.args.wandb:
            wandb.log({"Mel loss": mel_loss,
                       "Wave loss": wav_loss,
                       "Feature loss": loss_fm_s,
                       "Generator loss": loss_gen_s}, step=epoch)
        if not cross_valid:
            self.optimizer_gen.zero_grad()
            self.balancer.backward(losses=losses, input=y_pred)
            self.optimizer_gen.step()

        return loss_gen_all.item()

    def disc_step(self, y, y_pred, epoch, cross_valid=False):
        y_ds_hat_r, y_ds_hat_g, _, _ = self.msd(y, y_pred)
        loss_disc_s, losses_disc_s_r, losses_disc_s_g = discriminator_loss(y_ds_hat_r, y_ds_hat_g)
        loss_disc_s *= self.args.l_d_disc

        if not cross_valid:
            prob = random.random()
            if prob < self.args.disc_step_prob:
                self.optimizer_disc.zero_grad()
                loss_disc_s.backward()
                self.optimizer_disc.step()

        if self.args.wandb:
            wandb.log({"Discriminator loss": loss_disc_s}, step=epoch)
        return loss_disc_s.item()
